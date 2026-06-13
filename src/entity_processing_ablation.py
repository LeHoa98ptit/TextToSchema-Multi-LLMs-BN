"""
Ablation Study — Entity Processing
====================================
Entity processing is UNCHANGED in this ablation because it relies solely on
text-based signals (SBERT, TF-IDF, spaCy) and does NOT use Wikidata at all.

This file is a verbatim copy of src/entity_processing.py placed inside the
ablation package so the ablation pipeline is fully self-contained and does not
depend on the original src/ module.

What changed vs. the original:
  - NOTHING in terms of logic.
  - Module path changed: src.entity_processing → ablation.src.entity_processing_ablation
"""

import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)

import os
import re
import ssl
import json
import math
import string
import logging
import itertools
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import nltk
import numpy as np
import spacy
import matplotlib.pyplot as plt

from openie import StanfordOpenIE
from g4f.client import Client
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# =====================================================
# PROMPTING
# =====================================================

def entity_prompting_without_description_1(processed_data):
    """
    Zero-shot prompt: extract candidate entities for an ER model.
    Returns a formatted f-string ready for LLM input.
    """
    safe_data = processed_data.replace("{", "{{").replace("}", "}}")
    prompt = f"""
    Extract candidate ENTITIES from the text below for an Entity-Relationship (ER) model.

    Original Description:
    {safe_data}

    Rules:
    1. Do NOT extract attributes (e.g., names, ids, emails) as entities.
    2. Do NOT invent entities not mentioned explicitly in the text.
    3. Only extract entities explicitly present in the text.
    4. Ignore dataset-level, schema-level, or metadata terms (dataset, table, column, file, etc.).
    5. Do NOT extract optional or detail entities not central to the main ER model, such as payments,
       accounts, contracts, security deposits, maintenance records, or lease extensions—unless they are
       needed for an explicit associative relationship.
    6. Use UPPER_SNAKE_CASE for entity names.
    7. Separate core entities from associative entities only if it helps clarity.

    Output:
    Return ONLY a JSON array with key "entities".

    Example:
    {{
        "entities": []
    }}
    """
    return prompt


def entity_few_shot_prompting_without_description_1(processed_data):
    """
    Few-shot prompt: extract candidate entities including process/event entities.
    """
    safe_data = processed_data.replace("{", "{{").replace("}", "}}")
    prompt = f"""
    Extract candidate ENTITIES from the text below for an Entity-Relationship (ER) model.

    Original Description:
    {safe_data}

    Rules:
    1. Include BOTH physical entities (people, objects) AND process/event entities
       (services, sessions, registrations, transactions, participations).
       - A process/event entity is needed when an activity has its OWN attributes.
       - Do NOT replace a process/event entity with a direct relationship between two physical entities.
    2. Do NOT include any entity that is a description of the system, organization, or dataset itself.
    3. Do NOT extract subtypes, variations, or descriptive forms of an entity.
    4. Do NOT extract attributes (e.g., names, ids, emails) as entities.
    5. Ignore dataset-level, schema-level, or metadata terms (dataset, table, column, file, etc.).
    6. Use UPPER_SNAKE_CASE for entity names.

    Output:
    Return ONLY a JSON object with key "entities".

    Example 1 — system with a process/event entity:
    Description: A pet grooming salon manages groomers and pets. Each grooming service has a duration, pricing, and customer rating.
    Generated entities:
    {{
        "entities": ["GROOMER", "PET", "GROOMING_SERVICE"]
    }}
    Explanation: GROOMING_SERVICE is a process entity because it has its own attributes (duration, pricing, rating).

    Example 2 — system with standard entities:
    Description: A hospital manages medical appointments. For each appointment the dataset stores patient info
    (name, date of birth, gender, phone), doctor info (name, specialty, license number), the appointment date
    and time, and diagnosis notes.
    Generated entities:
    {{
        "entities": ["PATIENT", "DOCTOR", "APPOINTMENT"]
    }}
    """
    return prompt


# =====================================================
# LLM OUTPUT PARSING
# =====================================================

def extract_entity_names(llm_output: str) -> list:
    """Extract entity names from a ```json ... ``` block in LLM output."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", llm_output, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in LLM output.")
    json_str = match.group(1)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nOriginal JSON:\n{json_str}")
    return [e["name"] for e in data.get("entities", [])]


def extract_entities_from_text_1(raw_text: str) -> list:
    """Extract entities from raw text containing a ```json ... ``` block."""
    json_match = re.search(r"```json(.*?)```", raw_text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON block found in the text.")
    data = json.loads(json_match.group(1).strip())
    entities = []
    for ent in data.get("entities", []):
        if isinstance(ent, dict) and "name" in ent:
            entities.append(ent["name"])
        elif isinstance(ent, str):
            entities.append(ent)
    return entities


def extract_llama_entities_from_text(raw_text: str) -> list:
    """
    Extract entity names from LLM output.
    Handles LLAMA-style (```json``` block), GPT-style (direct JSON),
    or raw text with explanation preceding the JSON object.
    """
    logger = logging.getLogger(__name__)

    # Try ```json ... ``` block first
    json_match = re.search(r"```json(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    if json_match:
        json_text = json_match.group(1).strip()
    else:
        # Fallback: find outermost { ... }
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            json_text = match.group()
        else:
            logger.warning("No JSON object found in LLM response, returning empty list")
            return []

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(f"Cannot parse JSON: {e}\nRaw text: {raw_text}")
        return []

    entities = []
    for ent in data.get("entities", []):
        if isinstance(ent, dict) and "name" in ent:
            entities.append(ent["name"])
        elif isinstance(ent, str):
            entities.append(ent)
    return entities


def extract_gpt_entities(text: str) -> list:
    """Extract entities from GPT-style LLM output (raw JSON or ```json``` block)."""
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    json_text = match.group(1).strip() if match else text.strip()
    data = json.loads(json_text)
    entities = []
    for ent in data.get("entities", []):
        if isinstance(ent, dict) and "name" in ent:
            entities.append(ent["name"])
        elif isinstance(ent, str):
            entities.append(ent)
    return entities


def merge_entity_lists(list1: list, list2: list) -> list:
    """Merge two entity lists, removing duplicates while preserving order."""
    merged = []
    seen = set()
    for entity in list1 + list2:
        if entity not in seen:
            merged.append(entity)
            seen.add(entity)
    return merged


# =====================================================
# TEXT PREPROCESSING & TF-IDF KEYWORDS
# =====================================================

def preprocess_text(text: str) -> str:
    """Lower-case, remove special characters, remove English stop words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)


def extract_keywords_tfidf(text: str, top_n: int = 30) -> list:
    """Extract top-n TF-IDF keywords (unigrams + bigrams) from text."""
    processed_text = preprocess_text(text)
    vectorizer = TfidfVectorizer(max_features=top_n, ngram_range=(1, 2))
    vectorizer.fit_transform([processed_text])
    return list(vectorizer.get_feature_names_out())


# =====================================================
# SBERT COSINE SIMILARITY
# =====================================================

def cosine_similarity_score(list1: list, list2: list) -> np.ndarray:
    """
    Compute pairwise cosine similarity between two lists of strings.
    Returns a (len(list1) × len(list2)) numpy array.
    """
    emb1 = embedding_model.encode(list1, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = embedding_model.encode(list2, convert_to_tensor=True, normalize_embeddings=True)
    return util.pytorch_cos_sim(emb1, emb2).cpu().numpy()


# =====================================================
# ENTITY SIMILARITY & PROBABILITY
# =====================================================

def entity_similarity_checker(
    data_description: str,
    entities: list,
    similarity_threshold: float = 0.6,
) -> list:
    """
    For each entity find its best-matching TF-IDF keyword in data_description
    and return the corresponding cosine similarity as a confidence score.

    Returns:
        List of dicts {entity_name: best_cosine_score}
    """
    keywords = extract_keywords_tfidf(data_description, top_n=14)
    print(f"Extracted Keywords: {keywords}")

    cosine_scores = cosine_similarity_score(entities, keywords)

    results = []
    for i, entity in enumerate(entities):
        best_score = float(np.max(cosine_scores[i]))
        best_match_kw = keywords[int(np.argmax(cosine_scores[i]))]

        if best_score >= similarity_threshold:
            print(f"Entity '{entity}' best match: ('{best_match_kw}', {best_score:.4f})")
        else:
            print(f"Entity '{entity}' no match above {similarity_threshold}. Best: ('{best_match_kw}', {best_score:.4f})")

        results.append({entity: round(best_score, 4)})
    return results


def compute_entity_probabilities(
    entity_confidence: List[Dict[str, float]],
    bias: float,
    weight: float,
) -> List[Dict]:
    """
    Calibrate cosine similarity scores into probabilities with sigmoid.

    Args:
        entity_confidence: [{entity_name: cosine_score}, ...]
        bias, weight: sigmoid parameters

    Returns:
        [{"entity": ..., "cosine": ..., "z": ..., "P(E)": ...}, ...]
    """
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    results = []
    for item in entity_confidence:
        for entity, cosine in item.items():
            z = bias + weight * cosine
            p = sigmoid(z)
            results.append({
                "entity": entity,
                "cosine": round(cosine, 4),
                "z":      round(z,      4),
                "P(E)":   round(p,      6),
            })
    return results


def filter_low_prob_entities(entity_probs: dict, threshold: float = 0.5) -> dict:
    """Zero out entities whose P(E) is below threshold; keep the rest unchanged."""
    return {e: (p if p >= threshold else 0.0) for e, p in entity_probs.items()}


def extract_entity_probs(entity_list: List[Dict]) -> Dict[str, float]:
    """Convert a list of entity dicts to {entity_name: P(E)} mapping."""
    return {e["entity"]: e["P(E)"] for e in entity_list}


# =====================================================
# ENTITY NAME NORMALISATION
# =====================================================

def normalize_entity_name(name: str) -> str:
    """Upper-case and naively singularise (strip trailing 'S' when safe)."""
    name = name.upper()
    if name.endswith("S") and name not in {"BILLINGS"}:
        name = name[:-1]
    return name


# =====================================================
# DESCRIPTION-BASED CONFIDENCE (for ToT / structured output)
# =====================================================

def compute_entity_confidence_from_description(entities: list, processed_data: str) -> list:
    """
    Compute SBERT cosine similarity between each entity's description
    and the full processed_data text.

    Args:
        entities: [{"entity": "PATIENT", "description": "..."}, ...]
        processed_data: source text

    Returns:
        [{entity_name: cosine_score}, ...]
    """
    processed_emb = embedding_model.encode(processed_data, convert_to_tensor=True)
    results = []
    for ent in entities:
        desc_emb = embedding_model.encode(ent["description"], convert_to_tensor=True)
        score = util.cos_sim(desc_emb, processed_emb).item()
        results.append({ent["entity"]: round(score, 4)})
    return results


def extract_and_calibrate_entities(
    generated_entity: str,
    processed_data: str,
    bias: float = -1.0,
    weight: float = 3.0,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Parse structured LLM output (entities with descriptions), compute similarity
    against processed_data, apply sigmoid calibration, and filter by threshold.

    Returns:
        {normalised_entity_name: P(E)}
    """
    json_text = generated_entity.strip().strip("```").strip()
    if not json_text:
        raise ValueError("Empty string for JSON parsing")
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse JSON: {e}\nOriginal text: {json_text}")

    entities   = data.get("entities", [])
    data_emb   = embedding_model.encode(processed_data, convert_to_tensor=True)
    entity_probs = {}

    for item in entities:
        entity_name = item.get("entity")
        description = item.get("description", "")
        if not entity_name or not description:
            continue

        desc_emb = embedding_model.encode(description, convert_to_tensor=True)
        cosine   = util.cos_sim(desc_emb, data_emb).item()
        z        = bias + weight * cosine
        p        = 1 / (1 + math.exp(-z))

        norm_name = normalize_entity_name(entity_name)
        entity_probs[norm_name] = round(p, 6)

    return entity_probs


# =====================================================
# CALIBRATION VISUALISATION (diagnostic helper)
# =====================================================

def calibrate_and_visualize(
    entity_confidence,
    output_path: str = "Output/pictures/calibration_plot.png",
    biases=None,
    weights=None,
):
    """
    Try multiple (bias, weight) combinations for sigmoid calibration,
    visualise the curves, and save to output_path.

    Returns:
        List of dicts {entity, cosine, P(E), bias, weight}
    """
    if biases  is None: biases  = [-2.0, -1.5, -1.0, -0.5, 0.0]
    if weights is None: weights = [0.0,  0.5,  1.0,  1.5,  2.0]

    def sigmoid(x): return 1 / (1 + math.exp(-x))

    calibration_results = []
    for b, w in itertools.product(biases, weights):
        for item in entity_confidence:
            for entity, cosine in item.items():
                z = b + w * cosine
                calibration_results.append({
                    "entity": entity,
                    "cosine": cosine,
                    "P(E)":  sigmoid(z),
                    "bias":  b,
                    "weight": w,
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    for entity in set(d["entity"] for d in calibration_results):
        pts = [d for d in calibration_results if d["entity"] == entity]
        plt.plot(
            [d["cosine"] for d in pts],
            [d["P(E)"]   for d in pts],
            marker='o', linestyle='-', label=entity,
        )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Calibrated Probability P(E)")
    plt.title("Entity Probability Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Calibration plot saved to: {output_path}")

    return calibration_results
