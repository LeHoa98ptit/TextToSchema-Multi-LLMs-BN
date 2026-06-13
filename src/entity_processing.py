# import libraries

import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)
import ssl
from openie import StanfordOpenIE
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import string
import spacy
from collections import defaultdict
from openie import StanfordOpenIE
from g4f.client import Client
import os
from groq import Groq
import re
import nltk
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import itertools
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

# download stopwords if not already available
nltk.download("stopwords")
from nltk.corpus import stopwords

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def entity_prompting_without_description_1(processed_data):
    """
    Generates a prompt for extracting candidate entities from a textual description.
    Returns a properly formatted f-string for LLM input.
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
    5. Do NOT extract optional or detail entities not central to the main ER model, such as payments, accounts, contracts, security deposits, maintenance records, or lease extensions—unless they are needed for an explicit associative relationship.
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
    safe_data = processed_data.replace("{", "{{").replace("}", "}}")

    prompt = f"""
    Extract candidate ENTITIES from the text below for an Entity-Relationship (ER) model.

    Original Description:
    {safe_data}

    Rules:
    1. Include BOTH physical entities (people, objects) AND process/event entities (services, sessions, registrations, transactions, participations).
       - A process/event entity is needed when an activity has its OWN attributes (e.g., a "Grooming Service" has duration, price, rating).
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
    Description: A hospital manages medical appointments. For each appointment the dataset stores patient info (name, date of birth, gender, phone), doctor info (name, specialty, license number), the appointment date and time, and diagnosis notes.
    Generated entities:
    {{
        "entities": ["PATIENT", "DOCTOR", "APPOINTMENT"]
    }}
    """
    return prompt



def extract_entity_names(llm_output: str) -> list:
    """
    Extract entity names from LLM output that may contain text reasoning and markdown JSON.

    Args:
        llm_output (str): Raw output from LLM.

    Returns:
        list: List of entity names.
    """
    # 1. Find JSON block in string (between ```json and ```)
    match = re.search(r"```json\s*(\{.*?\})\s*```", llm_output, re.DOTALL)
    if not match:
        raise ValueError("No JSON block found in LLM output.")

    json_str = match.group(1)

    # 2. Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nOriginal JSON:\n{json_str}")

    # 3. get entities' name
    entity_names = [e["name"] for e in data.get("entities", [])]

    return entity_names


def preprocess_text(text: str):
    """Basic text preprocessing: lowercase, remove special characters, remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # remove special characters
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

def extract_keywords_tfidf(text: str, top_n: int = 30):
    """
    Extract keywords from text using TF-IDF.
    """
    processed_text = preprocess_text(text)
    vectorizer = TfidfVectorizer(max_features=top_n, ngram_range=(1, 2))
    X = vectorizer.fit_transform([processed_text])
    keywords = vectorizer.get_feature_names_out()
    return list(keywords)

def cosine_similarity_score(list1, list2):
    """
    Compute cosine similarity between two lists of strings (list1, list2).
    Args:
        list1 (list[str]): first list of words/strings
        list2 (list[str]): second list of words/strings
    Returns:
        np.ndarray: cosine similarity matrix (len(list1) x len(list2))
    """
    emb1 = embedding_model.encode(list1, convert_to_tensor=True, normalize_embeddings=True)
    emb2 = embedding_model.encode(list2, convert_to_tensor=True, normalize_embeddings=True)
    cosine_scores = util.pytorch_cos_sim(emb1, emb2)
    return cosine_scores.cpu().numpy()

def entity_similarity_checker(data_description: str, entities: list, similarity_threshold: float = 0.6):
    """
    For each entity, take the highest similarity (best match) instead of the average.
    Returns a list of dicts {entity: confidence_score}.
    """
    # 1. Extract keywords
    keywords = extract_keywords_tfidf(data_description, top_n=14)
    print(f"Extracted Keywords: {keywords}")

    # 2. Compute similarity
    cosine_scores = cosine_similarity_score(entities, keywords)

    # 3. Store results
    results = []
    for i, entity in enumerate(entities):
        # Get max score in the corresponding row
        best_score = float(np.max(cosine_scores[i]))
        # Best match keyword
        best_match_idx = np.argmax(cosine_scores[i])
        best_match_keyword = keywords[best_match_idx]

        if best_score >= similarity_threshold:
            print(f"Entity '{entity}' best match: ('{best_match_keyword}', {best_score:.4f})")
        else:
            print(f"Entity '{entity}' has no matches above threshold {similarity_threshold}. Best: ('{best_match_keyword}', {best_score:.4f})")

        results.append({entity:  round(best_score, 4)})

    return results


def extract_entities_from_text_1(raw_text):
    """
    Extracts JSON entities from a raw text that may contain descriptions and other text.
    Only keeps the entity names.
    """
    # 1. Find the JSON block (between ```json ... ```)
    json_match = re.search(r"```json(.*?)```", raw_text, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON block found in the text.")
    
    # 2. Get the JSON text
    json_text = json_match.group(1).strip()
    
    # 3. Parse JSON into a Python dictionary
    data = json.loads(json_text)
    
    # 4. Normalize entities:
    #    - if entity is a dict with 'name', take 'name'
    #    - if entity is a string, keep it
    entities = []
    for ent in data.get("entities", []):
        if isinstance(ent, dict) and "name" in ent:
            entities.append(ent["name"])
        elif isinstance(ent, str):
            entities.append(ent)
    
    return entities


def extract_llama_entities_from_text(raw_text):
    """
    Extract entity names from text. Handles LLAMA-style (```json``` block), GPT-style (direct JSON), 
    or raw text with extra explanation before JSON.
    """
    import json
    import re
    import logging

    logger = logging.getLogger(__name__)

    # 1️⃣ Find ```json ... ``` block
    json_match = re.search(r"```json(.*?)```", raw_text, re.DOTALL | re.IGNORECASE)
    if json_match:
        json_text = json_match.group(1).strip()
    else:
        # 2️⃣ fallback: find { ... } b
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            json_text = match.group()
        else:
            logger.warning("No JSON object found in LLM response, returning empty list")
            return []

    # 3️⃣ parse JSON
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.error(f"Cannot parse JSON: {e}\nRaw text: {raw_text}")
        return []

    # 4️⃣ normalize entities
    entities = []
    for ent in data.get("entities", []):
        if isinstance(ent, dict) and "name" in ent:
            entities.append(ent["name"])
        elif isinstance(ent, str):
            entities.append(ent)
    return entities


# ===== Extract entities from GPT output =====
def extract_gpt_entities(text):
    """
    Extract entities from GPT LLM output.
    GPT output may be a raw JSON dict or JSON block containing objects with 'name'.
    """
    # If it has ```json``` block, extract it
    match = re.search(r"```json(.*?)```", text, re.DOTALL)
    json_text = match.group(1).strip() if match else text.strip()
    
    data = json.loads(json_text)
    
    # Normalize: take 'name' if dict, or string directly
    entities = []
    for ent in data.get("entities", []):
        if isinstance(ent, dict) and "name" in ent:
            entities.append(ent["name"])
        elif isinstance(ent, str):
            entities.append(ent)
    return entities

def merge_entity_lists(list1, list2):
    """
    Merge two entity lists and remove duplicates while keeping order.
    """
    merged = []
    seen = set()
    
    for entity in list1 + list2:
        if entity not in seen:
            merged.append(entity)
            seen.add(entity)
    
    return merged


def compute_entity_probabilities(
    entity_confidence: List[Dict[str, float]], 
    bias: float, 
    weight: float,
) -> List[Dict]:
    """
    Compute probability P(E) for each entity using sigmoid function.

    Args:
        entity_confidence: List of dicts, each dict {entity_name: cosine_similarity}
        bias: Bias parameter for sigmoid (b)
        weight: Weight parameter for sigmoid (w)

    Returns:
        List of dicts with keys: entity, cosine, z, P(E)
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
                "z": round(z, 4),
                "P(E)": round(p, 6)
            })
    return results

def filter_low_prob_entities(entity_probs: dict, threshold: float = 0.5) -> dict:
    """
    Set entities with probability P(E) below threshold to 0.0,
    keep the others unchanged.
    
    Input: dict {entity_name: P(E)}
    Output: dict with low-prob entities set to 0.0
    """
    return {e: (p if p >= threshold else 0.0) for e, p in entity_probs.items()}


def extract_entity_probs(entity_list: List[Dict]) -> Dict[str, float]:
    """
    Extract entity probabilities from a list of dicts.
    
    Args:
        entity_list: list of dicts, each dict contains at least
                     'entity' and 'P(E)'
                     
    Returns:
        Dict[str, float]: mapping from entity name to its probability
    """
    return {e["entity"]: e["P(E)"] for e in entity_list}


def calibrate_and_visualize(entity_confidence, output_path="Output/pictures/calibration_plot.png",
                            biases=None, weights=None):
    """
    Calibrate cosine similarity into probability with multiple bias/weight values,
    then save visualization to a file.

    Args:
        entity_confidence: List of dicts {entity_name: cosine_similarity}
        output_path: str, path to save the figure (including filename)
        biases: list of float, bias values to try
        weights: list of float, weight values to try

    Returns:
        calibration_results: list of dicts with keys:
            entity, cosine, P(E), bias, weight
    """
    if biases is None:
        biases = [-2.0, -1.5, -1.0, -0.5, 0.0]
    if weights is None:
        weights = [0.0, 0.5, 1.0, 1.5, 2.0]

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    calibration_results = []

    for b, w in itertools.product(biases, weights):
        for item in entity_confidence:
            for entity, cosine in item.items():
                z = b + w * cosine
                p = sigmoid(z)
                calibration_results.append({
                    "entity": entity,
                    "cosine": cosine,
                    "P(E)": p,
                    "bias": b,
                    "weight": w
                })

    # --- Visualization ---
    # Ensure folder exists
    folder = os.path.dirname(output_path)
    os.makedirs(folder, exist_ok=True)

    plt.figure(figsize=(10,6))
    for entity in set(d["entity"] for d in calibration_results):
        data = [d for d in calibration_results if d["entity"] == entity]
        plt.plot(
            [d["cosine"] for d in data],
            [d["P(E)"] for d in data],
            marker='o',
            linestyle='-',
            label=entity
        )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Calibrated Probability P(E)")
    plt.title("Entity Probability Calibration Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close figure to avoid memory issues
    print(f"Calibration plot saved to: {output_path}")

    return calibration_results

# ==== DESCRIPTION =====

def normalize_entity_name(name: str) -> str:
    """
    Normalize entity name: uppercase and singular (remove trailing 'S' if needed)
    """
    name = name.upper()
    if name.endswith("S") and name not in ["BILLINGS"]:  # avoid incorrectly stripping "BILLINGS"
        name = name[:-1]
    return name


def compute_entity_confidence_from_description(entities, processed_data):
    """
    entities: list of dicts {"entity": "PATIENT", "description": "..."}
    processed_data: str
    """
    processed_emb = embedding_model.encode(processed_data, convert_to_tensor=True)
    results = []
    for ent in entities:
        desc_emb = embedding_model.encode(ent["description"], convert_to_tensor=True)
        cosine_score = util.cos_sim(desc_emb, processed_emb).item()
        results.append({ent["entity"]: round(cosine_score, 4)})
    return results


# function calculate with description
def extract_and_calibrate_entities(generated_entity: str, processed_data: str,
                                  bias: float = -1.0, weight: float = 3.0,
                                  threshold: float = 0.5) -> Dict[str, float]:
    """
    Extract entities with description, compute similarity vs processed_data,
    calibrate with sigmoid, filter by threshold, return dict {entity: P(E)}.
    """
    # ------------- Parse JSON safely -------------
    json_text = generated_entity.strip().strip("```").strip()
    if not json_text:
        raise ValueError("Empty string for JSON parsing")
    
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot parse JSON: {e}\nOriginal text: {json_text}")
    
    entities = data.get("entities", [])
    
    # ------------- Embeddings for similarity -------------
    data_emb = embedding_model.encode(processed_data, convert_to_tensor=True)
    
    entity_probs = {}
    for item in entities:
        entity_name = item.get("entity")
        description = item.get("description", "")
        if not entity_name or not description:
            continue
        
        desc_emb = embedding_model.encode(description, convert_to_tensor=True)
        cosine = util.cos_sim(desc_emb, data_emb).item()
        
        # sigmoid calibration
        z = bias + weight * cosine
        p = 1 / (1 + math.exp(-z))

        # normalize entity name
        norm_name = normalize_entity_name(entity_name)
        entity_probs[norm_name] = round(p, 6)
    return entity_probs
