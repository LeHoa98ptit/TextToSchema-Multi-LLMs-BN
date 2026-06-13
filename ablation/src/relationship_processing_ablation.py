"""
Ablation Study — Relationship Processing (no Wikidata)
=======================================================
This module is a drop-in replacement for src/relationship_processing.py for
the ablation experiment that evaluates the contribution of Wikidata knowledge.

What changed vs. the original:
  - REMOVED: search_wikidata_qid(), sparql_instance_property_counts(),
             sparql_property_label_matches(), compute_x_type()
             and all Wikidata URL constants.
  - CHANGED: _bert_predict_proba() hand-crafted vector is now 3-dim:
             hc = [x_text, x_dep, x_cooccur]  (was 4-dim: included x_type).
             The MLP model is loaded from ablation/train/bert_model/ so it
             is NOT the original model (which was trained with x_type).
  - CHANGED: The logistic-regression fallback formula drops the w_type term:
             z = w_text*x_text + w_dep*x_dep + w_cooccur*x_cooccur + bias
  - CHANGED: Parameters are loaded from
             ablation/train/dataset/relationship_parameters_ablation.json
             (trained by ablation/train/train_relationship_parameters.py).
  - UNCHANGED: All prompting templates, JSON-extraction helpers, x_text /
               x_dep / x_cooccur computation, and the public API
               (compute_relationship_probabilities_2, merge_relationships, …).

Usage:
    from ablation.src.relationship_processing_ablation import (
        compute_relationship_probabilities_2,
        compute_x_text, compute_x_dep, compute_x_cooccur,
        ...
    )
"""

import os
import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)
import re
import json
import math
import time
import requests
from functools import lru_cache
from typing import Dict, List, Tuple

import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util

# =====================================================
# PROMPTING (unchanged from original)
# =====================================================

def relationship_prompting(processed_data, attributed_result):
    return f"""
    Task: Extract relationships between entities for a conceptual Entity–Relationship (ER) model.

    Original Description:
    {processed_data}

    Identified Entities and Attributes:
    {attributed_result}

    CRITICAL CONSTRAINTS (MUST FOLLOW STRICTLY):
    - You MUST ONLY use entities explicitly listed in "Identified Entities and Attributes".
    - DO NOT introduce, invent, or rename any entity under any circumstance.
    - DO NOT infer entities based on common sense or domain knowledge.
    - For each entity pair (A, B), output AT MOST ONE relationship. DO NOT create both A→B and B→A for the same pair.
    - DO NOT create duplicate or redundant relationships between the same two entities.

    OBJECTIVE:
    Identify semantic relationships between the given entities and determine their correct cardinalities
    based strictly on the text.

  CRITICAL INSTRUCTIONS (MUST FOLLOW STRICTLY):

    - - ONLY create an associative entity if a N:M relationship exists AND the relationship itself carries semantic meaning.
    - IF a N:M relationship clearly exists from the text, and the relationship represents an action/association between entities, you MUST create an associative entity EVEN IF NO attributes are explicitly stated.
    - Include a primary key and foreign keys to the two entities.
        - When creating an associative entity:
        1. Include attributes:
            - Primary key of the associative entity (e.g., id_RENTAL)
            - Foreign keys referencing the two related entities (e.g., id_RENTER, id_PROPERTY)
            - Any other attributes explicitly mentioned in the text that belong to the relationship itself
        2. Replace the original N:M relationship with two 1:N relationships pointing to the associative entity.
    - Do NOT leave associative_entity as null if the above conditions are met.
    - If the text explicitly names the associative entity (e.g., "Rental"), use exactly that name.
    - If the associative entity already exists as a normal entity, DO NOT invent a new one. Use the existing entity as normal, and replace N:M with two 1:N relationships to that entity.

    INSTRUCTIONS FOR EACH RELATIONSHIP:

    1. For each relationship between TWO GIVEN entities:
        - Identify the relationship meaning
        - Determine the cardinality: 1:1, 1:N, or N:M
        - Only if N:M AND the relationship has semantic meaning, define an associative entity according to the rules above
    2. Only include attributes that belong to the relationship itself; do NOT duplicate entity attributes
    3. Forbidden behaviors:
        - DO NOT invent weak entities
        - DO NOT decompose relationships into implementation-level foreign keys
        - DO NOT include attributes if not explicitly stated for the relationship

    OUTPUT FORMAT:
    - Return ONLY valid JSON
    {{
        "relationships": [
            {{
                "entity_1": "ENTITY1",
                "entity_2": "ENTITY2",
                "cardinality": "1:1",
                "associative_entity": null,
                "relationship_attributes": [],
                "description": "short description of the relationship"
            }}
        ]
    }}
    - Do NOT include other entities or attributes outside the relationship scope
    - Use null if a field does not apply
    - Entity and associative entity names MUST be UPPER_SNAKE_CASE
    - Do NOT return explanations, examples, markdown, or comments

"""


def relationship_few_shot_prompting(processed_data, attributed_result):
    return f"""
    Task: Extract relationships between entities for a conceptual Entity–Relationship (ER) model.

    Original Description:
    {processed_data}

    Identified Entities and Attributes:
    {attributed_result}

    CRITICAL CONSTRAINTS (MUST FOLLOW STRICTLY):
    - You MUST ONLY use entities explicitly listed in "Identified Entities and Attributes".
    - DO NOT introduce, invent, or rename any entity under any circumstance.
    - DO NOT infer entities based on common sense or domain knowledge.
    - For each entity pair (A, B), output AT MOST ONE relationship. DO NOT create both A→B and B→A for the same pair.
    - DO NOT create duplicate or redundant relationships between the same two entities.

    CARDINALITY RULES (apply carefully):
    - Use 1:N when: one instance of A is associated with MANY instances of B, but each B belongs to ONE A.
    - Use N:M only when: many instances of A are associated with many instances of B AND the text explicitly implies this.
    - Default to 1:N when unsure.

    OBJECTIVE:
    Identify semantic relationships between the given entities and determine their correct cardinalities
    based strictly on the text.

    OUTPUT FORMAT:
    - Return ONLY valid JSON
    {{
        "relationships": [
            {{
                "entity_1": "ENTITY1",
                "entity_2": "ENTITY2",
                "cardinality": "1:1",
                "associative_entity": null,
                "relationship_attributes": [],
                "description": "short description of the relationship"
            }}
        ]
    }}
    - Entity and associative entity names MUST be UPPER_SNAKE_CASE
    - Do NOT return explanations, examples, markdown, or comments

"""


# =====================================================
# JSON / RELATIONSHIP EXTRACTION (unchanged)
# =====================================================

def extract_json_block(text: str):
    match = re.search(r"```json\s*(\{{.*?\}})\s*```", text, re.DOTALL)
    json_str = match.group(1) if match else text.strip()
    return json.loads(json_str)


def extract_core_relationships(raw_text: str):
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        return {"error": "No JSON found in text"}
    try:
        data = json.loads(match.group(0))
        if "relationships" not in data:
            return {"error": "No 'relationships' key in JSON"}
        results = []
        for rel in data["relationships"]:
            results.append({
                "entity_1": rel.get("entity_1"),
                "entity_2": rel.get("entity_2"),
                "cardinality": rel.get("cardinality"),
                "description": rel.get("description"),
                "relationship_attributes": rel.get("relationship_attributes", []),
                "associative_entity": rel.get("associative_entity"),
            })
        return results
    except Exception as e:
        return {"error": f"Invalid JSON: {e}"}


def extract_final_relationships(raw_input):
    data = None
    if isinstance(raw_input, str):
        match = re.search(r"\{[\s\S]*\}", raw_input)
        if not match:
            return {"error": "No JSON found in text"}
        data = json.loads(match.group(0))
    elif isinstance(raw_input, dict):
        data = raw_input
    else:
        return {"error": f"Unsupported input type: {type(raw_input)}"}
    if "final_relationships" not in data:
        return {"error": "No 'final_relationships' key in JSON"}
    results = []
    for rel in data["final_relationships"]:
        entity_1, entity_2 = rel.get("entities", [None, None])
        results.append({
            "entity_1": entity_1,
            "entity_2": entity_2,
            "cardinality": rel.get("type"),
            "relationship_attributes": rel.get("foreign_keys", []),
            "associative_entity": rel.get("associative_entity"),
        })
    return results


def extract_relationships(llm_output: str):
    data = extract_json_block(llm_output)
    if "relationships" not in data:
        raise ValueError("JSON does not contain 'relationships' field")
    cleaned = []
    for rel in data["relationships"]:
        cleaned.append({
            "entity_1": rel.get("entity_1", "").upper(),
            "entity_2": rel.get("entity_2", "").upper(),
            "description": rel.get("description", ""),
            "cardinality": rel.get("cardinality", ""),
            "associative_entity": rel.get("associative_entity"),
        })
    return cleaned


def extract_relationships_one_shot(relationships):
    if not isinstance(relationships, list):
        raise TypeError("Input must be a list of relationships")
    cleaned = []
    for rel in relationships:
        cleaned.append({
            "entity_1": rel.get("entity_1", "").upper() if isinstance(rel.get("entity_1"), str) else "",
            "entity_2": rel.get("entity_2", "").upper() if isinstance(rel.get("entity_2"), str) else "",
            "description": rel.get("description", "") if isinstance(rel.get("description"), str) else "",
            "cardinality": rel.get("cardinality", "") if isinstance(rel.get("cardinality"), str) else "",
            "associative_entity": rel.get("associative_entity", "").upper()
                if isinstance(rel.get("associative_entity"), str) else None,
        })
    return cleaned


# =====================================================
# MODELS & UTILITIES
# =====================================================

nlp = spacy.load("en_core_web_sm")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Paths resolve relative to this file: ablation/src/ → ablation/train/
_ABLATION_TRAIN_DIR = os.path.join(os.path.dirname(__file__), '..', 'train')

def _load_rel_params():
    path = os.path.join(_ABLATION_TRAIN_DIR, 'dataset', 'relationship_parameters_ablation.json')
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        # Fallback defaults (no w_type term)
        return {"w_text": 1.0349, "w_dep": -1.5494, "w_cooccur": 0.5, "bias": -1.7070}

_REL_PARAMS = _load_rel_params()


def _load_bert_rel_model():
    import pickle
    path = os.path.join(_ABLATION_TRAIN_DIR, 'bert_model', 'relationship_mlp_model_ablation.pkl')
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

_BERT_REL_MODEL = _load_bert_rel_model()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def normalize_cosine_to_01(sim):
    return float((sim + 1.0) / 2.0)

def score_from_counts(total_count, log_div=4.0):
    if total_count <= 0:
        return 0.0
    return float(math.tanh(math.log1p(total_count) / log_div))


# =====================================================
# FEATURE COMPUTATION (no x_type)
# =====================================================

def compute_x_text(description, processed_data):
    desc_emb = sbert.encode(description, convert_to_tensor=True)
    trig_embs = sbert.encode(processed_data, convert_to_tensor=True)
    cosine_sim = util.cos_sim(desc_emb, trig_embs)
    return float(cosine_sim.max().item())


def get_pair_context(source_text, e1, e2):
    e1_words = set(w for w in re.split(r'[_\s]+', e1.lower()) if len(w) > 1) or {e1.lower()}
    e2_words = set(w for w in re.split(r'[_\s]+', e2.lower()) if len(w) > 1) or {e2.lower()}
    shared   = e1_words & e2_words
    e1_exc   = e1_words - shared or e1_words
    e2_exc   = e2_words - shared or e2_words
    doc = nlp(source_text)
    for sent in doc.sents:
        low = sent.text.lower()
        if any(w in low for w in e1_exc) and any(w in low for w in e2_exc):
            return sent.text.strip()
    return None


def compute_x_dep(text, e1, e2):
    doc = nlp(text)
    e1_words = set(w for w in re.split(r'[_\s]+', e1.lower()) if len(w) > 1) or {e1.lower()}
    e2_words = set(w for w in re.split(r'[_\s]+', e2.lower()) if len(w) > 1) or {e2.lower()}
    shared = e1_words & e2_words
    e1_exclusive = e1_words - shared or e1_words
    e2_exclusive = e2_words - shared or e2_words
    e1_tokens = [t for t in doc if t.lemma_.lower() in e1_exclusive or t.text.lower() in e1_exclusive]
    e2_tokens = [t for t in doc if t.lemma_.lower() in e2_exclusive or t.text.lower() in e2_exclusive]
    if not e1_tokens or not e2_tokens:
        return 0.0
    min_path_len = float('inf')
    for t1 in e1_tokens:
        for t2 in e2_tokens:
            ancestors_1 = list(t1.ancestors) + [t1]
            ancestors_2 = list(t2.ancestors) + [t2]
            common = set(ancestors_1).intersection(ancestors_2)
            if common:
                path_len = len(set(ancestors_1 + ancestors_2)) - len(common)
                min_path_len = min(min_path_len, path_len)
    if min_path_len == float('inf'):
        return 0.0
    return round(1 / (1 + min_path_len), 3)


@lru_cache(maxsize=512)
def _sent_texts(full_text):
    return [s.text.lower() for s in nlp(full_text).sents]


def compute_x_cooccur(full_text, e1, e2):
    e1_words = set(w for w in re.split(r'[_\s]+', e1.lower()) if len(w) > 1) or {e1.lower()}
    e2_words = set(w for w in re.split(r'[_\s]+', e2.lower()) if len(w) > 1) or {e2.lower()}
    shared   = e1_words & e2_words
    e1_exc   = e1_words - shared or e1_words
    e2_exc   = e2_words - shared or e2_words
    sents = _sent_texts(full_text)
    if not sents:
        return 0.0
    count = sum(
        1 for s in sents
        if any(w in s for w in e1_exc) and any(w in s for w in e2_exc)
    )
    return round(count / len(sents), 3)


# =====================================================
# BERT MLP (3 hand-crafted features: no x_type)
# =====================================================

def _bert_predict_proba(e1, e2, x_text, x_dep, x_cooccur):
    """Use ablation MLP model (trained without x_type) if available."""
    if _BERT_REL_MODEL is None:
        return None
    try:
        import re as _re
        import numpy as _np

        def _clean(s):
            s = _re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
            return s.replace('_', ' ').replace('-', ' ').lower().strip()

        emb_e1 = sbert.encode([_clean(e1)])[0]
        emb_e2 = sbert.encode([_clean(e2)])[0]
        diff   = _np.abs(emb_e1 - emb_e2)
        prod   = emb_e1 * emb_e2
        cos    = float(_np.dot(emb_e1, emb_e2) / (_np.linalg.norm(emb_e1) * _np.linalg.norm(emb_e2) + 1e-8))
        # 3 hand-crafted features (x_type removed)
        hc     = _np.array([x_text, x_dep, x_cooccur])
        feat   = _np.concatenate([diff, prod, [cos], hc]).reshape(1, -1)
        return float(_BERT_REL_MODEL.predict_proba(feat)[0][1])
    except Exception:
        return None


# =====================================================
# MAIN PROBABILITY FUNCTION
# =====================================================

def compute_relation_probability(e1, e2, description, processed_data, bias=None, weight=None):
    x_text    = compute_x_text(description, processed_data)
    x_dep     = compute_x_dep(description, e1, e2)
    x_cooccur = compute_x_cooccur(processed_data, e1, e2)
    # x_type deliberately omitted (ablation)

    p_bert = _bert_predict_proba(e1, e2, x_text, x_dep, x_cooccur)
    if p_bert is not None:
        p = p_bert
    else:
        w_text    = _REL_PARAMS["w_text"]
        w_dep     = _REL_PARAMS["w_dep"]
        w_cooccur = _REL_PARAMS.get("w_cooccur", 0.0)
        b         = _REL_PARAMS["bias"]
        z = w_text * x_text + w_dep * x_dep + w_cooccur * x_cooccur + b
        p = sigmoid(z)

    return {
        "E1": e1,
        "E2": e2,
        "description": description,
        "x": {
            "x_text":    round(x_text,    4),
            "x_dep":     round(x_dep,     4),
            "x_cooccur": round(x_cooccur, 4),
        },
        "P(R|E1,E2,x)": round(p, 6)
    }


def compute_relationship_probabilities_1(data, processed_data):
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, list):
        relationships = data
    elif isinstance(data, dict) and "relationships" in data:
        relationships = data["relationships"]
    else:
        raise ValueError("Input does not contain 'relationships'")
    results = []
    for rel in relationships:
        r = compute_relation_probability(rel["entity_1"], rel["entity_2"], rel["description"], processed_data)
        results.append(r)
    return results


def compute_relationship_probabilities(data, processed_data):
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and "relationships" in data:
        relationships = data["relationships"]
    elif isinstance(data, list):
        relationships = data
    else:
        raise ValueError("Input does not contain 'relationships'")
    results = []
    for rel in relationships:
        if "entity_1" not in rel or "entity_2" not in rel:
            raise ValueError(f"Invalid relationship format: {rel}")
        e1 = rel["entity_1"].strip().upper()
        e2 = rel["entity_2"].strip().upper()
        description = rel.get("description") or ""
        prob_result = compute_relation_probability(e1, e2, description, processed_data)
        results.append({"e1": e1, "e2": e2, "p": prob_result.get("P(R|E1,E2,x)", 0.0)})
    return results


def compute_relationship_probabilities_2(data, processed_data, bias=None, weight=None):
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and "relationships" in data:
        relationships = data["relationships"]
    elif isinstance(data, list):
        relationships = data
    else:
        raise ValueError("Input does not contain 'relationships'")
    results = []
    for rel in relationships:
        if "entity_1" not in rel or "entity_2" not in rel:
            raise ValueError(f"Invalid relationship format: {rel}")
        e1 = rel["entity_1"].strip().upper()
        e2 = rel["entity_2"].strip().upper()
        description = rel.get("description") or ""
        cardinality = (rel.get("cardinality") or "").strip()
        associative = rel.get("associative_entity", None)
        prob_result = compute_relation_probability(e1, e2, description, processed_data, bias, weight)
        results.append({
            "e1": e1,
            "e2": e2,
            "p": prob_result.get("P(R|E1,E2,x)", 0.0),
            "associative_entity": associative,
            "cardinality": cardinality,
        })
    return results


def merge_relationships(*lists: List[Dict]) -> List[Dict]:
    merged_dict = {}
    for lst in lists:
        for r in lst:
            e1, e2, p = r["E1"], r["E2"], r["P(R|E1,E2,x)"]
            key = (e1, e2)
            if key not in merged_dict or p > merged_dict[key]["p"]:
                merged_dict[key] = {"e1": e1, "e2": e2, "p": p}
    return list(merged_dict.values())
