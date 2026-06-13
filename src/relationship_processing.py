# import libraries

import os
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
import re
from collections import defaultdict
from openie import StanfordOpenIE
from g4f.client import Client
from typing import List
from groq import Groq
import re
import nltk
import numpy as np
import math
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import requests
from functools import lru_cache
from sentence_transformers import SentenceTransformer, util


"""
Prompting
"""

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
    - Only include the following keys per relationship:
    {{
        "relationships": [
            {{
                "entity_1": "ENTITY1",
                "entity_2": "ENTITY2",
                "cardinality": "1:1",
                "associative_entity": null,
                "relationship_attributes": [],
                "description": "short description of the relationship"
            }},
            {{
                "entity_1": "ENTITY3",
                "entity_2": "ENTITY4",
                "cardinality": "N:M",
                "associative_entity": {{
                    "name": "ASSOCIATIVE_ENTITY",
                    "attributes": ["attr1", "attr2"]
                }},
                "relationship_attributes": [],
                "description": "another relationship description"
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
      Example: One doctor handles many appointments, but each appointment has exactly one doctor → 1:N.
    - Use N:M only when: many instances of A are associated with many instances of B AND the text explicitly implies this bidirectional many relationship.
      Example: A student can enroll in many courses AND each course has many students → N:M.
    - Default to 1:N when unsure. N:M is rarer and usually only when explicitly stated.
    - If a process/event entity exists (e.g., GROOMING_SERVICE, APPOINTMENT), it typically has 1:N relationships FROM parent entities TO it, NOT N:M.

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
    - Only include the following keys per relationship:
    {{
        "relationships": [
            {{
                "entity_1": "ENTITY1",
                "entity_2": "ENTITY2",
                "cardinality": "1:1",
                "associative_entity": null,
                "relationship_attributes": [],
                "description": "short description of the relationship"
            }},
            {{
                "entity_1": "ENTITY3",
                "entity_2": "ENTITY4",
                "cardinality": "N:M",
                "associative_entity": {{
                    "name": "ASSOCIATIVE_ENTITY",
                    "attributes": ["attr1", "attr2"]
                }},
                "relationship_attributes": [],
                "description": "another relationship description"
            }}
        ]
    }}
    - Do NOT include other entities or attributes outside the relationship scope
    - Use null if a field does not apply
    - Entity and associative entity names MUST be UPPER_SNAKE_CASE
    - Do NOT return explanations, examples, markdown, or comments

    Example: 
    This is a dataset description: The dataset contains 50,000 rows.
    Each row corresponds to one medical appointment at a hospital.
    For each appointment, the dataset stores:
    information about the patient (name, date of birth, gender, phone, address),
    information about the doctor (name, specialty, license number),
    the date and time of the appointment,
    the room where the appointment takes place,
    and the diagnosis and notes written by the doctor
    Given entities: 
    {{
        "entities": ["PATIENT", "DOCTOR", "APPOINTMENT"]
    }}
    Generated attributes: 
    {{
        "attributes": {{
            "PATIENT": [
                "patient_id",
                "full_name",
                "date_of_birth",
                "gender",
                "phone_number",
                "address"
            ],
            "DOCTOR": [
                "doctor_id",
                "full_name",
                "specialty",
                "license_number",
                "phone_number"
            ],
            "APPOINTMENT": [
                "appointment_id",
                "appointment_date",
                "appointment_time",
                "diagnosis",
                "notes"
            ]
        }}
    }}
    
    "relationships": [
        {{
            "entity_1": "PATIENT",
            "entity_2": "APPOINTMENT",
            "description": "Each appointment is scheduled for exactly one patient, and each patient can have many appointments over time.",
            "cardinality": "1:N",
            "is_identifying": false,
            "associative_entity": null
        }},
        {{
            "entity_1": "DOCTOR",
            "entity_2": "APPOINTMENT",
            "description": "Each appointment is handled by exactly one doctor, and each doctor can handle many appointments.",
            "cardinality": "1:N",
            "is_identifying": false,
            "associative_entity": null
        }}
    ]

"""


def extract_core_relationships(raw_text: str):
    """
    Extract only the necessary information from relationships:
    entity_1, entity_2, cardinality, relationship_attributes, associative_entity
    """
    # find JSON block in raw_text
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
                "associative_entity": rel.get("associative_entity")
            })
        return results
    except Exception as e:
        return {"error": f"Invalid JSON: {e}"}
    

def extract_final_relationships(raw_input):
    """
    Extract entity_1, entity_2, cardinality, relationship_attributes
    from the 'final_relationships' structure.
    """
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
            "associative_entity": rel.get("associative_entity")
        })
    return results

def extract_json_block(text: str):
    """
    Extract JSON inside ```json ... ``` block.
    If no block found, assume whole string is JSON.
    """
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = match.group(1) if match else text.strip()
    return json.loads(json_str)


def extract_relationships(llm_output: str):
    """
    Extract relationships in a clean and simplified format:
    Only keep: entity_1, entity_2, description, cardinality
    """
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
            "associative_entity": rel.get("associative_entity")
        })

    return cleaned


def extract_relationships_one_shot(relationships):
    """
    Extract relationships in a clean and simplified format:
    Only keep: entity_1, entity_2, description, cardinality
    Input: list of relationship dicts
    """
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
                if isinstance(rel.get("associative_entity"), str) else None
        })

    return cleaned



"""
Relation Probability Computation
Features: x_text, x_dep, x_type, x_KGE
Requirements:
    pip install spacy sentence-transformers requests scikit-learn
    python -m spacy download en_core_web_sm
"""

import math
import time
import requests
from functools import lru_cache
import spacy
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. Load models
# -------------------------------
nlp = spacy.load("en_core_web_sm")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Load learned relationship weights from training (fallback to defaults if missing)
def _load_rel_params():
    path = os.path.join(os.path.dirname(__file__), '..', 'train', 'dataset', 'relationship_parameters.json')
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {"w_text": 1.0349, "w_dep": -1.5494, "w_type": 1.4987, "bias": -1.7070}

_REL_PARAMS = _load_rel_params()

# Load BERT-based MLP classifier (optional — used when available)
def _load_bert_rel_model():
    import pickle
    path = os.path.join(os.path.dirname(__file__), '..', 'train', 'bert_model', 'relationship_mlp_model.pkl')
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return None

_BERT_REL_MODEL = _load_bert_rel_model()

# -------------------------------
# 2. Utilities
# -------------------------------
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

headers = {"User-Agent": "ER-schema-bot/1.0"}

def safe_request(url, params=None, timeout=30):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r
    except:
        return None

def normalize_cosine_to_01(sim):
    return float((sim + 1.0) / 2.0)

def score_from_counts(total_count, log_div=4.0):
    if total_count <= 0: return 0.0
    return float(math.tanh(math.log1p(total_count)/log_div))

# -------------------------------
# 3. Wikidata entity linking + SPARQL
# -------------------------------
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

@lru_cache(maxsize=1024)
def search_wikidata_qid(label, language="en"):
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": language,
        "format": "json",
        "limit": 3
    }
    r = safe_request(WIKIDATA_SEARCH_URL, params=params)
    if not r: return None, None, None
    hits = r.json().get("search", [])
    if not hits: return None, None, None
    top = hits[0]
    return top.get("id"), top.get("label"), top.get("description")

@lru_cache(maxsize=256)
def sparql_instance_property_counts(qid_e1, qid_e2, limit=50):
    query = f"""
    SELECT ?prop (COUNT(*) AS ?count) WHERE {{
      ?s wdt:P31/wdt:P279* wd:{qid_e1} .
      ?o wdt:P31/wdt:P279* wd:{qid_e2} .
      ?s ?prop ?o .
      FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
    }}
    GROUP BY ?prop
    ORDER BY DESC(?count)
    LIMIT {limit}
    """
    r = safe_request(WIKIDATA_SPARQL_URL, params={"query": query, "format": "json"})
    if not r: return []
    try: bindings = r.json()["results"]["bindings"]
    except: return []
    return [(b["prop"]["value"], int(b["count"]["value"])) for b in bindings]

@lru_cache(maxsize=256)
def sparql_property_label_matches(keywords_tuple, limit=50):
    contains_clauses = " || ".join(f'CONTAINS(LCASE(?label), "{kw}")' for kw in keywords_tuple)
    query = f"""
    SELECT ?prop ?label WHERE {{
      ?prop a wikibase:Property .
      ?prop rdfs:label ?label .
      FILTER({contains_clauses})
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """
    r = safe_request(WIKIDATA_SPARQL_URL, params={"query": query, "format": "json"})
    if not r: return []
    try: bindings = r.json()["results"]["bindings"]
    except: return []
    return [(b["prop"]["value"], b.get("label", {}).get("value","")) for b in bindings]

# -------------------------------
# 4. Compute x_type
# -------------------------------
def compute_x_type(e1_label, e2_label, debug=False):
    debug_info = {"e1": e1_label, "e2": e2_label}
    qid1, lbl1, desc1 = search_wikidata_qid(e1_label)
    qid2, lbl2, desc2 = search_wikidata_qid(e2_label)
    debug_info.update({"qid1": qid1, "qid2": qid2})

    # --- Instance-based (Wikidata SPARQL): strongest signal when available ---
    instances_score = 0.0
    if qid1 and qid2:
        props = sparql_instance_property_counts(qid1, qid2)
        total_count = sum(c for _, c in props)
        instances_score = score_from_counts(total_count)
    debug_info["instances_score"] = instances_score

    # --- Description embedding similarity (Wikidata descriptions → SBERT) ---
    # Uses the actual Wikidata concept descriptions, not just entity names.
    # desc1/desc2 examples: "type of video game character", "competitive game session"
    # This discriminates better than name similarity or property keyword counting.
    try:
        text1 = desc1 if desc1 else (lbl1 if lbl1 else e1_label.replace('_', ' ').lower())
        text2 = desc2 if desc2 else (lbl2 if lbl2 else e2_label.replace('_', ' ').lower())
        emb1 = sbert.encode(text1, convert_to_tensor=True)
        emb2 = sbert.encode(text2, convert_to_tensor=True)
        desc_sim = normalize_cosine_to_01(util.cos_sim(emb1, emb2).item())
    except:
        desc_sim = 0.5
    debug_info["desc_sim"] = desc_sim

    # --- Combine: Wikidata instance score + description similarity ---
    # instance_score is a strong Wikidata signal (use when non-zero),
    # desc_sim is always computed and provides a continuous discriminating signal.
    if instances_score > 0:
        x_type = 0.6 * instances_score + 0.4 * desc_sim
        source = "instance+desc"
    else:
        x_type = desc_sim
        source = "desc"

    x_type = max(0.0, min(1.0, x_type))
    debug_info.update({"x_type": x_type, "source": source})
    return (x_type, debug_info) if debug else x_type

# -------------------------------
# 5. Compute x_text
# -------------------------------
def compute_x_text(description, processed_data):
  desc_emb = sbert.encode(description, convert_to_tensor=True)
  trig_embs = sbert.encode(processed_data, convert_to_tensor=True)
  cosine_sim = util.cos_sim(desc_emb, trig_embs)
  return float(cosine_sim.max().item())

# -------------------------------
# 5b. Extract source-text sentence containing both entities (mirrors training)
# -------------------------------
def get_pair_context(source_text, e1, e2):
    """Return the sentence in source_text that contains words from both e1 and e2.
    Falls back to None so the caller can use the LLM description instead."""
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

# -------------------------------
# 6. Compute x_dep
# -------------------------------
def compute_x_dep(text, e1, e2):
    doc = nlp(text)
    e1_words = set(w for w in re.split(r'[_\s]+', e1.lower()) if len(w) > 1) or {e1.lower()}
    e2_words = set(w for w in re.split(r'[_\s]+', e2.lower()) if len(w) > 1) or {e2.lower()}

    # Remove shared words to avoid artificial zero-length paths (e.g. PLAYER vs SINGLE_PLAYER_GAME)
    shared = e1_words & e2_words
    e1_exclusive = e1_words - shared or e1_words
    e2_exclusive = e2_words - shared or e2_words

    e1_tokens = [t for t in doc if t.lemma_.lower() in e1_exclusive or t.text.lower() in e1_exclusive]
    e2_tokens = [t for t in doc if t.lemma_.lower() in e2_exclusive or t.text.lower() in e2_exclusive]
    if not e1_tokens or not e2_tokens: return 0.0

    min_path_len = float('inf')
    for t1 in e1_tokens:
        for t2 in e2_tokens:
            ancestors_1 = list(t1.ancestors) + [t1]
            ancestors_2 = list(t2.ancestors) + [t2]
            common = set(ancestors_1).intersection(ancestors_2)
            if common:
                path_len = len(set(ancestors_1 + ancestors_2)) - len(common)
                min_path_len = min(min_path_len, path_len)
    if min_path_len == float('inf'): return 0.0
    return round(1 / (1 + min_path_len), 3)

# -------------------------------
# 6b. Compute x_cooccur
# -------------------------------
@lru_cache(maxsize=512)
def _sent_texts(full_text):
    """Cache tokenized sentences for a given text."""
    return [s.text.lower() for s in nlp(full_text).sents]

def compute_x_cooccur(full_text, e1, e2):
    """Fraction of sentences in full_text where exclusive words of both e1 and e2 co-occur."""
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
        if any(w in s for w in e1_exc)
        and any(w in s for w in e2_exc)
    )
    return round(count / len(sents), 3)

# -------------------------------
# 7. Main probability function
# -------------------------------
def _bert_predict_proba(e1, e2, x_text, x_dep, x_type, x_cooccur):
    """Use fine-tuned MLP+SBERT model if available."""
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
        hc     = _np.array([x_text, x_dep, x_type, x_cooccur])
        feat   = _np.concatenate([diff, prod, [cos], hc]).reshape(1, -1)
        return float(_BERT_REL_MODEL.predict_proba(feat)[0][1])
    except Exception:
        return None


def compute_relation_probability(e1, e2, description, processed_data, bias=None, weight=None):
    x_text    = compute_x_text(description, processed_data)
    x_dep     = compute_x_dep(description, e1, e2)
    x_type    = compute_x_type(e1, e2)
    x_cooccur = compute_x_cooccur(processed_data, e1, e2)

    # Try BERT-based MLP first; fall back to logistic regression
    p_bert = _bert_predict_proba(e1, e2, x_text, x_dep, x_type, x_cooccur)
    if p_bert is not None:
        p = p_bert
    else:
        w_text    = _REL_PARAMS["w_text"]
        w_dep     = _REL_PARAMS["w_dep"]
        w_type    = _REL_PARAMS["w_type"]
        w_cooccur = _REL_PARAMS.get("w_cooccur", 0.0)
        b         = _REL_PARAMS["bias"]
        z = w_text * x_text + w_dep * x_dep + w_type * x_type + w_cooccur * x_cooccur + b
        p = sigmoid(z)

    return {
        "E1": e1,
        "E2": e2,
        "description": description,
        "x": {
            "x_text":    round(x_text,    4),
            "x_dep":     round(x_dep,     4),
            "x_type":    round(x_type,    4),
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
    """
    Compute calibrated probabilities for relationships.

    Input (from LLM):
        - JSON string
        - dict with key "relationships"
        - list of relationship dicts

    Output (normalized):
        List of dicts with keys:
        {e1, e2, relation, confidence, P(R)}
    """

    # -------- Parse input safely --------
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
        # -------- Validate LLM schema --------
        if "entity_1" not in rel or "entity_2" not in rel:
            raise ValueError(f"Invalid relationship format: {rel}")

        e1 = rel["entity_1"].strip().upper()
        e2 = rel["entity_2"].strip().upper()
        #relation_name = rel.get("relation", "RELATED_TO")
        description = rel.get("description") or ""

        # -------- Compute probability --------
        prob_result = compute_relation_probability(
            e1,
            e2,
            description,
            processed_data
        )

        # -------- Normalize output schema --------
        normalized = {
            "e1": e1,
            "e2": e2,
            "p": prob_result.get("P(R|E1,E2,x)", 0.0)
        }

        results.append(normalized)

    return results


def compute_relationship_probabilities_2(data, processed_data, bias=None, weight=None):
    """
    Compute calibrated probabilities for relationships.

    Input (from LLM):
        - JSON string
        - dict with key "relationships"
        - list of relationship dicts

    Output (normalized):
        List of dicts with keys:
        {
          e1,
          e2,
          p,             # P(R | E1, E2, x)
          cardinality,   # "1:1" | "1:N" | "N:1" | "N:M"
          description    # optional, kept if needed
        }
    """

    # -------- Parse input safely --------
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
        # -------- Validate LLM schema --------
        if "entity_1" not in rel or "entity_2" not in rel:
            raise ValueError(f"Invalid relationship format: {rel}")

        e1 = rel["entity_1"].strip().upper()
        e2 = rel["entity_2"].strip().upper()
        description = rel.get("description") or ""
        cardinality = (rel.get("cardinality") or "").strip()
        associative = rel.get("associative_entity", None)

        # -------- Compute probability --------
        prob_result = compute_relation_probability(
            e1,
            e2,
            description,
            processed_data, 
            bias, 
            weight
        )

        # -------- Normalize output schema --------
        normalized = {
            "e1": e1,
            "e2": e2,
            "p": prob_result.get("P(R|E1,E2,x)", 0.0),
            "associative_entity": associative,
            "cardinality": cardinality
        }

        results.append(normalized)

    return results


def merge_relationships(*lists: List[Dict]) -> List[Dict]:
    """
    Merge multiple lists of relationships, keeping only E1, E2, p,
    and selecting the maximum probability if duplicates exist.
    
    Args:
        *lists: any number of lists containing dicts with keys:
                'E1', 'E2', 'P(R|E1,E2,x)'
    
    Returns:
        List[Dict]: merged list with keys 'e1', 'e2', 'p'
    """
    merged_dict = {}
    
    for lst in lists:
        for r in lst:
            e1, e2, p = r["E1"], r["E2"], r["P(R|E1,E2,x)"]
            key = (e1, e2)
            if key not in merged_dict or p > merged_dict[key]["p"]:
                merged_dict[key] = {"e1": e1, "e2": e2, "p": p}
    
    # Return list of dicts
    return list(merged_dict.values())
