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
from src.entity_processing import entity_similarity_checker, merge_entity_lists, compute_entity_probabilities, extract_llama_entities_from_text, extract_gpt_entities


def attribute_prompting(processed_data, inferred_entity):
    return f"""

    Task: Extract ALL attributes for each entity from the text below, following database design best practices.

    Original Description:
    {processed_data}

    Identified Entities:
    {inferred_entity}

    Instructions:
    1. For each entity, list:
    - Atomic attributes (single-valued)
    - Composite attributes (if any)
    - Multi-valued attributes (if any)
    2. Specify the primary key (PK) for each entity
    3. For each attribute, infer:
    - Data type (e.g., INT, TEXT, DATE)
    - Constraints (e.g., NOT NULL, UNIQUE)
    4. Format output as VALID JSON.

    Rules:
    - The fact table contains numerical or additive measurements used to compute quantitative answers about the business.
    The dimension tables give the complete descriptions of the dimensions of the business
    - Only include attributes EXPLICITLY mentioned in the text or logically required (e.g., PK).
    - PK must uniquely identify each entity
    - Multi-valued attributes should be marked with `is_array: true`
    - Composite attributes should have `is_composite: true` and list sub-attributes
    - Use consistent naming (snake_case for attributes, UPPER_SNAKE_CASE for entity)

    Output Format:
    - Use JSON format: {{ "attributes": {{ "ENTITY_1": [attr1, attr2, ...], "ENTITY_2": [...], ... }} }}

"""

def attribute_few_shot_prompting(processed_data, inferred_entity):
    return f"""

    Task: Extract ALL attributes for each entity from the text below, following database design best practices.

    Original Description:
    {processed_data}

    Identified Entities:
    {inferred_entity}

    Instructions:
    1. For each entity, list:
    - Atomic attributes (single-valued)
    - Composite attributes (if any)
    - Multi-valued attributes (if any)
    2. Specify the primary key (PK) for each entity
    3. For each attribute, infer:
    - Data type (e.g., INT, TEXT, DATE)
    - Constraints (e.g., NOT NULL, UNIQUE)
    4. Format output as VALID JSON.

    Rules:
    - Only include attributes EXPLICITLY mentioned in the text or logically required (e.g., PK).
    - PK must uniquely identify each entity.
    - For process/event entities (e.g., APPOINTMENT, GROOMING_SERVICE, REGISTRATION), include the attributes that describe THAT specific event/service, not the parent entities' attributes.
    - Use consistent naming (snake_case for attributes, UPPER_SNAKE_CASE for entity).
    - Do NOT duplicate attributes across entities.

    Output Format:
    - Use JSON format: {{ "attributes": {{ "ENTITY_1": [attr1, attr2, ...], "ENTITY_2": [...], ... }} }}

    Example 1 — system with a process/event entity:
    Description: A pet grooming salon manages groomers and pets. Groomer info includes name, specialization, years of experience, and rating. Pets have breed, age, weight, and fur characteristics. Each grooming service records service items, service duration, pricing, and customer rating.
    Given entities: ["GROOMER", "PET", "GROOMING_SERVICE"]
    Generated attributes:
    {{
        "attributes": {{
            "GROOMER": ["groomer_id", "name", "specialization", "years_of_experience", "rating"],
            "PET": ["pet_id", "breed", "age", "weight", "fur_characteristics"],
            "GROOMING_SERVICE": ["service_id", "service_items", "service_duration", "pricing", "customer_rating"]
        }}
    }}

    Example 2 — standard hospital system:
    Description: A hospital manages medical appointments. Patient info: name, date of birth, gender, phone, address. Doctor info: name, specialty, license number. Appointment: date, time, diagnosis, notes.
    Given entities: ["PATIENT", "DOCTOR", "APPOINTMENT"]
    Generated attributes:
    {{
        "attributes": {{
            "PATIENT": ["patient_id", "full_name", "date_of_birth", "gender", "phone_number", "address"],
            "DOCTOR": ["doctor_id", "full_name", "specialty", "license_number"],
            "APPOINTMENT": ["appointment_id", "appointment_date", "appointment_time", "diagnosis", "notes"]
        }}
    }}

"""

def normalize_name(name: str):
    """
    Convert snake_case to natural language:
    patient_id -> patient id
    date_of_birth -> date of birth
    """
    return re.sub(r"_", " ", name).strip().lower()

def normalize_schema(schema_dict):
    """
    Convert your original schema into:
    { ENTITY: [normalized_attribute names] }
    """
    output = {}
    for entity, attributes in schema_dict.items():
        normalized_attrs = []
        for attr in attributes:
            normalized_attrs.append(normalize_name(attr["name"]))
        output[entity] = normalized_attrs
    return output

import re
import json

# ---------------------------
# 1. Extract JSON text safely
# ---------------------------
def extract_json_block(text: str):
    """
    Extracts the outermost JSON object from LLM output.
    Handles both raw JSON and ```json ... ``` fenced blocks.
    Uses balanced-bracket matching to correctly handle nested objects.
    """
    if not text:
        raise ValueError("Empty input")

    # Strip ```json ... ``` fence if present (greedy to get full content)
    fence = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    candidate = fence.group(1) if fence else text

    # Find outermost { ... } using balanced bracket matching
    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in text")

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(candidate[start:], start=start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_str = candidate[start:i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON structure: {e}")

    raise ValueError("Unbalanced JSON braces in text")


# ---------------------------------------------------------
# 2. Extract only entity + list of attribute names (ignore others)
# ---------------------------------------------------------
def extract_attribute_names(llm_output: str):
    """
    Returns:
    {
        "ENTITY": ["attr1", "attr2", ...],
        ...
    }
    Handles both formats:
    - [{"name": "attr_name", ...}, ...]  (object list)
    - ["attr_name", ...]                 (string list)
    """
    data = extract_json_block(llm_output)

    if "attributes" not in data or not isinstance(data["attributes"], dict):
        raise ValueError("Expected a JSON object with field 'attributes'.")

    simplified = {}
    for entity, attrs in data["attributes"].items():
        names = []
        for attr in attrs:
            if isinstance(attr, dict):
                name = attr.get("name") or attr.get("attribute_name") or attr.get("column_name")
                if name:
                    names.append(str(name))
            elif isinstance(attr, str) and attr.strip():
                names.append(attr.strip())
        simplified[entity] = names

    return simplified


# -----------------------------------------------
# 3. Convert snake_case → natural language
# -----------------------------------------------
def normalize_name(name: str):
    """
    patient_id -> patient id
    date_of_birth -> date of birth
    """
    return re.sub(r"_", " ", name).strip().lower()


# -----------------------------------------------
# 4. Normalize schema
# -----------------------------------------------
def normalize_schema(schema_dict):
    """
    Converts attribute dict to flat normalized names.
    Handles both:
    - string list: ["patient_id", "first_name"]
    - object list: [{"name": "patient_id", ...}]
    """
    if not isinstance(schema_dict, dict):
        raise ValueError("schema_dict must be a dictionary.")

    output = {}
    for entity, attributes in schema_dict.items():
        normalized_attrs = []
        for attr in attributes:
            if isinstance(attr, dict):
                name = attr.get("name") or attr.get("attribute_name") or attr.get("column_name")
                if name:
                    normalized_attrs.append(normalize_name(str(name)))
            elif isinstance(attr, str) and attr.strip():
                normalized_attrs.append(normalize_name(attr))
        output[entity] = normalized_attrs

    return output


def merge_normalized_schemas(schema_a: dict, schema_b: dict):
    """
    Merge two normalized schemas produced by LLMs.
    Input format:
        { ENTITY: [list of normalized attributes] }
    Output:
        { ENTITY: sorted unique attributes }
    """
    merged = {}

    all_entities = set(schema_a.keys()) | set(schema_b.keys())

    for entity in all_entities:
        attrs_a = set(schema_a.get(entity, []))
        attrs_b = set(schema_b.get(entity, []))

        merged_attrs = sorted(attrs_a | attrs_b)
        merged[entity] = merged_attrs

    return merged


"""
Attribute Probability Computation
=================================
Compute P(Attribute | Entity) using:
  - Wikidata entity lookup
  - SPARQL property patterns
  - lexical fallback (SBERT cosine)
"""



# =====================================================
# 0. MODEL & BASIC UTILITIES
# =====================================================

sbert = SentenceTransformer("all-MiniLM-L6-v2")

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def normalize_cosine_to_01(sim):
    """Convert [-1,1] → [0,1]."""
    return float((sim + 1.0) / 2.0)

def score_from_counts(total_count, log_div=4.0):
    """Maps Wikidata counts → [0,1]"""
    if total_count <= 0:
        return 0.0
    return float(math.tanh(math.log1p(total_count) / log_div))


def compute_x_text_for_attribute(entity_name: str, attr_name: str, processed_text: str) -> float:
    """
    Text-evidence score: measures whether this (entity, attribute) pair
    is semantically supported by the source description.

    Encodes the query "<entity words> <attr_name>" with BERT and computes
    cosine similarity against the full processed text embedding.
    Returns a value in [0, 1] via normalize_cosine_to_01.
    """
    try:
        # Build a natural-language query from entity + attribute
        entity_words = entity_name.replace('_', ' ').lower()
        query = f"{entity_words} {attr_name}"
        query_emb = sbert.encode(query, convert_to_tensor=True)
        text_emb  = sbert.encode(processed_text, convert_to_tensor=True)
        sim = util.cos_sim(query_emb, text_emb).item()
        return normalize_cosine_to_01(sim)
    except Exception:
        return 0.5


# =====================================================
# 1. NETWORK SAFE REQUEST
# =====================================================

headers = {"User-Agent": "Attribute-KG-bot/1.0"}

def safe_request(url, params=None, timeout=30):
    try:
        r = requests.get(url, params=params, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r
    except:
        return None


# =====================================================
# 2. WIKIDATA SEARCH + SPARQL
# =====================================================

WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

@lru_cache(maxsize=1024)
def search_wikidata_qid(label, language="en"):
    """
    Search label on Wikidata → QID, label, description
    """
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": language,
        "format": "json",
        "limit": 3
    }

    r = safe_request(WIKIDATA_SEARCH_URL, params=params)
    if not r:
        return None, None, None

    hits = r.json().get("search", [])
    if not hits:
        return None, None, None

    top = hits[0]
    return top.get("id"), top.get("label"), top.get("description")


@lru_cache(maxsize=256)
def sparql_instance_property_counts(qid_entity, limit=50):
    """
    Count properties across instances of entity type.
    """
    query = f"""
    SELECT ?prop (COUNT(*) AS ?count) WHERE {{
      ?s wdt:P31/wdt:P279* wd:{qid_entity} .
      ?s ?prop ?o .
      FILTER(STRSTARTS(STR(?prop), "http://www.wikidata.org/prop/direct/"))
    }}
    GROUP BY ?prop
    ORDER BY DESC(?count)
    LIMIT {limit}
    """

    r = safe_request(WIKIDATA_SPARQL_URL, params={"query": query, "format": "json"})
    if not r:
        return []

    try:
        return [
            (b["prop"]["value"], int(b["count"]["value"]))
            for b in r.json()["results"]["bindings"]
        ]
    except:
        return []


@lru_cache(maxsize=256)
def sparql_property_label_matches(keywords_tuple, limit=50):
    """
    Find Wikidata properties containing keyword.
    """
    contains = " || ".join(
        f'CONTAINS(LCASE(?label), "{kw}")' for kw in keywords_tuple
    )

    query = f"""
    SELECT ?prop ?label WHERE {{
      ?prop a wikibase:Property .
      ?prop rdfs:label ?label .
      FILTER({contains})
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """

    r = safe_request(WIKIDATA_SPARQL_URL, params={"query": query, "format": "json"})
    if not r:
        return []

    try:
        return [
            (b["prop"]["value"], b.get("label", {}).get("value", ""))
            for b in r.json()["results"]["bindings"]
        ]
    except:
        return []


# =====================================================
# 3. COMPUTE x_type
# =====================================================

def compute_x_type(entity_name, attr_name, debug=False):
    """
    Compute x_type using:
      - Instance statistics
      - Property label match
      - Lexical cosine fallback
    """
    debug_info = {"entity": entity_name, "attribute": attr_name}

    # 1) Map entity → Wikidata QID
    qid, _, _ = search_wikidata_qid(entity_name)

    # ----- INSTANCE SCORE -----
    instance_score = 0.0
    prop_matches = []

    if qid:
        keywords = [w.lower() for w in attr_name.split() if len(w) > 1]
        prop_matches = sparql_property_label_matches(tuple(keywords))

        if prop_matches:
            props_count = sparql_instance_property_counts(qid)
            total = sum(c for _, c in props_count)
            instance_score = score_from_counts(total)

    debug_info["instance_score"] = instance_score

    # ----- PROPERTY LABEL SCORE -----
    prop_label_score = 0.0
    if prop_matches:
        n = len(prop_matches)
        prop_label_score = math.tanh(n / 5)

    debug_info["prop_label_score"] = prop_label_score

    # ----- LEXICAL FALLBACK -----
    try:
        e = sbert.encode(entity_name, convert_to_tensor=True)
        a = sbert.encode(attr_name, convert_to_tensor=True)
        sim = util.cos_sim(e, a).item()
        lexical_score = normalize_cosine_to_01(sim)
    except:
        lexical_score = 0.5

    debug_info["lexical_score"] = lexical_score

    # ----- DECISION -----
    if instance_score > 0:
        x = instance_score
        src = "instance"
    elif prop_label_score > 0:
        x = prop_label_score
        src = "property"
    else:
        x = lexical_score
        src = "lexical"

    x = max(0.0, min(1.0, x))

    debug_info.update({"x_type": x, "source": src})

    return (x, debug_info) if debug else x


# =====================================================
# 4. FINAL PROBABILITY
# =====================================================


def compute_attribute_probability(
    entity_name: str,
    attr_name: str,
    bias: float,
    weight: float,
    processed_text: str = None,
):
    """
    Estimate P(Attribute | Entity, Text).

    x = 0.6 * x_text + 0.4 * x_type   (when processed_text is provided)
      x_text: BERT similarity of "<entity> <attr>" vs source description
              — checks if the attribute is actually mentioned in the text
      x_type: Wikidata / lexical type-based plausibility score
              — fallback signal based on knowledge-graph priors

    Without processed_text: falls back to x_type only (legacy behaviour).
    """
    x_type, dbg = compute_x_type(entity_name, attr_name, debug=True)

    if processed_text:
        x_text = compute_x_text_for_attribute(entity_name, attr_name, processed_text)
        # Text evidence weighted higher: we only want attributes mentioned in the description
        x = 0.6 * x_text + 0.4 * x_type
    else:
        x_text = x_type
        x = x_type

    z = weight * x + bias
    p = sigmoid(z)

    return {
        "Entity":              entity_name,
        "Attribute":           attr_name,
        "x_type":              round(x_type, 4),
        "x_text":              round(x_text, 4),
        "x_combined":          round(x,      4),
        "P(Attribute|Entity)": round(p,      6),
        "debug": dbg | {"weight": weight, "bias": bias, "z": round(z, 4)},
    }


def compute_all_attribute_probabilities(
    normalized_schema: dict,
    bias: float,
    weight: float,
    processed_text: str = None,
):
    """
    Compute P(Attribute | Entity, Text) for all entities in the schema.

    Input:
        normalized_schema: { "ENTITY": ["attr name", ...], ... }
        bias, weight: sigmoid calibration parameters
        processed_text: source description (enables x_text feature)

    Output:
        { "ENTITY": [ {prob_dict}, ... ], ... }
    """
    results = {}

    for entity, attr_list in normalized_schema.items():
        print(f"\n=== Entity: {entity} ===")
        entity_results = []
        for attr in attr_list:
            prob = compute_attribute_probability(
                entity, attr,
                bias=bias, weight=weight,
                processed_text=processed_text
            )
            entity_results.append(prob)
            print(prob)
        results[entity] = entity_results

    return results


def extract_attribute_probs(attribute_data: Dict[str, List[Dict]]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Extract attribute probabilities in simplified format.
    
    Args:
        attribute_data: dict, key=entity, value=list of attribute dicts
                        each dict contains at least 'Attribute' and 'P(Attribute|Entity)'
                        
    Returns:
        Dict[str, List[Tuple[str, float]]]: mapping from entity to list of (attribute, probability)
    """
    result = {}
    for entity, attr_list in attribute_data.items():
        simplified = [(attr["Attribute"], attr["P(Attribute|Entity)"]) for attr in attr_list]
        result[entity] = simplified
    return result