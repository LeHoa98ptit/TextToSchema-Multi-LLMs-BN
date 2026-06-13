"""
Ablation Study — Attribute Processing (no Wikidata)
=====================================================
This module is a drop-in replacement for src/attribute_processing.py for the
ablation experiment that evaluates the contribution of Wikidata knowledge.

What changed vs. the original:
  - REMOVED: search_wikidata_qid(), sparql_instance_property_counts(),
             sparql_property_label_matches(), compute_x_type()
             and all Wikidata URL constants.
  - CHANGED: compute_attribute_probability() now uses x_text as the sole
             signal.  Formula: x = x_text  (was: x = 0.6*x_text + 0.4*x_type)
  - CHANGED: When processed_text is absent the score defaults to 0.5
             (neutral prior) instead of falling back to x_type.
  - REMOVED: 'x_type' key from the output dict of compute_attribute_probability().

Everything else (prompting templates, JSON parsing, schema normalisation,
entity-level helpers) is identical to the original file.

Usage:
    from ablation.src.attribute_processing_ablation import (
        compute_all_attribute_probabilities,
        normalize_schema,
        ...
    )
"""

import warnings
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)
import re
import json
import math
import requests
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util

from ablation.src.entity_processing_ablation import (
    entity_similarity_checker,
    merge_entity_lists,
    compute_entity_probabilities,
    extract_llama_entities_from_text,
    extract_gpt_entities,
)


# =====================================================
# PROMPTING (unchanged from original)
# =====================================================

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


# =====================================================
# JSON EXTRACTION UTILITIES (unchanged)
# =====================================================

def extract_json_block(text: str):
    if not text:
        raise ValueError("Empty input")
    fence = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    candidate = fence.group(1) if fence else text
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


def extract_attribute_names(llm_output: str):
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


def normalize_name(name: str):
    return re.sub(r"_", " ", name).strip().lower()


def normalize_schema(schema_dict):
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
    merged = {}
    all_entities = set(schema_a.keys()) | set(schema_b.keys())
    for entity in all_entities:
        attrs_a = set(schema_a.get(entity, []))
        attrs_b = set(schema_b.get(entity, []))
        merged[entity] = sorted(attrs_a | attrs_b)
    return merged


# =====================================================
# MODEL & UTILITIES
# =====================================================

sbert = SentenceTransformer("all-MiniLM-L6-v2")

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def normalize_cosine_to_01(sim):
    return float((sim + 1.0) / 2.0)


# =====================================================
# TEXT-BASED SCORE (x_text only — no Wikidata)
# =====================================================

def compute_x_text_for_attribute(entity_name: str, attr_name: str, processed_text: str) -> float:
    """
    BERT similarity of "<entity> <attr>" vs the source description.
    This is the only signal used in the ablation (Wikidata x_type removed).
    """
    try:
        entity_words = entity_name.replace('_', ' ').lower()
        query = f"{entity_words} {attr_name}"
        query_emb = sbert.encode(query, convert_to_tensor=True)
        text_emb  = sbert.encode(processed_text, convert_to_tensor=True)
        sim = util.cos_sim(query_emb, text_emb).item()
        return normalize_cosine_to_01(sim)
    except Exception:
        return 0.5


# =====================================================
# PROBABILITY COMPUTATION (text-only, no Wikidata)
# =====================================================

def compute_attribute_probability(
    entity_name: str,
    attr_name: str,
    bias: float,
    weight: float,
    processed_text: str = None,
):
    """
    Ablation: P(Attribute | Entity, Text) using only x_text.
    Wikidata x_type is removed. Falls back to 0.5 when no text is provided.
    """
    if processed_text:
        x = compute_x_text_for_attribute(entity_name, attr_name, processed_text)
        x_text = x
    else:
        x_text = 0.5
        x = 0.5

    z = weight * x + bias
    p = sigmoid(z)

    return {
        "Entity":              entity_name,
        "Attribute":           attr_name,
        "x_text":              round(x_text, 4),
        "x_combined":          round(x,      4),
        "P(Attribute|Entity)": round(p,      6),
        "debug": {"weight": weight, "bias": bias, "z": round(z, 4)},
    }


def compute_all_attribute_probabilities(
    normalized_schema: dict,
    bias: float,
    weight: float,
    processed_text: str = None,
):
    results = {}
    for entity, attr_list in normalized_schema.items():
        print(f"\n=== Entity: {entity} ===")
        entity_results = []
        for attr in attr_list:
            prob = compute_attribute_probability(
                entity, attr,
                bias=bias, weight=weight,
                processed_text=processed_text,
            )
            entity_results.append(prob)
            print(prob)
        results[entity] = entity_results
    return results


def extract_attribute_probs(attribute_data: Dict[str, List[Dict]]) -> Dict[str, List[Tuple[str, float]]]:
    result = {}
    for entity, attr_list in attribute_data.items():
        result[entity] = [(attr["Attribute"], attr["P(Attribute|Entity)"]) for attr in attr_list]
    return result
