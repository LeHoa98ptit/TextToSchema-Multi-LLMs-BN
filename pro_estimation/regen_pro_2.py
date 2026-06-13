"""
Probability estimation for 10 regenerated exercises that previously had no relationships.
Target IDs: 363, 408, 410, 411, 413, 423, 424, 428, 451, 469

Input  generation : output/generation/multi-llms/few-shot-llama/
Input  text       : dataset/Datasets/Full-Dataset/input/
Output probability: output/probability/multi-llms/pro_fewshot_llama_0.5_1.0_ver2/

Parameters: BIAS=0.5, WEIGHT=1.0
Relationship weights from train/dataset/relationship_parameters.json:
  w_text=0.0373, w_dep=1.9500, w_type=0.7028, w_cooccur=1.1879, bias=-1.3886
"""

import os
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress CORENLP_HOME KeyError on exit
if "CORENLP_HOME" not in os.environ:
    os.environ["CORENLP_HOME"] = ""
_orig_delitem = os.environ.__class__.__delitem__
def _safe_del(self, key):
    try: _orig_delitem(self, key)
    except KeyError:
        if key != "CORENLP_HOME": raise
os.environ.__class__.__delitem__ = _safe_del

from src.pre_processing import preprocess_text
from src.entity_processing import (
    entity_similarity_checker,
    compute_entity_probabilities,
    extract_entity_probs,
)
from src.attribute_processing import (
    compute_all_attribute_probabilities,
    extract_attribute_probs,
)
from src.relationship_processing import compute_relationship_probabilities_2

# ── Config ────────────────────────────────────────────────────────────────────

TARGET_IDS    = [363, 408, 410, 411, 413, 423, 424, 428, 451, 469]
BIAS          = 0.5
WEIGHT        = 1.0
MAX_WORKERS   = 4

INPUT_TXT_DIR = os.path.join(project_root, "dataset/Datasets/Full-Dataset/input")
INPUT_JSON_DIR = os.path.join(project_root, "output/generation/multi-llms/few-shot-llama")
OUTPUT_DIR     = os.path.join(project_root, "output/probability/multi-llms/pro_fewshot_llama_0.5_1.0_ver2")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Helpers (same as main estimation script) ─────────────────────────────────

def get_rel_prob(rel_probs_data, e1, e2):
    if not e1 or not e2: return 0.0
    e1s, e2s = str(e1).strip().upper(), str(e2).strip().upper()
    if isinstance(rel_probs_data, list):
        for item in rel_probs_data:
            if isinstance(item, dict):
                i1 = str(item.get("entity_1", item.get("e1", ""))).strip().upper()
                i2 = str(item.get("entity_2", item.get("e2", ""))).strip().upper()
                if (i1 == e1s and i2 == e2s) or (i1 == e2s and i2 == e1s):
                    return float(item.get("probability", item.get("p", 0.0)))
    elif isinstance(rel_probs_data, dict):
        for k, v in rel_probs_data.items():
            if e1s in str(k).upper() and e2s in str(k).upper():
                return float(v)
    return 0.0


def get_attr_prob(attribute_probs, ent, attr):
    if not ent or not attr: return 0.0
    ent_str  = str(ent).strip().upper()
    attr_clean = re.sub(r"[^a-zA-Z0-9]", "", str(attr).strip().lower())

    def extract_val(v):
        try:
            if isinstance(v, (float, int)): return float(v)
            if isinstance(v, str): return float(v)
            if isinstance(v, dict):
                for pk in ["P(Attribute|Entity)", "probability", "p", "prob"]:
                    if pk in v: return float(v[pk])
        except: pass
        return None

    if isinstance(attribute_probs, list):
        for item in attribute_probs:
            if isinstance(item, dict):
                i_ent  = str(item.get("Entity", item.get("entity", ""))).strip().upper()
                i_attr = re.sub(r"[^a-zA-Z0-9]", "", str(item.get("Attribute", item.get("attribute", ""))).strip().lower())
                if i_ent == ent_str and i_attr == attr_clean:
                    val = extract_val(item)
                    if val is not None: return val
    elif isinstance(attribute_probs, dict):
        for k, v in attribute_probs.items():
            if isinstance(k, tuple) and len(k) >= 2:
                if str(k[0]).strip().upper() == ent_str:
                    k_attr = re.sub(r"[^a-zA-Z0-9]", "", str(k[1]).lower())
                    if k_attr == attr_clean:
                        val = extract_val(v)
                        if val is not None: return val
                continue
            k_str = str(k)
            if k_str.strip().upper() == ent_str and isinstance(v, dict):
                for ak, av in v.items():
                    if re.sub(r"[^a-zA-Z0-9]", "", str(ak).lower()) == attr_clean:
                        val = extract_val(av)
                        if val is not None: return val
            elif k_str.strip().upper() == ent_str and isinstance(v, list):
                # format: {'ENTITY': [('attr_name', prob), ...]}
                for item in v:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        if re.sub(r"[^a-zA-Z0-9]", "", str(item[0]).lower()) == attr_clean:
                            val = extract_val(item[1])
                            if val is not None: return val
    return 0.0


def build_output(gen_data, entities, attributes, relationships,
                 entity_probs, attribute_probs, rel_probs):
    out_entities   = {ent: entity_probs.get(ent, 0.0) for ent in entities}
    out_attributes = {ent: {a: get_attr_prob(attribute_probs, ent, a) for a in attrs}
                      for ent, attrs in attributes.items()}
    out_relationships = []

    for rel in relationships:
        e1, e2 = rel.get("entity_1"), rel.get("entity_2")
        prob = get_rel_prob(rel_probs, e1, e2)
        card = str(rel.get("cardinality", "")).upper()

        if card in ("N:M", "M:N"):
            assoc_info = rel.get("associative_entity")
            if assoc_info and isinstance(assoc_info, dict) and assoc_info.get("name"):
                assoc_name  = str(assoc_info["name"]).strip().upper()
                assoc_attrs = assoc_info.get("attributes", [])
            else:
                assoc_name  = f"ASSOC_{e1}_{e2}".upper()
                assoc_attrs = []

            out_entities[assoc_name] = max(out_entities.get(assoc_name, 0.0), prob)
            if assoc_name not in out_attributes:
                out_attributes[assoc_name] = {}
            pk = f"{assoc_name.lower()}_id"
            out_attributes[assoc_name][pk] = max(out_attributes[assoc_name].get(pk, 0.0), prob)
            for fk in [f"{e1.lower()}_id", f"{e2.lower()}_id"]:
                out_attributes[assoc_name][fk] = max(out_attributes[assoc_name].get(fk, 0.0), prob)
            for a in assoc_attrs:
                out_attributes[assoc_name][a] = max(out_attributes[assoc_name].get(a, 0.0), prob)

            for new_rel in [
                {"entity_1": e1, "entity_2": assoc_name, "cardinality": "1:N",
                 "associative_entity": None, "probability": prob},
                {"entity_1": e2, "entity_2": assoc_name, "cardinality": "1:N",
                 "associative_entity": None, "probability": prob},
            ]:
                merged = False
                for r in out_relationships:
                    if r["entity_1"] == new_rel["entity_1"] and r["entity_2"] == new_rel["entity_2"]:
                        r["probability"] = max(r["probability"], prob); merged = True; break
                if not merged:
                    out_relationships.append(new_rel)
        else:
            rel_copy = dict(rel)
            rel_copy["probability"] = prob
            out_relationships.append(rel_copy)

    out = dict(gen_data)
    out["entity"]        = out_entities
    out.pop("attribut", None)
    out["attribute"]     = out_attributes
    out["relationship"]  = out_relationships
    return out


# ── Process one file ──────────────────────────────────────────────────────────

def process(ex_id):
    fname     = f"{ex_id}.json"
    txt_path  = os.path.join(INPUT_TXT_DIR,  f"{ex_id}.txt")
    json_path = os.path.join(INPUT_JSON_DIR, fname)
    out_path  = os.path.join(OUTPUT_DIR,     fname)

    if not os.path.exists(txt_path):
        print(f"  [SKIP] {ex_id}: no text file"); return False
    if not os.path.exists(json_path):
        print(f"  [SKIP] {ex_id}: no generation file"); return False

    try:
        with open(txt_path,  encoding="utf-8") as f: raw_text  = f.read().strip()
        with open(json_path, encoding="utf-8") as f: gen_data  = json.load(f)

        processed_text = preprocess_text(raw_text)
        entities       = gen_data.get("entity", [])
        attributes     = gen_data.get("attribut", gen_data.get("attribute", {}))
        relationships  = gen_data.get("relationship", [])

        entity_probs    = extract_entity_probs(
            compute_entity_probabilities(
                entity_similarity_checker(processed_text, entities),
                bias=BIAS, weight=WEIGHT
            )
        )
        attribute_probs = extract_attribute_probs(
            compute_all_attribute_probabilities(
                attributes, bias=BIAS, weight=WEIGHT, processed_text=processed_text
            )
        )
        rel_probs = compute_relationship_probabilities_2(
            json.dumps(relationships), processed_text, bias=BIAS, weight=WEIGHT
        )

        out_data = build_output(gen_data, entities, attributes, relationships,
                                entity_probs, attribute_probs, rel_probs)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        n_rels = len(out_data.get("relationship", []))
        print(f"  [OK] {ex_id} → E:{len(entities)} R:{n_rels}")
        return True

    except Exception as e:
        print(f"  [ERROR] {ex_id}: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Probability estimation for {len(TARGET_IDS)} exercises")
    print(f"  BIAS={BIAS}  WEIGHT={WEIGHT}")
    print(f"  Output → {OUTPUT_DIR}")
    print("=" * 60)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, ex_id): ex_id for ex_id in TARGET_IDS}
        done = sum(1 for f in as_completed(futures) if f.result())

    print(f"\nDone: {done}/{len(TARGET_IDS)} files processed.")
