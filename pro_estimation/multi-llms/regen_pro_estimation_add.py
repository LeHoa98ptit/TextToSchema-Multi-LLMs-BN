"""
Compute probabilities for 18 regenerated files in output/add/generation/.
BIAS=0.5, WEIGHT=1.0 (same config as opt_fewshot_llama_0.5_1.0).
Output: output/add/probability/
"""
import os, json, sys, time, re

if 'CORENLP_HOME' not in os.environ:
    os.environ['CORENLP_HOME'] = ''

_orig_delitem = os.environ.__class__.__delitem__
def safe_environ_delitem(self, key):
    try:
        _orig_delitem(self, key)
    except KeyError:
        if key != 'CORENLP_HOME':
            raise
os.environ.__class__.__delitem__ = safe_environ_delitem

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

INPUT_TXT_FOLDER = os.path.join(project_root, "dataset/Datasets/Full-Dataset/input")
INPUT_JSON_FOLDER = os.path.join(project_root, "output/add/generation")
OUTPUT_FOLDER     = os.path.join(project_root, "output/add/probability")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

BIAS   = 0.5
WEIGHT = 1.0


def get_rel_prob(rel_probs_data, e1, e2):
    if not e1 or not e2:
        return 0.0
    e1_str, e2_str = str(e1).strip().upper(), str(e2).strip().upper()
    if isinstance(rel_probs_data, list):
        for item in rel_probs_data:
            if isinstance(item, dict):
                i_e1 = str(item.get("entity_1", item.get("e1", ""))).strip().upper()
                i_e2 = str(item.get("entity_2", item.get("e2", ""))).strip().upper()
                if (i_e1 == e1_str and i_e2 == e2_str) or (i_e1 == e2_str and i_e2 == e1_str):
                    return float(item.get("probability", item.get("p", 0.0)))
    return 0.0


def get_attr_prob(attribute_probs, ent, attr):
    if not ent or not attr:
        return 0.0
    ent_str   = str(ent).strip().upper()
    attr_clean = re.sub(r'[^a-zA-Z0-9]', '', str(attr).strip().lower())

    if isinstance(attribute_probs, list):
        for item in attribute_probs:
            if not isinstance(item, dict):
                continue
            i_ent  = str(item.get("Entity", item.get("entity", ""))).strip().upper()
            i_attr = re.sub(r'[^a-zA-Z0-9]', '', str(item.get("Attribute", item.get("attribute", ""))).strip().lower())
            if i_ent == ent_str and i_attr == attr_clean:
                for pk in ["P(Attribute|Entity)", "probability", "p", "prob"]:
                    if pk in item:
                        try:
                            return float(item[pk])
                        except Exception:
                            pass
    elif isinstance(attribute_probs, dict):
        for k, v in attribute_probs.items():
            if str(k).strip().upper() == ent_str:
                if isinstance(v, dict):
                    for ak, av in v.items():
                        ak_clean = re.sub(r'[^a-zA-Z0-9]', '', str(ak).lower())
                        if ak_clean == attr_clean:
                            try:
                                return float(av)
                            except Exception:
                                pass
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, tuple) and len(item) == 2:
                            ak, av = item
                            ak_clean = re.sub(r'[^a-zA-Z0-9]', '', str(ak).lower())
                            if ak_clean == attr_clean:
                                try:
                                    return float(av)
                                except Exception:
                                    pass
                        elif isinstance(item, dict):
                            ak = item.get("Attribute", item.get("attribute", ""))
                            ak_clean = re.sub(r'[^a-zA-Z0-9]', '', str(ak).lower())
                            if ak_clean == attr_clean:
                                for pk in ["P(Attribute|Entity)", "probability", "p", "prob"]:
                                    if pk in item:
                                        try:
                                            return float(item[pk])
                                        except Exception:
                                            pass
    return 0.0


def build_output_data(gen_data, entities, attributes, relationships,
                      entity_probs, attribute_probs, rel_probs):
    out_entities = {ent: entity_probs.get(ent, 0.0) for ent in entities}

    out_attributes = {}
    for ent, attrs in attributes.items():
        out_attributes[ent] = {attr: get_attr_prob(attribute_probs, ent, attr) for attr in attrs}

    out_relationships = []
    for rel in relationships:
        e1, e2 = rel.get("entity_1"), rel.get("entity_2")
        prob = get_rel_prob(rel_probs, e1, e2)
        card = str(rel.get("cardinality", "")).upper()

        if card in ["N:M", "M:N"]:
            assoc_info = rel.get("associative_entity")
            if assoc_info and isinstance(assoc_info, dict) and assoc_info.get("name"):
                assoc_name = str(assoc_info.get("name")).strip().upper()
                assoc_attrs = assoc_info.get("attributes", [])
            else:
                assoc_name = f"ASSOC_{e1}_{e2}".upper()
                assoc_attrs = []

            if assoc_name not in out_entities:
                out_entities[assoc_name] = prob
            else:
                out_entities[assoc_name] = max(out_entities[assoc_name], prob)

            if assoc_name not in out_attributes:
                out_attributes[assoc_name] = {}

            pk_name = f"{assoc_name.lower()}_id"
            out_attributes[assoc_name][pk_name] = max(
                out_attributes[assoc_name].get(pk_name, 0.0), prob)

            for fk in [f"{e1.lower()}_id", f"{e2.lower()}_id"]:
                out_attributes[assoc_name][fk] = max(
                    out_attributes[assoc_name].get(fk, 0.0), prob)

            for a in assoc_attrs:
                out_attributes[assoc_name][a] = max(
                    out_attributes[assoc_name].get(a, 0.0), prob)

            for new_rel in [
                {"entity_1": e1, "entity_2": assoc_name, "cardinality": "1:N",
                 "associative_entity": None, "probability": prob},
                {"entity_1": e2, "entity_2": assoc_name, "cardinality": "1:N",
                 "associative_entity": None, "probability": prob},
            ]:
                merged = False
                for r in out_relationships:
                    if r.get("entity_1") == new_rel["entity_1"] and r.get("entity_2") == new_rel["entity_2"]:
                        r["probability"] = max(r.get("probability", 0.0), new_rel["probability"])
                        merged = True
                        break
                if not merged:
                    out_relationships.append(new_rel)
        else:
            rel_copy = dict(rel)
            rel_copy["probability"] = prob
            out_relationships.append(rel_copy)

    out_data = dict(gen_data)
    out_data["entity"] = out_entities
    out_data.pop("attribut", None)
    out_data["attribute"] = out_attributes
    out_data["relationship"] = out_relationships
    return out_data


def process_file(filename):
    txt_path  = os.path.join(INPUT_TXT_FOLDER,  filename.replace(".json", ".txt"))
    json_path = os.path.join(INPUT_JSON_FOLDER, filename)
    out_path  = os.path.join(OUTPUT_FOLDER,     filename)

    if not os.path.exists(txt_path):
        print(f"  [SKIP] missing txt: {txt_path}")
        return False
    if not os.path.exists(json_path):
        print(f"  [SKIP] missing json: {json_path}")
        return False

    start = time.time()
    with open(txt_path,  encoding="utf-8") as f: raw_text  = f.read().strip()
    with open(json_path, encoding="utf-8") as f: gen_data  = json.load(f)

    processed_text = preprocess_text(raw_text)
    entities       = gen_data.get("entity", [])
    attributes     = gen_data.get("attribute", gen_data.get("attribut", {}))
    relationships  = gen_data.get("relationship", [])

    entity_probs    = extract_entity_probs(
        compute_entity_probabilities(
            entity_similarity_checker(processed_text, entities),
            bias=BIAS, weight=WEIGHT))

    attribute_probs = extract_attribute_probs(
        compute_all_attribute_probabilities(
            attributes, bias=BIAS, weight=WEIGHT, processed_text=processed_text))

    rel_probs = compute_relationship_probabilities_2(
        json.dumps(relationships), processed_text, bias=BIAS, weight=WEIGHT)

    out_data = build_output_data(
        gen_data, entities, attributes, relationships,
        entity_probs, attribute_probs, rel_probs)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)

    elapsed = round(time.time() - start, 2)
    n_rels = len(out_data.get("relationship", []))
    n_ents = len(out_data.get("entity", {}))
    n_attrs = sum(len(v) for v in out_data.get("attribute", {}).values())
    print(f"  [OK] {filename}  E={n_ents} A={n_attrs} R={n_rels}  ({elapsed}s)")
    return True


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(INPUT_JSON_FOLDER) if f.endswith(".json"))
    print("=" * 65)
    print(f"Probability estimation for {len(files)} files")
    print(f"  BIAS={BIAS}  WEIGHT={WEIGHT}")
    print(f"  Input : {INPUT_JSON_FOLDER}")
    print(f"  Output: {OUTPUT_FOLDER}")
    print("=" * 65)

    ok, fail = 0, []
    for fname in files:
        print(f"\n[{fname}]")
        try:
            if process_file(fname):
                ok += 1
            else:
                fail.append(fname)
        except Exception as e:
            print(f"  [ERROR] {e}")
            fail.append(fname)

    print("\n" + "=" * 65)
    print(f"Done: {ok}/{len(files)} succeeded")
    if fail:
        print(f"Failed: {fail}")

    # Quick verification
    print("\n── Verification ──")
    for fname in files:
        p = os.path.join(OUTPUT_FOLDER, fname)
        if os.path.exists(p):
            with open(p) as f: d = json.load(f)
            n_r = len(d.get("relationship", []))
            n_a = sum(len(v) for v in d.get("attribute", {}).values())
            flag = " ← WARN" if (n_a == 0 or n_r == 0) else ""
            print(f"  {fname}: E={len(d.get('entity',{}))} A={n_a} R={n_r}{flag}")
        else:
            print(f"  {fname}: NOT GENERATED")
