"""
Fix no-attr exercises in main opt folder, save results to output/add/optimization/.

Two root causes handled:
  A) ILP_dropped_attr   – prob file correct but old code stored attribut:[]
                          → just re-run optimization with current code
  B) attr_missing_in_prob – entity completely absent from prob file
                          → compute attr probs from generation file, patch prob, then optimize

Also applies no_isolated=True constraint.
"""
import os, json, sys, re, time

if 'CORENLP_HOME' not in os.environ:
    os.environ['CORENLP_HOME'] = ''

_orig_del = os.environ.__class__.__delitem__
def _safe_del(self, key):
    try: _orig_del(self, key)
    except KeyError:
        if key != 'CORENLP_HOME': raise
os.environ.__class__.__delitem__ = _safe_del

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.finding_the_best_er_model import JointERILPSolverFinal_1
from src.pre_processing import preprocess_text
from src.attribute_processing import compute_all_attribute_probabilities, extract_attribute_probs

# ── Paths ─────────────────────────────────────────────────────────────────────
MAIN_PRO = os.path.join(project_root, "output/probability/multi-llms/pro_fewshot_llama_0.5_1.0")
MAIN_GEN = os.path.join(project_root, "output/generation/multi-llms/few-shot-llama")
INPUT_TXT = os.path.join(project_root, "dataset/Datasets/Full-Dataset/input")
ADD_OPT   = os.path.join(project_root, "output/add/optimization")
os.makedirs(ADD_OPT, exist_ok=True)

LAMBDA_E, LAMBDA_R, LAMBDA_A = 1.2, -0.5, 1.0
MIN_ENTITIES = 3
BIAS, WEIGHT = 0.5, 1.0

# All 55 unique problematic exercises (isolated + no-attr from main folder)
ALL_PROBLEM = sorted(set([
    # no-attr: attr_missing_in_prob
    261,270,277,284,286,294,316,339,347,353,373,379,385,401,437,448,464,479,488,492,498,
    # no-attr: ILP_dropped_attr
    256,306,332,347,395,411,424,428,438,465,470,479,484,498,
    # isolated: ILP_dropped_rel (re-opt with no_isolated will fix)
    292,313,331,345,346,368,396,398,408,411,413,419,423,424,428,438,451,462,469,478,489,490,
    # isolated: no_rel_in_prob (won't fix isolated but at least get attrs right)
    346,379,388,392,406,410,423,436,451,469,
]))

print(f"Processing {len(ALL_PROBLEM)} exercises\n")


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def patch_prob_file(ex_num):
    """
    Compute attribute probabilities for entities that are completely missing
    from the prob file's attribute section.  Updates the prob file in-place.
    Returns True if changes were made.
    """
    pro_path = os.path.join(MAIN_PRO, f"{ex_num}.json")
    gen_path = os.path.join(MAIN_GEN, f"{ex_num}.json")
    txt_path = os.path.join(INPUT_TXT, f"{ex_num}.txt")

    if not os.path.exists(pro_path): return False
    if not os.path.exists(gen_path): return False
    if not os.path.exists(txt_path): return False

    pro = load_json(pro_path)
    gen = load_json(gen_path)

    pro_entity = pro.get("entity", {})
    pro_attr   = pro.get("attribute", {})
    gen_attrs  = gen.get("attribut", gen.get("attribute", {}))
    gen_ents   = gen.get("entity", [])

    # Find entities present in prob entity dict but missing from prob attr dict
    missing_in_attr = [e for e in pro_entity if e not in pro_attr or not pro_attr.get(e)]

    if not missing_in_attr:
        return False  # nothing to fix

    raw_text = open(txt_path, encoding='utf-8').read().strip()
    processed_text = preprocess_text(raw_text)

    changed = False
    for e in missing_in_attr:
        # Get attr list from generation file (key might be different case)
        attr_list = None
        for k in gen_attrs:
            if k.strip().upper() == e.strip().upper():
                attr_list = gen_attrs[k]
                break

        if not attr_list:
            # Entity not in gen attrs either → try ASSOC pattern or skip
            # For ASSOC_ entities created from N:M, use entity prob as attr prob
            if e.upper().startswith("ASSOC_"):
                ep = float(pro_entity.get(e, 0.5))
                # generate standard FK/PK names
                rest = e[6:] if e.upper().startswith("ASSOC_") else e
                parts = rest.lower().split("_")
                pk  = f"{rest.lower()}_id"
                attrs_dict = {pk: ep}
                if len(parts) >= 2:
                    attrs_dict[f"{parts[0]}_id"] = ep
                    attrs_dict[f"_".join(parts[1:])+"_id"] = ep
                pro_attr[e] = attrs_dict
                changed = True
            # else: no attrs available, skip
            continue

        if not isinstance(attr_list, list) or not attr_list:
            continue

        # Compute attribute probabilities
        try:
            raw_probs = compute_all_attribute_probabilities(
                {e: attr_list}, bias=BIAS, weight=WEIGHT, processed_text=processed_text)
            extracted = extract_attribute_probs(raw_probs)
            if e in extracted and extracted[e]:
                pro_attr[e] = {a: p for a, p in extracted[e]}
                changed = True
        except Exception as ex:
            print(f"  [WARN] Could not compute attrs for {e} in {ex_num}: {ex}")

    if changed:
        pro["attribute"] = pro_attr
        save_json(pro_path, pro)

    return changed


def run_optimization(ex_num):
    pro_path = os.path.join(MAIN_PRO, f"{ex_num}.json")
    out_path = os.path.join(ADD_OPT,  f"{ex_num}.json")

    if not os.path.exists(pro_path):
        print(f"  Ex {ex_num}: NO PROB FILE")
        return False

    try:
        data = load_json(pro_path)
        entities_prob      = data.get("entity", {})
        attributes_prob    = data.get("attribute", {})
        relationships_prob = data.get("relationship", [])

        solver = JointERILPSolverFinal_1(entities_prob, relationships_prob, attributes_prob)
        score, sel_e, sel_r, sel_a, runtime = solver.solve(
            lambda_E=LAMBDA_E, lambda_R=LAMBDA_R, lambda_A=LAMBDA_A,
            min_entities=MIN_ENTITIES, no_isolated=True,
        )

        formatted_rels = [
            {"entity_1": r.get("entity_1"), "entity_2": r.get("entity_2"),
             "cardinality": r.get("cardinality","1:N"),
             "associative_entity": r.get("associative_entity")}
            for r in sel_r
        ]

        result = {"entity": sel_e, "attribut": sel_a, "relationship": formatted_rels}
        save_json(out_path, result)

        n_a = sum(len(v) for v in sel_a.values())
        connected = set()
        for r in formatted_rels:
            connected.add(r["entity_1"]); connected.add(r["entity_2"])
        iso = [e for e in sel_e if e not in connected]
        iso_str = f" ISO={iso}" if iso else ""
        noattr = [e for e in sel_e if not sel_a.get(e)]
        na_str = f" NO_ATTR={noattr}" if noattr else ""
        print(f"  Ex {ex_num}: E={len(sel_e)} A={n_a} R={len(formatted_rels)}{iso_str}{na_str}")
        return True
    except Exception as e:
        print(f"  Ex {ex_num}: ERROR {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 65)
print("Step 1: Patching prob files for attr_missing_in_prob cases")
print("=" * 65)
patched = []
for ex in ALL_PROBLEM:
    changed = patch_prob_file(ex)
    if changed:
        patched.append(ex)
        print(f"  Patched prob: {ex}")
print(f"  Total patched: {len(patched)}")

print()
print("=" * 65)
print("Step 2: Re-running optimization (no_isolated=True)")
print("=" * 65)
ok, fail = 0, []
for ex in ALL_PROBLEM:
    if run_optimization(ex):
        ok += 1
    else:
        fail.append(ex)

print()
print("=" * 65)
print(f"Done: {ok}/{len(ALL_PROBLEM)}")
if fail: print(f"Failed: {fail}")

# ── Verification ─────────────────────────────────────────────────────────────
print()
print("── Verification ──")
for ex in ALL_PROBLEM:
    p = os.path.join(ADD_OPT, f"{ex}.json")
    if not os.path.exists(p):
        print(f"  {ex}: NOT GENERATED")
        continue
    d = load_json(p)
    ents  = d.get("entity", [])
    attrs = d.get("attribut", d.get("attribute", {}))
    rels  = d.get("relationship", [])
    n_a   = sum(len(v) for v in attrs.values()) if isinstance(attrs, dict) else 0
    connected = {r.get("entity_1") for r in rels} | {r.get("entity_2") for r in rels}
    iso    = [e for e in ents if e not in connected]
    noattr = [e for e in ents if not (attrs.get(e) if isinstance(attrs, dict) else [])]
    flags  = ""
    if n_a == 0:   flags += " attr=0"
    if not rels:   flags += " rel=0"
    if iso:        flags += f" ISO={iso}"
    if noattr:     flags += f" noattr={noattr}"
    print(f"  {ex}: E={len(ents)} A={n_a} R={len(rels)}{flags}")
