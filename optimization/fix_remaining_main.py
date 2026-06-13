"""
Fix remaining problematic exercises in main opt folder, save to output/add/optimization/.

Issues handled:
  A) no_rel_in_prob (379,388,392,406,410) – prob file has 0 rels despite gen file having rels
                                            → compute rel probs from gen+text, patch prob, re-opt
  B) spurious_entity (347,492)            – OWNER/MANAGER not in gen file but stuck in prob
                                            → remove from prob, re-opt
  C) assoc_attr_below_threshold (436)     – ASSOC attrs p=0.725 just below sigmoid(1.0)=0.731
                                            → boost ASSOC FK attrs to 0.74, re-opt
"""
import os, json, sys, re

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

from difflib import SequenceMatcher
from src.finding_the_best_er_model import JointERILPSolverFinal_1
from src.pre_processing import preprocess_text
from src.relationship_processing import compute_relationship_probabilities_2

MAIN_PRO = os.path.join(project_root, "output/probability/multi-llms/pro_fewshot_llama_0.5_1.0")
MAIN_GEN = os.path.join(project_root, "output/generation/multi-llms/few-shot-llama")
INPUT_TXT = os.path.join(project_root, "dataset/Datasets/Full-Dataset/input")
ADD_OPT   = os.path.join(project_root, "output/add/optimization")
os.makedirs(ADD_OPT, exist_ok=True)

LAMBDA_E, LAMBDA_R, LAMBDA_A = 1.2, -0.5, 1.0
MIN_ENTITIES = 3
BIAS, WEIGHT = 0.5, 1.0

# Group A: no relationships in prob file
NO_REL_PROB = [379, 388, 392, 406, 410]

# Group B: entity present in prob but absent from gen (spurious)
SPURIOUS = {
    347: 'OWNER',
    492: 'MANAGER',
    392: 'ADMINISTRATOR',  # not in gen entities, no attrs
}

# Group C: ASSOC entity attrs just below threshold
ASSOC_BOOST = [436]


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_rel_prob(rel_probs_data, e1, e2):
    if not e1 or not e2:
        return 0.0
    e1u, e2u = e1.strip().upper(), e2.strip().upper()
    if isinstance(rel_probs_data, list):
        for item in rel_probs_data:
            if not isinstance(item, dict):
                continue
            i1 = str(item.get("entity_1", item.get("e1", ""))).strip().upper()
            i2 = str(item.get("entity_2", item.get("e2", ""))).strip().upper()
            if (i1 == e1u and i2 == e2u) or (i1 == e2u and i2 == e1u):
                return float(item.get("probability", item.get("p", 0.0)))
    return 0.0


def patch_add_rel_probs(ex_num):
    """Compute relationship probabilities from generation file, patch prob file."""
    pro_path = os.path.join(MAIN_PRO, f"{ex_num}.json")
    gen_path = os.path.join(MAIN_GEN, f"{ex_num}.json")
    txt_path = os.path.join(INPUT_TXT, f"{ex_num}.txt")

    pro = load_json(pro_path)
    gen = load_json(gen_path)

    raw_text = open(txt_path, encoding='utf-8').read().strip()
    processed_text = preprocess_text(raw_text)

    gen_rels = gen.get("relationship", [])
    if not gen_rels:
        print(f"  {ex_num}: no rels in gen file, skipping")
        return False

    rel_probs = compute_relationship_probabilities_2(
        json.dumps(gen_rels), processed_text, bias=BIAS, weight=WEIGHT)

    pro_entity = pro.get("entity", {})
    valid_ents  = set(pro_entity.keys())

    def norm(s):
        return re.sub(r'[^a-z0-9]', '', s.lower())

    def fuzzy_match(name, candidates):
        """Map a gen entity name to the best-matching prob entity name."""
        if name in candidates:
            return name
        n = norm(name)
        # Exact after normalization
        for c in candidates:
            if norm(c) == n:
                return c
        # Substring (e.g. SALARY_PAYMENT → SALARY)
        for c in candidates:
            cn = norm(c)
            if cn in n or n in cn:
                return c
        # Sequence similarity ≥ 0.82 (e.g. SALES_ORDER → SALE_ORDER, edit dist=1)
        best, best_r = None, 0.0
        for c in candidates:
            r = SequenceMatcher(None, n, norm(c)).ratio()
            if r > best_r:
                best_r, best = r, c
        if best_r >= 0.82:
            return best
        return None

    new_rels = []
    for r in gen_rels:
        e1_raw = str(r.get("entity_1", "")).strip().upper()
        e2_raw = str(r.get("entity_2", "")).strip().upper()
        e1 = fuzzy_match(e1_raw, valid_ents) or e1_raw
        e2 = fuzzy_match(e2_raw, valid_ents) or e2_raw
        if e1 not in valid_ents or e2 not in valid_ents:
            continue
        p = get_rel_prob(rel_probs, e1, e2)
        if p < 0.3:  # too low / missing → use entity average as fallback
            p = float(pro_entity.get(e1, 0.5)) * 0.5 + float(pro_entity.get(e2, 0.5)) * 0.5
        card = str(r.get("cardinality", "1:N")).upper()

        if card in ["N:M", "M:N"]:
            assoc_info = r.get("associative_entity")
            if assoc_info and isinstance(assoc_info, dict) and assoc_info.get("name"):
                assoc_name = str(assoc_info.get("name")).strip().upper()
                assoc_attrs_gen = assoc_info.get("attributes", [])
            else:
                assoc_name = f"ASSOC_{e1}_{e2}".upper()
                assoc_attrs_gen = []

            # Create or update ASSOC entity in prob file
            if assoc_name not in valid_ents or float(pro_entity.get(assoc_name, 0)) < p:
                pro_entity[assoc_name] = p
                valid_ents.add(assoc_name)
                pk  = f"{assoc_name.lower()}_id"
                fk1 = f"{e1.lower()}_id"
                fk2 = f"{e2.lower()}_id"
                assoc_attr_dict = {pk: p, fk1: p, fk2: p}
                for a in assoc_attrs_gen:
                    assoc_attr_dict[str(a).lower()] = p
                pro.setdefault("attribute", {})[assoc_name] = assoc_attr_dict
                print(f"    {ex_num}: upsert ASSOC entity '{assoc_name}' p={p:.3f}")

            for e_main in [e1, e2]:
                new_rels.append({"entity_1": e_main, "entity_2": assoc_name,
                                 "cardinality": "1:N", "associative_entity": None,
                                 "probability": p})
        else:
            new_rels.append({"entity_1": e1, "entity_2": e2,
                             "cardinality": card, "associative_entity": None,
                             "probability": p})

    if not new_rels:
        print(f"  {ex_num}: no valid rels after filtering")
        return False

    pro["entity"] = pro_entity
    pro["relationship"] = new_rels
    save_json(pro_path, pro)
    print(f"  {ex_num}: patched {len(new_rels)} relationships, entities={len(pro_entity)}")
    return True


def remove_spurious_entity(ex_num, entity_name):
    """Remove a spurious entity (not in gen file) from the prob file."""
    pro_path = os.path.join(MAIN_PRO, f"{ex_num}.json")
    pro = load_json(pro_path)

    ename = entity_name.strip().upper()
    changed = False

    if ename in pro.get("entity", {}):
        del pro["entity"][ename]
        changed = True

    if ename in pro.get("attribute", {}):
        del pro["attribute"][ename]
        changed = True

    if changed:
        save_json(pro_path, pro)
        print(f"  {ex_num}: removed spurious entity '{ename}'")
    return changed


def boost_assoc_attrs(ex_num, target_prob=0.74):
    """Boost ASSOC entity FK attribute probabilities above the ILP threshold."""
    pro_path = os.path.join(MAIN_PRO, f"{ex_num}.json")
    pro = load_json(pro_path)

    changed = False
    for ename in list(pro.get("attribute", {}).keys()):
        if not ename.upper().startswith("ASSOC_"):
            continue
        attrs = pro["attribute"][ename]
        if isinstance(attrs, dict):
            for ak in attrs:
                if float(attrs[ak]) < target_prob:
                    attrs[ak] = target_prob
                    changed = True
        pro["attribute"][ename] = attrs

    if changed:
        save_json(pro_path, pro)
        print(f"  {ex_num}: boosted ASSOC attrs to {target_prob}")
    return changed


def run_optimization(ex_num):
    pro_path = os.path.join(MAIN_PRO, f"{ex_num}.json")
    out_path = os.path.join(ADD_OPT,  f"{ex_num}.json")

    if not os.path.exists(pro_path):
        print(f"  Ex {ex_num}: NO PROB FILE")
        return False

    try:
        data = load_json(pro_path)
        solver = JointERILPSolverFinal_1(
            data.get("entity", {}),
            data.get("relationship", []),
            data.get("attribute", {}))
        score, sel_e, sel_r, sel_a, runtime = solver.solve(
            lambda_E=LAMBDA_E, lambda_R=LAMBDA_R, lambda_A=LAMBDA_A,
            min_entities=MIN_ENTITIES, no_isolated=True)

        formatted_rels = [
            {"entity_1": r.get("entity_1"), "entity_2": r.get("entity_2"),
             "cardinality": r.get("cardinality", "1:N"),
             "associative_entity": r.get("associative_entity")}
            for r in sel_r
        ]

        result = {"entity": sel_e, "attribut": sel_a, "relationship": formatted_rels}
        save_json(out_path, result)

        n_a = sum(len(v) for v in sel_a.values())
        connected = set()
        for r in formatted_rels:
            connected.add(r["entity_1"]); connected.add(r["entity_2"])
        iso    = [e for e in sel_e if e not in connected]
        noattr = [e for e in sel_e if not sel_a.get(e)]
        iso_str    = f" ISO={iso}"    if iso    else ""
        noattr_str = f" NOATTR={noattr}" if noattr else ""
        print(f"  Ex {ex_num}: E={len(sel_e)} A={n_a} R={len(formatted_rels)}{iso_str}{noattr_str}")
        return True
    except Exception as e:
        print(f"  Ex {ex_num}: ERROR {e}")
        return False


# ── Group A: no_rel_in_prob ────────────────────────────────────────────────────
print("=" * 65)
print("Group A: Patch relationship probabilities (379,388,392,406,410)")
print("=" * 65)
for ex in NO_REL_PROB:
    patch_add_rel_probs(ex)

# ── Group B: spurious entities ────────────────────────────────────────────────
print()
print("=" * 65)
print("Group B: Remove spurious entities (347→OWNER, 492→MANAGER)")
print("=" * 65)
for ex, ent in SPURIOUS.items():
    remove_spurious_entity(ex, ent)

# ── Group C: ASSOC attr boost ─────────────────────────────────────────────────
print()
print("=" * 65)
print("Group C: Boost ASSOC attrs below threshold (436)")
print("=" * 65)
for ex in ASSOC_BOOST:
    boost_assoc_attrs(ex)

# ── Re-run optimization for all ───────────────────────────────────────────────
all_fix = sorted(set(NO_REL_PROB + list(SPURIOUS.keys()) + ASSOC_BOOST))
print()
print("=" * 65)
print(f"Re-running optimization (no_isolated=True) for {len(all_fix)} exercises")
print("=" * 65)
ok, fail = 0, []
for ex in all_fix:
    if run_optimization(ex):
        ok += 1
    else:
        fail.append(ex)

print()
print("=" * 65)
print(f"Done: {ok}/{len(all_fix)}")
if fail:
    print(f"Failed: {fail}")

print()
print("── Verification ──")
for ex in all_fix:
    p = os.path.join(ADD_OPT, f"{ex}.json")
    if not os.path.exists(p):
        print(f"  {ex}: NOT GENERATED"); continue
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
