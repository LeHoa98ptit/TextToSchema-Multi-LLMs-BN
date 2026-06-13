"""
Fix isolated entities in add/multi folders.
Root cause: prob files with 0 relationships → ILP infeasible with no_isolated=True.
Strategy: patch relationship probs IN-MEMORY from gen files, re-optimize.
Does NOT modify original prob files.
Output: overwrites affected exercises in output/add/multi/{folder}/
"""
import os, json, sys, re
from difflib import SequenceMatcher

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
from src.relationship_processing import compute_relationship_probabilities_2

PRO_BASE   = os.path.join(project_root, "output/probability/multi-llms")
GEN_BASE   = os.path.join(project_root, "output/generation/multi-llms")
ADD_MULTI  = os.path.join(project_root, "output/add/multi")
INPUT_TXT  = os.path.join(project_root, "dataset/Datasets/Full-Dataset/input")
MIN_ENTITIES = 3
BIAS, WEIGHT = 0.5, 1.0


# ── Config: opt_folder → (prob_folder, gen_folder, lambdas) ──────────────────
FOLDER_MAP = {
    "opt_fewshot_gpt_-0.5_1.0-(0.0-0.0-0.0)":  ("pro_fewshot_gpt_-0.5_1.0",  "few-shot-gpt",  (0.0, 0.0, 0.0)),
    "opt_fewshot_gpt_0.5_1.0-(1.2-1.0-1.0)":   ("pro_fewshot_gpt_0.5_1.0",   "few-shot-gpt",  (1.2, 1.0, 1.0)),
    "opt_fewshot_gpt_1.0_3.0-(1.5-2.0-2.0)":   ("pro_fewshot_gpt_1.0_3.0",   "few-shot-gpt",  (1.5, 2.0, 2.0)),
    "opt_fewshot_llama-0.5_1.0-(0.5-0.5-0.5)": ("pro_fewshot_llama_-0.5_1.0","few-shot-llama",(0.5, 0.5, 0.5)),
    "opt_fewshot_llama_1.0_3.0-(1.5-2.0-2.0)": ("pro_fewshot_llama_1.0_3.0", "few-shot-llama",(1.5, 2.0, 2.0)),
    "opt_zeroshot_gpt_-0.5_1.0-(0.0-0.0-0.0)": ("pro_zeroshot_gpt_-0.5_1.0", "zero-shot-gpt", (0.0, 0.0, 0.0)),
    "opt_zeroshot_gpt_0.5_1.0-(1.2-1.0-1.0)":  ("pro_zeroshot_gpt_0.5_1.0",  "zero-shot-gpt", (1.2, 1.0, 1.0)),
    "opt_zeroshot_gpt_1.0_3.0-(1.5-2.0-2.0)":  ("pro_zeroshot_gpt_1.0_3.0",  "zero-shot-gpt", (1.5, 2.0, 2.0)),
    "opt_zeroshot_llama-0.5_1.0-(0.5-0.5-0.5)":("pro_zeroshot_llama_-0.5_1.0","zero-shot-llama",(0.5, 0.5, 0.5)),
    "opt_zeroshot_llama_0.5_1.0-(1.2-1.0-1.0)":("pro_zeroshot_llama_0.5_1.0","zero-shot-llama",(1.2, 1.0, 1.0)),
    "opt_zeroshot_llama_1.0_3.0-(1.5-2.0-2.0)":("pro_zeroshot_llama_1.0_3.0","zero-shot-llama",(1.5, 2.0, 2.0)),
}


def norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

def fuzzy_match(name, candidates):
    if name in candidates: return name
    n = norm(name)
    for c in candidates:
        if norm(c) == n: return c
    for c in candidates:
        cn = norm(c)
        if cn in n or n in cn: return c
    best, best_r = None, 0.0
    for c in candidates:
        r = SequenceMatcher(None, n, norm(c)).ratio()
        if r > best_r: best_r, best = r, c
    if best_r >= 0.82: return best
    return None

def get_rel_prob(rel_probs_data, e1, e2):
    if not e1 or not e2: return 0.0
    e1u, e2u = e1.strip().upper(), e2.strip().upper()
    if isinstance(rel_probs_data, list):
        for item in rel_probs_data:
            if not isinstance(item, dict): continue
            i1 = str(item.get("entity_1", item.get("e1", ""))).strip().upper()
            i2 = str(item.get("entity_2", item.get("e2", ""))).strip().upper()
            if (i1 == e1u and i2 == e2u) or (i1 == e2u and i2 == e1u):
                return float(item.get("probability", item.get("p", 0.0)))
    return 0.0


def patch_rels_inMemory(pro_data, gen_data, ex_num):
    """Compute rel probs from gen file, return patched pro_data copy (no disk write)."""
    import copy
    pro = copy.deepcopy(pro_data)

    txt_path = os.path.join(INPUT_TXT, f"{ex_num}.txt")
    if not os.path.exists(txt_path):
        return pro, 0

    raw_text = open(txt_path, encoding='utf-8').read().strip()
    processed_text = preprocess_text(raw_text)

    gen_rels = gen_data.get("relationship", [])
    if not gen_rels:
        return pro, 0

    rel_probs = compute_relationship_probabilities_2(
        json.dumps(gen_rels), processed_text, bias=BIAS, weight=WEIGHT)

    pro_entity = pro.get("entity", {})
    valid_ents = set(pro_entity.keys())

    new_rels = []
    for r in gen_rels:
        e1_raw = str(r.get("entity_1", "")).strip().upper()
        e2_raw = str(r.get("entity_2", "")).strip().upper()
        # Skip self-referential
        if e1_raw == e2_raw: continue
        e1 = fuzzy_match(e1_raw, valid_ents) or e1_raw
        e2 = fuzzy_match(e2_raw, valid_ents) or e2_raw
        if e1 not in valid_ents or e2 not in valid_ents:
            continue
        p = get_rel_prob(rel_probs, e1, e2)
        if p < 0.3:
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

            for e_main in [e1, e2]:
                new_rels.append({"entity_1": e_main, "entity_2": assoc_name,
                                 "cardinality": "1:N", "associative_entity": None,
                                 "probability": p})
        else:
            new_rels.append({"entity_1": e1, "entity_2": e2,
                             "cardinality": card, "associative_entity": None,
                             "probability": p})

    pro["entity"] = pro_entity
    if new_rels:
        pro["relationship"] = new_rels
    return pro, len(new_rels)


def optimize(pro_data, lE, lR, lA):
    solver = JointERILPSolverFinal_1(
        pro_data.get("entity", {}),
        pro_data.get("relationship", []),
        pro_data.get("attribute", {}))
    _, sel_e, sel_r, sel_a, _ = solver.solve(
        lambda_E=lE, lambda_R=lR, lambda_A=lA,
        min_entities=MIN_ENTITIES, no_isolated=True)
    fmt_rels = [{"entity_1": r.get("entity_1"), "entity_2": r.get("entity_2"),
                 "cardinality": r.get("cardinality", "1:N"),
                 "associative_entity": r.get("associative_entity")}
                for r in sel_r]
    return {"entity": sel_e, "attribut": sel_a, "relationship": fmt_rels}


# ── Main ──────────────────────────────────────────────────────────────────────
total_fixed = total_skipped = total_failed = 0

for opt_name, (pro_name, gen_name, lambdas) in FOLDER_MAP.items():
    out_dir = os.path.join(ADD_MULTI, opt_name)
    pro_dir = os.path.join(PRO_BASE, pro_name)
    gen_dir = os.path.join(GEN_BASE, gen_name)

    if not os.path.exists(out_dir):
        continue

    # Find iso exercises in this folder
    iso_exs = []
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith('.json') or f.startswith('._'): continue
        ex = int(f.replace('.json',''))
        d = json.load(open(os.path.join(out_dir, f)))
        ents = d.get('entity', [])
        rels = d.get('relationship', [])
        conn = {r.get('entity_1') for r in rels} | {r.get('entity_2') for r in rels}
        iso = [e for e in ents if e not in conn]
        if iso:
            iso_exs.append((ex, iso))

    if not iso_exs:
        continue

    lE, lR, lA = lambdas
    print(f"\n{'='*65}")
    print(f"{opt_name}  λE={lE} λR={lR} λA={lA}")
    print(f"  iso exercises: {[ex for ex,_ in iso_exs]}")
    print(f"{'='*65}")

    for ex, iso_ents in iso_exs:
        pro_path = os.path.join(pro_dir, f"{ex}.json")
        gen_path = os.path.join(gen_dir, f"{ex}.json")
        out_path = os.path.join(out_dir, f"{ex}.json")

        if not os.path.exists(pro_path):
            print(f"  Ex {ex}: NO PROB FILE — skip")
            total_skipped += 1
            continue

        pro_data = json.load(open(pro_path, encoding='utf-8'))
        pro_rels = pro_data.get("relationship", [])
        pro_entity_keys = set(pro_data.get("entity", {}).keys())
        # Filter self-referential AND rels where either entity not in entity dict
        valid_rels = [r for r in pro_rels
                      if r.get("entity_1") != r.get("entity_2")
                      and r.get("entity_1") in pro_entity_keys
                      and r.get("entity_2") in pro_entity_keys]

        entities_with_rels = set()
        for r in valid_rels:
            entities_with_rels.add(r.get("entity_1"))
            entities_with_rels.add(r.get("entity_2"))

        need_patch = (len(valid_rels) == 0) or \
                     (len(pro_entity_keys) < MIN_ENTITIES) or \
                     (len(entities_with_rels) < MIN_ENTITIES)

        if need_patch:
            if not os.path.exists(gen_path):
                print(f"  Ex {ex}: NO GEN FILE — skip (iso={iso_ents})")
                total_skipped += 1
                continue
            gen_data = json.load(open(gen_path, encoding='utf-8'))
            patched, n_new = patch_rels_inMemory(pro_data, gen_data, ex)
            if n_new == 0:
                print(f"  Ex {ex}: patch yielded 0 rels — skip (iso={iso_ents})")
                total_skipped += 1
                continue
            data_to_opt = patched
            print(f"  Ex {ex}: patched {n_new} rels from gen", end="")
        else:
            data_to_opt = pro_data
            print(f"  Ex {ex}: has {len(valid_rels)} valid rels but still iso", end="")

        try:
            result = optimize(data_to_opt, lE, lR, lA)
            ents   = result['entity']
            rels   = result['relationship']
            conn   = {r['entity_1'] for r in rels} | {r['entity_2'] for r in rels}
            iso_after = [e for e in ents if e not in conn]
            json.dump(result, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
            status = "✓" if not iso_after else f"still iso={iso_after}"
            print(f"  → E={len(ents)} R={len(rels)} {status}")
            total_fixed += 1
        except Exception as e:
            print(f"  → ERROR: {e}")
            total_failed += 1


# ── Final summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"Fixed: {total_fixed}  Skipped: {total_skipped}  Failed: {total_failed}")
print()

# Quality check all folders
print("Final quality per folder:")
for opt_name in sorted(FOLDER_MAP):
    out_dir = os.path.join(ADD_MULTI, opt_name)
    if not os.path.exists(out_dir): continue
    files = [f for f in os.listdir(out_dir) if f.endswith('.json') and not f.startswith('._')]
    n_iso = n_noattr = 0
    for fname in files:
        d = json.load(open(os.path.join(out_dir, fname)))
        ents  = d.get('entity', [])
        attrs = d.get('attribut', d.get('attribute', {}))
        rels  = d.get('relationship', [])
        conn  = {r.get('entity_1') for r in rels} | {r.get('entity_2') for r in rels}
        if any(e not in conn for e in ents): n_iso += 1
        if any(not (attrs.get(e) if isinstance(attrs,dict) else []) for e in ents): n_noattr += 1
    print(f"  {opt_name:<55}  iso={n_iso:3d}  noattr={n_noattr:3d}  total={len(files)}")
