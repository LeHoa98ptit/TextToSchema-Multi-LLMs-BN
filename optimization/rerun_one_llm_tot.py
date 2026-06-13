"""
Re-run ILP with no_isolated=True for all one-llm and ToT folders.
Also fixes:
  - iso entities: patch rels in-memory from gen file
  - noattr entities: patch attrs in-memory from gen file
  - degraded E=1 R=0 (infeasible fallback): restore from original opt

Output:
  output/add/one-llm/{folder}/
  output/add/ToT/{folder}/
"""
import os, json, sys, re, copy, time
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor, as_completed

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

PRO_BASE   = os.path.join(project_root, "output/probability")
GEN_BASE   = os.path.join(project_root, "output/generation")
OPT_BASE   = os.path.join(project_root, "output/optimization")
ADD_BASE   = os.path.join(project_root, "output/add")
INPUT_TXT  = os.path.join(project_root, "dataset/Datasets/Full-Dataset/input")
MIN_ENTITIES = 3
BIAS, WEIGHT = 0.5, 1.0


# ── Folder config: opt_folder → (prob_folder, gen_folder, lambdas) ───────────
FOLDER_MAP = {
    # one-llm
    "one-llm/opt_one_fewshot_gpt_0.5_1.0-(1.2-1.0-1.0)": (
        "one-llm/pro_fewshot_gpt_0.5_1.0",
        "one-llm/one_llm_few_shot_gpt",
        (1.2, 1.0, 1.0)),
    "one-llm/opt_one_fewshot_llama_0.5_1.0-(1.2-1.0-1.0)": (
        "one-llm/pro_few_shot_llama_0.5_1.0",
        "one-llm/one_llm_few_shot_llama",
        (1.2, 1.0, 1.0)),
    "one-llm/opt_one_zeroshot_gpt_0.5_1.0-(1.2-1.0-1.0)": (
        "one-llm/pro_zeroshot_gpt_0.5_1.0",
        "one-llm/one_llm_zero_shot_gpt",
        (1.2, 1.0, 1.0)),
    "one-llm/opt_one_zeroshot_llama_0.5_1.0-(1.2-1.0-1.0)": (
        "one-llm/pro_zeroshot_llama_0.5_1.0",
        "one-llm/one_llm_zero_shot_llama",
        (1.2, 1.0, 1.0)),
    # ToT
    "ToT/opt_ToT_gpt_-0.5_1.0-(0.0-0.0-0.0)": (
        "ToT/pro_ToT_gpt_-0.5_1.0",
        "prompt_ToT_gpt",
        (0.0, 0.0, 0.0)),
    "ToT/opt_ToT_gpt_0.5_1.0-(1.2-1.0-1.0)": (
        "ToT/pro_ToT_gpt_0.5_1.0",
        "prompt_ToT_gpt",
        (1.2, 1.0, 1.0)),
    "ToT/opt_ToT_gpt_1.0_3.0-(1.5-2.0-2.0)": (
        "ToT/pro_ToT_gpt_1.0_3.0",
        "prompt_ToT_gpt",
        (1.5, 2.0, 2.0)),
    "ToT/opt_ToT_llama-0.5_1.0-(0.5-0.5-0.5)": (
        "ToT/pro_ToT_llama_-0.5_1.0",
        "prompt_ToT_llama",
        (0.5, 0.5, 0.5)),
    "ToT/opt_ToT_llama_0.5_1.0-(1.2-1.0-1.0)": (
        "ToT/pro_ToT_llama_0.5_1.0",
        "prompt_ToT_llama",
        (1.2, 1.0, 1.0)),
    "ToT/opt_ToT_llama_1.0_3.0-(1.5-2.0-2.0)": (
        "ToT/pro_ToT_llama_1.0_3.0",
        "prompt_ToT_llama",
        (1.5, 2.0, 2.0)),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def sigmoid(x):
    import math
    return 1.0 / (1.0 + math.exp(-x))

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


def patch_rels(pro_data, gen_data, ex_num):
    """Patch rel probs in-memory from gen file. Returns (patched_data, n_new_rels)."""
    pro = copy.deepcopy(pro_data)
    txt_path = os.path.join(INPUT_TXT, f"{ex_num}.txt")
    if not os.path.exists(txt_path): return pro, 0

    raw_text = open(txt_path, encoding='utf-8').read().strip()
    processed_text = preprocess_text(raw_text)

    gen_rels = gen_data.get("relationship", [])
    if not gen_rels: return pro, 0

    rel_probs = compute_relationship_probabilities_2(
        json.dumps(gen_rels), processed_text, bias=BIAS, weight=WEIGHT)

    pro_entity = pro.get("entity", {})
    valid_ents = set(pro_entity.keys())
    new_rels = []

    for r in gen_rels:
        e1_raw = str(r.get("entity_1", "")).strip().upper()
        e2_raw = str(r.get("entity_2", "")).strip().upper()
        if e1_raw == e2_raw: continue
        e1 = fuzzy_match(e1_raw, valid_ents) or e1_raw
        e2 = fuzzy_match(e2_raw, valid_ents) or e2_raw
        if e1 not in valid_ents or e2 not in valid_ents: continue
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
                ad  = {pk: p, fk1: p, fk2: p}
                for a in assoc_attrs_gen:
                    ad[str(a).lower()] = p
                pro.setdefault("attribute", {})[assoc_name] = ad
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


def patch_attrs(pro_data, gen_data, noattr_ents, lambda_A):
    """Add attributes from gen file for entities missing attrs. Returns patched copy."""
    pro = copy.deepcopy(pro_data)
    # target prob: slightly above sigmoid(lambda_A)
    import math
    attr_threshold = sigmoid(lambda_A)
    attr_prob_default = min(0.95, max(attr_threshold + 0.05, 0.75))

    gen_attrs = gen_data.get("attribute", gen_data.get("attribut", {}))
    pro_entity = pro.get("entity", {})
    pro_attrs = pro.get("attribute", pro.get("attribut", {}))

    patched = 0
    for ent in noattr_ents:
        # Find entity in gen (fuzzy match)
        gen_ent_match = fuzzy_match(ent, set(gen_attrs.keys()) if isinstance(gen_attrs, dict) else set())
        if gen_ent_match:
            gen_attr_list = gen_attrs[gen_ent_match]
            if isinstance(gen_attr_list, list) and gen_attr_list:
                p = max(float(pro_entity.get(ent, 0.5)), attr_prob_default)
                pro_attrs[ent] = {str(a).lower(): p for a in gen_attr_list}
                patched += 1
        elif ent not in pro_attrs or not pro_attrs[ent]:
            # Fallback: add generic PK attribute
            p = max(float(pro_entity.get(ent, 0.5)), attr_prob_default)
            pk = f"{ent.lower()}_id"
            pro_attrs[ent] = {pk: p}
            patched += 1

    pro["attribute"] = pro_attrs
    return pro, patched


def run_ilp(pro_data, lE, lR, lA):
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


def process_one(ex, pro_dir, gen_dir, orig_dir, out_dir, lE, lR, lA):
    pro_path  = os.path.join(pro_dir,  f"{ex}.json")
    gen_path  = os.path.join(gen_dir,  f"{ex}.json")
    orig_path = os.path.join(orig_dir, f"{ex}.json")
    out_path  = os.path.join(out_dir,  f"{ex}.json")

    if not os.path.exists(pro_path):
        return ex, None, "no_prob"

    try:
        pro_data = json.load(open(pro_path, encoding='utf-8'))
        gen_data = json.load(open(gen_path, encoding='utf-8')) if os.path.exists(gen_path) else {}

        # ── Step 1: initial ILP run ────────────────────────────────────────────
        data = pro_data
        result = run_ilp(data, lE, lR, lA)

        # ── Step 2: fix iso ────────────────────────────────────────────────────
        ents = result['entity']
        rels = result['relationship']
        conn = {r['entity_1'] for r in rels} | {r['entity_2'] for r in rels}
        iso  = [e for e in ents if e not in conn]

        if iso and gen_data:
            pro_entity_keys = set(pro_data.get("entity", {}).keys())
            valid_rels = [r for r in pro_data.get("relationship", [])
                          if r.get("entity_1") != r.get("entity_2")
                          and r.get("entity_1") in pro_entity_keys
                          and r.get("entity_2") in pro_entity_keys]
            entities_with_rels = set()
            for r in valid_rels:
                entities_with_rels.add(r.get("entity_1"))
                entities_with_rels.add(r.get("entity_2"))
            need_rel_patch = (len(valid_rels) == 0) or (len(entities_with_rels) < MIN_ENTITIES)

            if need_rel_patch:
                patched_data, n_new = patch_rels(pro_data, gen_data, ex)
                if n_new > 0:
                    data = patched_data
                    result = run_ilp(data, lE, lR, lA)
                    ents = result['entity']
                    rels = result['relationship']
                    conn = {r['entity_1'] for r in rels} | {r['entity_2'] for r in rels}
                    iso  = [e for e in ents if e not in conn]

        # ── Step 3: fix noattr ─────────────────────────────────────────────────
        attrs = result['attribut']
        noattr = [e for e in ents if not attrs.get(e)]

        if noattr and gen_data:
            patched_data, n_patched = patch_attrs(data, gen_data, noattr, lA)
            if n_patched > 0:
                data = patched_data
                result = run_ilp(data, lE, lR, lA)
                ents   = result['entity']
                rels   = result['relationship']
                conn   = {r['entity_1'] for r in rels} | {r['entity_2'] for r in rels}
                iso    = [e for e in ents if e not in conn]
                attrs  = result['attribut']
                noattr = [e for e in ents if not attrs.get(e)]

        # ── Step 4: restore degraded (E=1, R=0) from original ─────────────────
        if len(ents) <= 1 and not rels and os.path.exists(orig_path):
            orig = json.load(open(orig_path, encoding='utf-8'))
            orig_ents = orig.get('entity', [])
            if len(orig_ents) > 1:
                result = orig
                ents   = result.get('entity', [])
                rels   = result.get('relationship', [])
                conn   = {r.get('entity_1') for r in rels} | {r.get('entity_2') for r in rels}
                iso    = [e for e in ents if e not in conn]
                attrs  = result.get('attribut', result.get('attribute', {}))
                noattr = [e for e in ents if not (attrs.get(e) if isinstance(attrs, dict) else [])]

        json.dump(result, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
        return ex, result, {"iso": iso, "noattr": noattr}

    except Exception as e:
        return ex, None, str(e)


# ── Main ──────────────────────────────────────────────────────────────────────
for rel_opt_path, (rel_pro_path, rel_gen_path, lambdas) in FOLDER_MAP.items():
    section, opt_name = rel_opt_path.split("/", 1)

    opt_dir  = os.path.join(OPT_BASE, section, opt_name)
    pro_dir  = os.path.join(PRO_BASE, rel_pro_path)
    gen_dir  = os.path.join(GEN_BASE, rel_gen_path)
    out_dir  = os.path.join(ADD_BASE, rel_opt_path)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(pro_dir):
        print(f"SKIP {rel_opt_path} — prob folder not found")
        continue

    lE, lR, lA = lambdas

    # Collect exercises: intersection of prob and opt
    prob_exs = {int(f.replace('.json','')) for f in os.listdir(pro_dir)
                if f.endswith('.json') and not f.startswith('._')}
    opt_exs  = {int(f.replace('.json','')) for f in os.listdir(opt_dir)
                if f.endswith('.json') and not f.startswith('._')}
    exercises = sorted(prob_exs & opt_exs)

    print(f"\n{'='*65}")
    print(f"  {rel_opt_path}")
    print(f"  λE={lE}  λR={lR}  λA={lA}  exercises={len(exercises)}")
    print(f"{'='*65}")

    t0 = time.time()
    ok = fail = iso_count = noattr_count = 0

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(process_one, ex, pro_dir, gen_dir, opt_dir, out_dir, lE, lR, lA): ex
                   for ex in exercises}
        for fut in as_completed(futures):
            ex, result, info = fut.result()
            if result is None:
                fail += 1
                if info != "no_prob":
                    print(f"  ERROR {ex}: {info}")
            else:
                ok += 1
                if isinstance(info, dict):
                    iso_count    += len(info["iso"])
                    noattr_count += len(info["noattr"])

    elapsed = time.time() - t0
    print(f"  Done {ok}/{len(exercises)} in {elapsed:.1f}s  iso_entities={iso_count}  noattr_entities={noattr_count}")
    if fail: print(f"  Failed: {fail}")

    # Quality summary
    files = [f for f in os.listdir(out_dir) if f.endswith('.json') and not f.startswith('._')]
    n_iso = n_noattr = n_norel = 0
    for fname in files:
        d = json.load(open(os.path.join(out_dir, fname)))
        ents  = d.get('entity', [])
        attrs = d.get('attribut', d.get('attribute', {}))
        rels  = d.get('relationship', [])
        conn  = {r.get('entity_1') for r in rels} | {r.get('entity_2') for r in rels}
        if any(e not in conn for e in ents): n_iso += 1
        if any(not (attrs.get(e) if isinstance(attrs,dict) else []) for e in ents): n_noattr += 1
        if not rels: n_norel += 1
    print(f"  Quality: iso_exs={n_iso}  noattr_exs={n_noattr}  norel_exs={n_norel}  total={len(files)}")

print("\n" + "="*65)
print("ALL DONE")
