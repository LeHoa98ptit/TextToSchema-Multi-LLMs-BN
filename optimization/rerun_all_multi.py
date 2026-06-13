"""
Re-run ILP optimization with no_isolated=True for all multi-llms folders
(except opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0) already done).
Output: output/add/multi/{folder_name}/{ex}.json
"""
import os, json, sys, re, time
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

OPT_BASE = os.path.join(project_root, "output/optimization/multi-llms")
PRO_BASE = os.path.join(project_root, "output/probability/multi-llms")
ADD_MULTI = os.path.join(project_root, "output/add/multi")
MIN_ENTITIES = 3

# Already handled separately
SKIP = {"opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0)"}


def parse_lambdas(lambdas_str):
    """Parse '1.2--0.5-1.0' → (1.2, -0.5, 1.0) using lookbehind on digit."""
    parts = re.split(r'(?<=\d)-', lambdas_str)
    return tuple(float(p) for p in parts)


def find_prob_folder(opt_name):
    """Map opt folder name → prob folder path."""
    m = re.match(r'opt_(.+?)-\((.+)\)$', opt_name)
    if not m:
        return None, None
    config = m.group(1)       # e.g. fewshot_llama_0.5_1.0
    lambdas_str = m.group(2)  # e.g. 1.2--0.5-1.0

    lambdas = parse_lambdas(lambdas_str)

    # Try direct match first, then with underscore fix (llama-0.5 → llama_-0.5)
    for candidate in [f"pro_{config}", f"pro_{config.replace('-', '_-', 1)}"]:
        p = os.path.join(PRO_BASE, candidate)
        if os.path.exists(p):
            return p, lambdas

    # Broader search: list and match
    for d in os.listdir(PRO_BASE):
        norm_d = d.replace('pro_', '').replace('-', '_').replace('__', '_')
        norm_c = config.replace('-', '_').replace('__', '_')
        if norm_d == norm_c:
            return os.path.join(PRO_BASE, d), lambdas

    return None, lambdas


def process_one(ex, prob_folder, out_folder, lE, lR, lA):
    pro_path = os.path.join(prob_folder, f"{ex}.json")
    out_path = os.path.join(out_folder, f"{ex}.json")
    if not os.path.exists(pro_path):
        return ex, None, "no_prob"
    try:
        data = json.load(open(pro_path, encoding='utf-8'))
        solver = JointERILPSolverFinal_1(
            data.get("entity", {}),
            data.get("relationship", []),
            data.get("attribute", {}))
        _, sel_e, sel_r, sel_a, _ = solver.solve(
            lambda_E=lE, lambda_R=lR, lambda_A=lA,
            min_entities=MIN_ENTITIES, no_isolated=True)

        fmt_rels = [{"entity_1": r.get("entity_1"), "entity_2": r.get("entity_2"),
                     "cardinality": r.get("cardinality", "1:N"),
                     "associative_entity": r.get("associative_entity")}
                    for r in sel_r]
        result = {"entity": sel_e, "attribut": sel_a, "relationship": fmt_rels}
        json.dump(result, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

        connected = {r["entity_1"] for r in fmt_rels} | {r["entity_2"] for r in fmt_rels}
        iso    = [e for e in sel_e if e not in connected]
        noattr = [e for e in sel_e if not sel_a.get(e)]
        return ex, result, {"iso": iso, "noattr": noattr}
    except Exception as e:
        return ex, None, str(e)


def run_folder(opt_name):
    opt_folder  = os.path.join(OPT_BASE, opt_name)
    prob_folder, lambdas = find_prob_folder(opt_name)
    out_folder  = os.path.join(ADD_MULTI, opt_name)
    os.makedirs(out_folder, exist_ok=True)

    if not prob_folder:
        print(f"  [{opt_name}] SKIP — prob folder not found")
        return

    lE, lR, lA = lambdas
    print(f"\n{'='*65}")
    print(f"  {opt_name}")
    print(f"  λE={lE}  λR={lR}  λA={lA}")
    print(f"  prob: {os.path.basename(prob_folder)}")
    print(f"{'='*65}")

    # Collect exercise numbers from prob folder (union with opt folder)
    prob_exs = {int(f.replace('.json',''))
                for f in os.listdir(prob_folder)
                if f.endswith('.json') and not f.startswith('._')}
    opt_exs  = {int(f.replace('.json',''))
                for f in os.listdir(opt_folder)
                if f.endswith('.json') and not f.startswith('._')}
    exercises = sorted(prob_exs & opt_exs)  # only re-run what already had output

    t0 = time.time()
    iso_count = noattr_count = ok = fail = 0

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(process_one, ex, prob_folder, out_folder, lE, lR, lA): ex
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
    print(f"  Done {ok}/{len(exercises)} in {elapsed:.1f}s  "
          f"| remaining iso_entities={iso_count}  noattr_entities={noattr_count}")
    if fail:
        print(f"  Failed: {fail}")

    # Quick quality check
    files = [f for f in os.listdir(out_folder) if f.endswith('.json') and not f.startswith('._')]
    n_iso = n_noattr = n_norel = 0
    for fname in files:
        d = json.load(open(os.path.join(out_folder, fname)))
        ents  = d.get('entity', [])
        attrs = d.get('attribut', d.get('attribute', {}))
        rels  = d.get('relationship', [])
        connected = {r.get('entity_1') for r in rels} | {r.get('entity_2') for r in rels}
        if any(e not in connected for e in ents): n_iso += 1
        if any(not (attrs.get(e) if isinstance(attrs,dict) else []) for e in ents): n_noattr += 1
        if not rels: n_norel += 1
    print(f"  Quality: iso_exs={n_iso}  noattr_exs={n_noattr}  norel_exs={n_norel}  total={len(files)}")


# ── Main ──────────────────────────────────────────────────────────────────────
folders = sorted(
    d for d in os.listdir(OPT_BASE)
    if os.path.isdir(os.path.join(OPT_BASE, d)) and d not in SKIP
)

print(f"Processing {len(folders)} folders → {ADD_MULTI}")
for opt_name in folders:
    run_folder(opt_name)

print("\n" + "="*65)
print("ALL DONE")
