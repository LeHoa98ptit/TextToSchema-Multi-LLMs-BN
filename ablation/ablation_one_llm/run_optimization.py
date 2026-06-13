"""
Ablation — Hard-Constraint ILP Optimization for One-LLM Few-Shot LLaMA
=======================================================================
Reads probability files from ablation/ablation_one_llm/probability/
and applies HardConstraintERILP (no Wikidata, isolated + no-attr as hard constraints).

Output:
    ablation/ablation_one_llm/optimization/{id}.json

Usage:
    python ablation/ablation_one_llm/run_optimization.py
    python ablation/ablation_one_llm/run_optimization.py --workers 16
"""

import os, sys, json, time, argparse
from concurrent.futures import ThreadPoolExecutor

_SELF_DIR    = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_SELF_DIR, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ablation.src.select_best_ER_schema_ablation_hard import HardConstraintERILP

PROB_FOLDER = os.path.join(_SELF_DIR, 'probability')
OPT_FOLDER  = os.path.join(_SELF_DIR, 'optimization')
LAMBDA_FILE = os.path.join(project_root, 'optimization2', 'best_lambdas_with_isolated.json')
os.makedirs(OPT_FOLDER, exist_ok=True)

_DEFAULT_LAMBDAS = {"lambda_E": 0.8006, "lambda_A": 0.5875,
                    "lambda_R": 0.8489, "lambda_NM": 0.7217}

def _load_lambdas():
    try:
        lam = json.load(open(LAMBDA_FILE))
        lam = lam.get("lambdas", lam)
        return {k: lam[k] for k in ("lambda_E", "lambda_A", "lambda_R", "lambda_NM") if k in lam}
    except Exception:
        return _DEFAULT_LAMBDAS


def process_file(args):
    filename, prob_folder, opt_folder, lambdas = args
    out_path = os.path.join(opt_folder, filename)
    if os.path.exists(out_path):
        return True
    try:
        data = json.load(open(os.path.join(prob_folder, filename), encoding='utf-8'))
        entity_probs    = data.get('entity',       {})
        attribute_probs = data.get('attribute',    {})
        relation_rows   = data.get('relationship', [])

        if not entity_probs:
            json.dump({"entity": [], "attribute": {}, "relationship": []},
                      open(out_path, 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=2)
            return True

        ilp = HardConstraintERILP(entity_probs, relation_rows, attribute_probs)
        score, sel_ents, sel_rels, sel_attrs, runtime = ilp.solve(
            lambda_E=lambdas['lambda_E'], lambda_A=lambdas['lambda_A'],
            lambda_R=lambdas['lambda_R'], lambda_NM=lambdas['lambda_NM'],
            min_entities=3,
        )

        result = {
            "entity":       sel_ents,
            "attribute":    sel_attrs,
            "relationship": [
                {"entity_1": r["entity_1"], "entity_2": r["entity_2"],
                 "cardinality": r.get("cardinality", "1:N"),
                 "associative_entity": r.get("associative_entity")}
                for r in sel_rels
            ],
            "ilp_score":   round(score, 4) if score is not None else None,
            "ilp_runtime": round(runtime, 3),
        }
        json.dump(result, open(out_path, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f'  [ERR] {filename}: {e}')
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    lambdas = _load_lambdas()
    json_files = sorted(f for f in os.listdir(PROB_FOLDER)
                        if f.endswith('.json') and not f.startswith('.'))

    print(f'Input  : {PROB_FOLDER}')
    print(f'Output : {OPT_FOLDER}')
    print(f'Files  : {len(json_files)}  |  workers={args.workers}')
    print(f'Lambdas: {json.dumps(lambdas, indent=2)}\n')

    task_args = [(fn, PROB_FOLDER, OPT_FOLDER, lambdas) for fn in json_files]
    ok = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(process_file, task_args):
            if result: ok   += 1
            else:      fail += 1

    print(f'\nDone: {ok} OK, {fail} failed — {time.time()-t0:.1f}s')
    print(f'Output → {OPT_FOLDER}')


if __name__ == '__main__':
    main()
