"""
Ablation Study — Hard-Constraint ILP Optimization
==================================================
Uses HardConstraintERILP: isolated-entity and entity-without-attribute
are enforced as hard constraints (not soft lambda penalties).

Lambdas used: lambda_E, lambda_A, lambda_R, lambda_NM from
optimization2/best_lambdas_with_isolated.json (same base as complex ILP,
minus lambda_noattr and lambda_isolated which are now hard constraints).

Input:
    ablation/probability_estimation_ablation/output_0.5_1.0/{variant}/{id}.json

Output:
    ablation/optimization_ablation_hard/output/{variant}/{id}.json

Usage:
    python ablation/optimization_ablation_hard/run_optimization_ablation_hard.py
    python ablation/optimization_ablation_hard/run_optimization_ablation_hard.py --variant few-shot-llama
    python ablation/optimization_ablation_hard/run_optimization_ablation_hard.py --workers 16
"""

import os
import sys
import json
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor

_SELF_DIR    = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_SELF_DIR, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ablation.src.select_best_ER_schema_ablation_hard import HardConstraintERILP

PROB_ROOT   = os.path.join(project_root, 'ablation', 'probability_estimation_ablation',
                           'output_0.5_1.0')
OUT_ROOT    = os.path.join(_SELF_DIR, 'output')
LOG_ROOT    = os.path.join(_SELF_DIR, 'log')
LAMBDA_FILE = os.path.join(project_root, 'optimization2', 'best_lambdas_with_isolated.json')

_DEFAULT_LAMBDAS = {
    "lambda_E":  0.8006,
    "lambda_A":  0.5875,
    "lambda_R":  0.8489,
    "lambda_NM": 0.7217,
}

def _load_lambdas() -> dict:
    try:
        with open(LAMBDA_FILE, 'r') as f:
            data = json.load(f)
        lam = data.get("lambdas", data)
        return {k: lam[k] for k in ("lambda_E", "lambda_A", "lambda_R", "lambda_NM") if k in lam}
    except Exception:
        return _DEFAULT_LAMBDAS


VARIANTS = [
    (os.path.join('multi-llms', 'few-shot-gpt'),            os.path.join('multi-llms', 'few-shot-gpt')),
    (os.path.join('multi-llms', 'few-shot-llama'),          os.path.join('multi-llms', 'few-shot-llama')),
    (os.path.join('multi-llms', 'zero-shot-gpt'),           os.path.join('multi-llms', 'zero-shot-gpt')),
    (os.path.join('multi-llms', 'zero-shot-llama'),         os.path.join('multi-llms', 'zero-shot-llama')),
    (os.path.join('one-llm', 'one_llm_few_shot_gpt'),       os.path.join('one-llm', 'one_llm_few_shot_gpt')),
    (os.path.join('one-llm', 'one_llm_few_shot_llama'),     os.path.join('one-llm', 'one_llm_few_shot_llama')),
    (os.path.join('one-llm', 'one_llm_zero_shot_gpt'),      os.path.join('one-llm', 'one_llm_zero_shot_gpt')),
    (os.path.join('one-llm', 'one_llm_zero_shot_llama'),    os.path.join('one-llm', 'one_llm_zero_shot_llama')),
    (os.path.join('ToT', 'gpt'),                            os.path.join('ToT', 'gpt')),
    (os.path.join('ToT', 'llama'),                          os.path.join('ToT', 'llama')),
]


def _setup_logger(variant_key: str):
    os.makedirs(LOG_ROOT, exist_ok=True)
    log_name = variant_key.replace(os.sep, '-').replace('/', '-')
    log_path = os.path.join(LOG_ROOT, f'{log_name}.txt')
    logger = logging.getLogger(f'opt_hard.{log_name}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(message)s')
    fh  = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh  = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger, log_path


def _process_file(args: tuple) -> bool:
    filename, in_folder, out_folder, lambdas = args
    out_path = os.path.join(out_folder, filename)
    if os.path.exists(out_path):
        return True
    try:
        with open(os.path.join(in_folder, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)

        entity_probs    = data.get('entity',       {})
        attribute_probs = data.get('attribute',    {})
        relation_rows   = data.get('relationship', [])

        if not entity_probs:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({"entity": [], "attribute": {}, "relationship": []},
                          f, ensure_ascii=False, indent=2)
            return True

        ilp = HardConstraintERILP(entity_probs, relation_rows, attribute_probs)
        score, sel_entities, sel_relations, sel_attributes, runtime = ilp.solve(
            lambda_E     = lambdas['lambda_E'],
            lambda_A     = lambdas['lambda_A'],
            lambda_R     = lambdas['lambda_R'],
            lambda_NM    = lambdas['lambda_NM'],
            min_entities = 3,
        )

        result = {
            "entity":       sel_entities,
            "attribute":    sel_attributes,
            "relationship": [
                {
                    "entity_1":           r["entity_1"],
                    "entity_2":           r["entity_2"],
                    "cardinality":        r.get("cardinality", "1:N"),
                    "associative_entity": r.get("associative_entity"),
                }
                for r in sel_relations
            ],
            "ilp_score":   round(score, 4) if score is not None else None,
            "ilp_runtime": round(runtime, 3),
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return True

    except Exception as e:
        print(f'  [ERR] {filename}: {e}')
        return False


def _run_variant(prob_sub: str, out_sub: str, lambdas: dict, max_workers: int) -> tuple:
    in_folder  = os.path.join(PROB_ROOT, prob_sub)
    out_folder = os.path.join(OUT_ROOT,  out_sub)
    logger, log_path = _setup_logger(out_sub)

    if not os.path.isdir(in_folder):
        logger.info(f'  [WARN] Probability folder not found: {in_folder}')
        return 0, 0

    os.makedirs(out_folder, exist_ok=True)
    json_files = sorted(f for f in os.listdir(in_folder)
                        if f.endswith('.json') and not f.startswith('._'))

    logger.info(f'\n{"="*65}')
    logger.info(f'Variant  : {prob_sub}')
    logger.info(f'Files    : {len(json_files)}')
    logger.info(f'Input    : {in_folder}')
    logger.info(f'Output   : {out_folder}')
    logger.info(f'Lambdas  : {json.dumps(lambdas, indent=12)}')
    logger.info(f'{"="*65}')

    t0   = time.time()
    ok   = fail = 0
    args = [(fn, in_folder, out_folder, lambdas) for fn in json_files]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for success in pool.map(_process_file, args):
            if success: ok   += 1
            else:       fail += 1

    elapsed = time.time() - t0
    logger.info(f'\n  Done: {ok} OK, {fail} failed — {elapsed:.1f}s')
    logger.info(f'  Log saved → {log_path}')
    return ok, fail


def main():
    tuned = _load_lambdas()

    parser = argparse.ArgumentParser(
        description='Hard-constraint ILP (isolated + no-attr as hard constraints).'
    )
    parser.add_argument('--variant', default=None)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--lambda_E',  type=float, default=tuned['lambda_E'])
    parser.add_argument('--lambda_A',  type=float, default=tuned['lambda_A'])
    parser.add_argument('--lambda_R',  type=float, default=tuned['lambda_R'])
    parser.add_argument('--lambda_NM', type=float, default=tuned['lambda_NM'])
    args = parser.parse_args()

    lambdas = {
        'lambda_E':  args.lambda_E,
        'lambda_A':  args.lambda_A,
        'lambda_R':  args.lambda_R,
        'lambda_NM': args.lambda_NM,
    }

    variants = VARIANTS
    if args.variant:
        variants = [(p, o) for p, o in VARIANTS if args.variant in p]
        if not variants:
            print(f'No variant matches "{args.variant}". Available:')
            for p, _ in VARIANTS: print(f'  {p}')
            sys.exit(1)

    print(f'\nHard-Constraint ILP (isolated + no-attr enforced as hard constraints)')
    print(f'Lambdas : {json.dumps(lambdas, indent=2)}')
    print(f'Workers : {args.workers}\n')

    total_ok = total_fail = 0
    for prob_sub, out_sub in variants:
        ok, fail = _run_variant(prob_sub, out_sub, lambdas, args.workers)
        total_ok += ok; total_fail += fail

    print(f'\n{"="*65}')
    print(f'All variants done.  OK={total_ok}  Failed={total_fail}')
    print(f'Output root: {OUT_ROOT}')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()
