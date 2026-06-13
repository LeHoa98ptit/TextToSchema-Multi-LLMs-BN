"""
Ablation Study — ILP Optimization Pipeline
===========================================
Reads ablation probability files and runs the complexity-aware ILP
(JointERILPComplexity) to select the best ER schema for each exercise.

Input:
    ablation/probability_estimation_ablation/output_0.5_1.0/{variant}/{id}.json

Output:
    Optimization_Ablation/output/{variant}/{id}.json

ILP used:
    ablation/src/select_best_ER_schema_ablation.py :: JointERILPComplexity
    — includes both constraints:
        • lambda_noattr   : penalises entities with no selected attributes
        • lambda_isolated : penalises entities with no selected relationships

Lambda values:
    Loaded from optimization2/best_lambdas_with_isolated.json (tuned on training set).
    Override via CLI: --lambda_E, --lambda_A, --lambda_R,
                      --lambda_noattr, --lambda_NM, --lambda_isolated

Variants processed (10 total):
    multi-llms : few-shot-gpt, few-shot-llama, zero-shot-gpt, zero-shot-llama
    one-llm    : one_llm_few_shot_gpt, one_llm_few_shot_llama,
                 one_llm_zero_shot_gpt, one_llm_zero_shot_llama
    ToT        : prompt_ToT_gpt (→ ToT/gpt), prompt_ToT_llama (→ ToT/llama)

Logs:
    Optimization_Ablation/log/{variant}.txt   (one file per variant)

Usage:
    python Optimization_Ablation/run_optimization_ablation.py
    python Optimization_Ablation/run_optimization_ablation.py --variant few-shot-gpt
    python Optimization_Ablation/run_optimization_ablation.py --workers 16
    python Optimization_Ablation/run_optimization_ablation.py --lambda_isolated 1.5
"""

import os
import sys
import re
import json
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# ── Project root ──────────────────────────────────────────────────────────────
_SELF_DIR    = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_SELF_DIR, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── Ablation ILP ──────────────────────────────────────────────────────────────
from ablation.src.select_best_ER_schema_ablation import JointERILPComplexity

# ── Paths ─────────────────────────────────────────────────────────────────────
PROB_ROOT  = os.path.join(project_root, 'ablation', 'probability_estimation_ablation',
                          'output_0.5_1.0')
OUT_ROOT   = os.path.join(_SELF_DIR, 'output')
LOG_ROOT   = os.path.join(_SELF_DIR, 'log')
LAMBDA_FILE = os.path.join(project_root, 'optimization2', 'best_lambdas_with_isolated.json')

# ── Default lambdas (loaded from tuned file; fallback to these if file missing) ─
_DEFAULT_LAMBDAS = {
    "lambda_E":        0.8006,
    "lambda_A":        0.5875,
    "lambda_R":        0.8489,
    "lambda_noattr":   1.2993,
    "lambda_NM":       0.7217,
    "lambda_isolated": 1.3144,
}

def _load_lambdas() -> dict:
    """Load tuned lambda values; fall back to defaults if file is missing."""
    try:
        with open(LAMBDA_FILE, 'r') as f:
            data = json.load(f)
        # Support both {"lambdas": {...}} and flat {"lambda_E": ...} formats
        return data.get("lambdas", data)
    except Exception:
        return _DEFAULT_LAMBDAS


# ── All generation variants ───────────────────────────────────────────────────
# (probability_subfolder, output_subfolder)
VARIANTS = [
    (os.path.join('multi-llms', 'few-shot-gpt'),   os.path.join('multi-llms', 'few-shot-gpt')),
    (os.path.join('multi-llms', 'few-shot-llama'),  os.path.join('multi-llms', 'few-shot-llama')),
    (os.path.join('multi-llms', 'zero-shot-gpt'),   os.path.join('multi-llms', 'zero-shot-gpt')),
    (os.path.join('multi-llms', 'zero-shot-llama'), os.path.join('multi-llms', 'zero-shot-llama')),
    (os.path.join('one-llm', 'one_llm_few_shot_gpt'),   os.path.join('one-llm', 'one_llm_few_shot_gpt')),
    (os.path.join('one-llm', 'one_llm_few_shot_llama'),  os.path.join('one-llm', 'one_llm_few_shot_llama')),
    (os.path.join('one-llm', 'one_llm_zero_shot_gpt'),   os.path.join('one-llm', 'one_llm_zero_shot_gpt')),
    (os.path.join('one-llm', 'one_llm_zero_shot_llama'), os.path.join('one-llm', 'one_llm_zero_shot_llama')),
    (os.path.join('ToT', 'gpt'),   os.path.join('ToT', 'gpt')),
    (os.path.join('ToT', 'llama'), os.path.join('ToT', 'llama')),
]


# ── Logger setup ──────────────────────────────────────────────────────────────
def _setup_logger(variant_key: str):
    """
    Create a logger that writes simultaneously to stdout and
    Optimization_Ablation/log/{variant_key}.txt.
    """
    os.makedirs(LOG_ROOT, exist_ok=True)
    log_name = variant_key.replace(os.sep, '-').replace('/', '-')
    log_path = os.path.join(LOG_ROOT, f'{log_name}.txt')

    logger = logging.getLogger(f'opt_ablation.{log_name}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(message)s')

    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger, log_path


# ── Process a single JSON file ────────────────────────────────────────────────
def _process_file(args: tuple) -> bool:
    """
    args = (filename, in_folder, out_folder, lambdas)

    Reads the probability-annotated JSON, runs the ILP, and writes the
    selected ER schema (entities, attributes, relationships) to out_folder.

    Returns True on success, False on error or skip.
    """
    filename, in_folder, out_folder, lambdas = args

    out_path = os.path.join(out_folder, filename)
    if os.path.exists(out_path):
        return True   # already processed — idempotent re-runs

    try:
        t0 = time.time()
        with open(os.path.join(in_folder, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)

        entity_probs   = data.get('entity',       {})
        attribute_probs = data.get('attribute',    {})
        relation_rows   = data.get('relationship', [])

        # Skip files with no candidates (LLM produced empty output)
        if not entity_probs:
            # Write an empty schema so downstream evaluation doesn't choke
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({"entity": [], "attribute": {}, "relationship": []}, f,
                          ensure_ascii=False, indent=2)
            return True

        # Run ILP
        ilp = JointERILPComplexity(entity_probs, relation_rows, attribute_probs)
        score, sel_entities, sel_relations, sel_attributes, runtime = ilp.solve(
            lambda_E        = lambdas['lambda_E'],
            lambda_A        = lambdas['lambda_A'],
            lambda_R        = lambdas['lambda_R'],
            lambda_noattr   = lambdas['lambda_noattr'],
            lambda_NM       = lambdas['lambda_NM'],
            lambda_isolated = lambdas['lambda_isolated'],
            min_entities    = 3,
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
            "ilp_score":    round(score, 4) if score is not None else None,
            "ilp_runtime":  round(runtime, 3),
        }

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return True

    except Exception as e:
        print(f'  [ERR] {filename}: {e}')
        return False


# ── Run one variant ───────────────────────────────────────────────────────────
def _run_variant(prob_sub: str, out_sub: str, lambdas: dict,
                 max_workers: int) -> tuple:
    """Process all JSON files for one variant and return (ok, fail) counts."""
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
    logger.info(f'Log      : {log_path}')
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


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    # Load tuned lambdas (used as defaults; any CLI flag overrides individually)
    tuned = _load_lambdas()

    parser = argparse.ArgumentParser(
        description='Ablation ILP optimization for all probability variants.'
    )
    parser.add_argument('--variant', default=None,
        help='Run only variants whose prob_sub contains this string.')
    parser.add_argument('--workers', type=int, default=16,
        help='ThreadPoolExecutor workers per variant (default: 16).')
    parser.add_argument('--lambda_E',        type=float,
        default=tuned['lambda_E'],        help='Entity penalty.')
    parser.add_argument('--lambda_A',        type=float,
        default=tuned['lambda_A'],        help='Attribute penalty.')
    parser.add_argument('--lambda_R',        type=float,
        default=tuned['lambda_R'],        help='Relationship penalty.')
    parser.add_argument('--lambda_noattr',   type=float,
        default=tuned['lambda_noattr'],   help='No-attribute entity penalty.')
    parser.add_argument('--lambda_NM',       type=float,
        default=tuned['lambda_NM'],       help='N:M relationship extra penalty.')
    parser.add_argument('--lambda_isolated', type=float,
        default=tuned['lambda_isolated'], help='Isolated entity penalty.')
    args = parser.parse_args()

    lambdas = {
        'lambda_E':        args.lambda_E,
        'lambda_A':        args.lambda_A,
        'lambda_R':        args.lambda_R,
        'lambda_noattr':   args.lambda_noattr,
        'lambda_NM':       args.lambda_NM,
        'lambda_isolated': args.lambda_isolated,
    }

    variants = VARIANTS
    if args.variant:
        variants = [(p, o) for p, o in VARIANTS if args.variant in p]
        if not variants:
            print(f'No variant matches "{args.variant}". Available:')
            for p, _ in VARIANTS:
                print(f'  {p}')
            sys.exit(1)

    print(f'\nAblation ILP Optimization')
    print(f'Lambdas: {json.dumps(lambdas, indent=2)}')
    print(f'Workers per variant: {args.workers}\n')

    total_ok = total_fail = 0
    for prob_sub, out_sub in variants:
        ok, fail = _run_variant(prob_sub, out_sub, lambdas, args.workers)
        total_ok   += ok
        total_fail += fail

    print(f'\n{"="*65}')
    print(f'All variants done.  OK={total_ok}  Failed={total_fail}')
    print(f'Output root: {OUT_ROOT}')
    print(f'{"="*65}')


if __name__ == '__main__':
    main()
