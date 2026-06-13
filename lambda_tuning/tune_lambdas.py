"""
Tune ILP lambda hyperparameters using Optuna (Bayesian optimisation).

Strategy
--------
- Probability files exist for exercises 251–500 only.
- We split: 251–375 = tuning (train), 376–500 = held-out test.
- Objective : maximise mean F1 (Entity + Attribute + Relationship) on the tune set.
- Search    : Optuna TPE sampler (Bayesian) over all 5 lambda values.

Usage
-----
    python lambda_tuning/tune_lambdas.py \
        --prob_dir  output/probability/multi-llms/pro_fewshot_gpt_0.5_1.0 \
        --ref_dir   dataset/Datasets/Full-Dataset/Reference \
        --n_trials  60 \
        --n_jobs    1 \
        --workers   16 \
        --sample    50
"""

import os
import sys
import json
import math
import re
import random
import argparse
import csv
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ilp_complexity import JointERILPComplexity

try:
    from sentence_transformers import SentenceTransformer, util
    from scipy.optimize import linear_sum_assignment
    _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except ImportError:
    raise ImportError("pip install sentence-transformers scipy")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("pip install optuna")


# ──────────────────────────────────────────────────────────────────────
# Evaluation helpers  (same logic as evaluate_optimization_results)
# ──────────────────────────────────────────────────────────────────────

def clean_name(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return s.replace("_", " ").replace("-", " ").lower().strip()


def get_smart_mapping(list_out, list_ref, threshold=0.65):
    if not list_out or not list_ref:
        return {}
    emb_out = _ST_MODEL.encode([clean_name(i) for i in list_out], convert_to_tensor=True)
    emb_ref = _ST_MODEL.encode([clean_name(i) for i in list_ref], convert_to_tensor=True)
    cos_mat = util.cos_sim(emb_out, emb_ref).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(1 - cos_mat)
    return {
        list_out[r]: list_ref[c]
        for r, c in zip(row_ind, col_ind)
        if cos_mat[r, c] >= threshold
    }


def calc_f1(tp, total_out, total_ref):
    p = tp / total_out if total_out > 0 else 0.0
    r = tp / total_ref if total_ref > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_er(out_data: dict, ref_data: dict) -> float:
    """Return mean F1 (entity + attribute + relationship) for one exercise."""
    out_ents = [e for e in out_data.get("entity", [])
                if not str(e).upper().startswith("ASSOC_")]
    ref_ents = ref_data.get("entity", [])

    def _score(entities_to_use):
        e_map = get_smart_mapping(entities_to_use, ref_ents)
        f1_e = calc_f1(len(e_map), len(entities_to_use), len(ref_ents))

        tp_a = total_oa = total_ra = 0
        for oe, re_ in e_map.items():
            oa = out_data.get("attribut", out_data.get("attribute", {})).get(oe, [])
            ra = ref_data.get("attribut", ref_data.get("attribute", {})).get(re_, [])
            a_map = get_smart_mapping(oa, ra)
            tp_a += len(a_map); total_oa += len(oa); total_ra += len(ra)
        f1_a = calc_f1(tp_a, total_oa, total_ra)

        def edges(data, valid=None, mapping=None):
            s = set()
            for r in data.get("relationship", []):
                e1, e2 = r.get("entity_1"), r.get("entity_2")
                if valid and (e1 not in valid or e2 not in valid):
                    continue
                if mapping:
                    e1, e2 = mapping.get(e1), mapping.get(e2)
                if e1 and e2:
                    s.add(tuple(sorted((str(e1), str(e2)))))
            return s

        emap_keys = set(e_map.keys())
        out_rel = edges(out_data, valid=entities_to_use, mapping=e_map)
        ref_rel = edges(ref_data)
        total_out_rel = sum(
            1 for r in out_data.get("relationship", [])
            if r.get("entity_1") in emap_keys and r.get("entity_2") in emap_keys
        )
        f1_r = calc_f1(len(out_rel & ref_rel), total_out_rel, len(ref_rel))

        return (f1_e + f1_a + f1_r) / 3.0

    return max(_score(out_ents), _score(out_data.get("entity", [])))


# ──────────────────────────────────────────────────────────────────────
# One trial: run ILP on a sample, return mean F1
# ──────────────────────────────────────────────────────────────────────

def run_trial(
    ex_ids: List[int],
    prob_dir: str,
    ref_dir: str,
    lambdas: Dict[str, float],
    max_workers: int = 16,
) -> float:

    def process_one(ex_id):
        prob_path = os.path.join(prob_dir, f"{ex_id}.json")
        ref_path  = os.path.join(ref_dir,  f"exercise{ex_id}-baseline.txt")
        if not (os.path.exists(prob_path) and os.path.exists(ref_path)):
            return None
        try:
            with open(prob_path, "r") as f:
                data = json.load(f)
            with open(ref_path, "r") as f:
                ref_data = json.load(f)

            solver = JointERILPComplexity(
                data.get("entity", {}),
                data.get("relationship", []),
                data.get("attribute", {}),
            )
            _, sel_ents, sel_rels, sel_attrs, _ = solver.solve(
                min_entities=3, **lambdas
            )

            out_data = {
                "entity": sel_ents,
                "attribut": sel_attrs,
                "relationship": sel_rels,
            }
            return evaluate_er(out_data, ref_data)
        except Exception:
            return None

    scores = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(process_one, i): i for i in ex_ids}
        for fut in as_completed(futs):
            v = fut.result()
            if v is not None:
                scores.append(v)

    return float(np.mean(scores)) if scores else 0.0


# ──────────────────────────────────────────────────────────────────────
# Optuna objective
# ──────────────────────────────────────────────────────────────────────

def make_objective(prob_dir, ref_dir, ex_ids, max_workers):
    def objective(trial: optuna.Trial) -> float:
        # Lower bounds are meaningful thresholds:
        #   log_odds(p) < lambda  →  item excluded
        #   lambda = 0.5  →  excludes items with p < sigmoid(0.5) ≈ 0.62
        #   lambda = 1.0  →  excludes items with p < sigmoid(1.0) ≈ 0.73
        # We enforce a floor of 0.4 so the ILP always does real filtering.
        lambdas = {
            "lambda_E":        trial.suggest_float("lambda_E",        0.4, 3.0, log=True),
            "lambda_A":        trial.suggest_float("lambda_A",        0.4, 3.0, log=True),
            "lambda_R":        trial.suggest_float("lambda_R",        0.4, 3.0, log=True),
            "lambda_noattr":   trial.suggest_float("lambda_noattr",   0.4, 5.0, log=True),
            "lambda_NM":       trial.suggest_float("lambda_NM",       0.4, 5.0, log=True),
            "lambda_isolated": trial.suggest_float("lambda_isolated", 0.4, 5.0, log=True),
        }
        score = run_trial(ex_ids, prob_dir, ref_dir, lambdas, max_workers)
        return score  # Optuna maximises by default when direction="maximize"
    return objective


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_dir",  required=True,
                        help="Path to probability JSON folder (e.g. output/probability/...)")
    parser.add_argument("--ref_dir",   required=True,
                        help="Path to reference folder (exercise251-baseline.txt, ...)")
    parser.add_argument("--n_trials",  type=int, default=60,
                        help="Number of Optuna trials")
    parser.add_argument("--n_jobs",    type=int, default=1,
                        help="Parallel Optuna workers (each worker uses --workers threads)")
    parser.add_argument("--workers",   type=int, default=16,
                        help="ThreadPool workers for ILP inside each trial")
    parser.add_argument("--sample",    type=int, default=50,
                        help="Exercises sampled per trial from the tune split (251-375)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--out",       type=str,
                        default=os.path.join(project_root, "lambda_tuning/best_lambdas.json"),
                        help="Where to save the best lambda values")
    args = parser.parse_args()

    random.seed(args.seed)

    # Tune split: 251–375  (376–500 is held-out test, never touched here)
    all_ids = []
    for i in range(251, 376):
        prob_path = os.path.join(args.prob_dir, f"{i}.json")
        ref_path  = os.path.join(args.ref_dir,  f"exercise{i}-baseline.txt")
        if os.path.exists(prob_path) and os.path.exists(ref_path):
            all_ids.append(i)

    print(f"Available training exercises: {len(all_ids)}")
    sample_size = min(args.sample, len(all_ids))
    ex_ids = random.sample(all_ids, sample_size)
    print(f"Using {len(ex_ids)} exercises per trial.\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    # Enqueue two baseline trials to anchor the search
    study.enqueue_trial({  # original hand-picked
        "lambda_E": 1.2, "lambda_A": 1.0, "lambda_R": 1.0,
        "lambda_noattr": 2.0, "lambda_NM": 1.5, "lambda_isolated": 2.0,
    })
    study.enqueue_trial({  # more aggressive filtering
        "lambda_E": 0.8, "lambda_A": 0.8, "lambda_R": 0.8,
        "lambda_noattr": 1.5, "lambda_NM": 1.0, "lambda_isolated": 1.5,
    })
    study.enqueue_trial({  # previous best + isolated
        "lambda_E": 0.603, "lambda_A": 0.455, "lambda_R": 0.688,
        "lambda_noattr": 1.656, "lambda_NM": 1.996, "lambda_isolated": 2.0,
    })

    t0 = time.time()
    study.optimize(
        make_objective(args.prob_dir, args.ref_dir, ex_ids, args.workers),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"Best mean F1 : {best.value:.4f}  (after {args.n_trials} trials, {elapsed:.1f}s)")
    print(f"Best lambdas :")
    for k, v in best.params.items():
        print(f"  {k:15s} = {v:.4f}")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"best_f1": best.value, "lambdas": best.params}, f, indent=2)
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()
