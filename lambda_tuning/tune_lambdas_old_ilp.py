"""
Tune lambda_E, lambda_A, lambda_R for the original JointERILPSolverFinal_1
(probability-only, no complexity terms).

Usage:
    python lambda_tuning/tune_lambdas_old_ilp.py \
        --prob_dir output/probability/multi-llms/pro_fewshot_gpt_0.5_1.0 \
        --ref_dir  dataset/Datasets/Full-Dataset/Reference \
        --n_trials 60 --workers 16 --sample 50
"""

import os, sys, json, math, re, random, argparse, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.finding_the_best_er_model import JointERILPSolverFinal_1

from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

_ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def clean_name(s):
    if not s: return ""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    return s.replace("_", " ").replace("-", " ").lower().strip()

def get_smart_mapping(list_out, list_ref, threshold=0.65):
    if not list_out or not list_ref: return {}
    emb_out = _ST_MODEL.encode([clean_name(i) for i in list_out], convert_to_tensor=True)
    emb_ref = _ST_MODEL.encode([clean_name(i) for i in list_ref], convert_to_tensor=True)
    cos_mat = util.cos_sim(emb_out, emb_ref).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(1 - cos_mat)
    return {list_out[r]: list_ref[c] for r, c in zip(row_ind, col_ind) if cos_mat[r, c] >= threshold}

def calc_f1(tp, total_out, total_ref):
    p = tp / total_out if total_out > 0 else 0.0
    r = tp / total_ref if total_ref > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

def evaluate_er(out_data, ref_data):
    out_ents = [e for e in out_data.get("entity", []) if not str(e).upper().startswith("ASSOC_")]
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
                if valid and (e1 not in valid or e2 not in valid): continue
                if mapping: e1, e2 = mapping.get(e1), mapping.get(e2)
                if e1 and e2: s.add(tuple(sorted((str(e1), str(e2)))))
            return s
        emap_keys = set(e_map.keys())
        out_rel = edges(out_data, valid=entities_to_use, mapping=e_map)
        ref_rel = edges(ref_data)
        total_out_rel = sum(1 for r in out_data.get("relationship", [])
                            if r.get("entity_1") in emap_keys and r.get("entity_2") in emap_keys)
        f1_r = calc_f1(len(out_rel & ref_rel), total_out_rel, len(ref_rel))
        return (f1_e + f1_a + f1_r) / 3.0

    return max(_score(out_ents), _score(out_data.get("entity", [])))


def run_trial(ex_ids, prob_dir, ref_dir, lambda_E, lambda_A, lambda_R, max_workers=16):
    def process_one(ex_id):
        prob_path = os.path.join(prob_dir, f"{ex_id}.json")
        ref_path  = os.path.join(ref_dir,  f"exercise{ex_id}-baseline.txt")
        if not (os.path.exists(prob_path) and os.path.exists(ref_path)):
            return None
        try:
            with open(prob_path) as f: data = json.load(f)
            with open(ref_path)  as f: ref_data = json.load(f)
            solver = JointERILPSolverFinal_1(
                data.get("entity", {}),
                data.get("relationship", []),
                data.get("attribute", {}),
            )
            _, sel_ents, sel_rels, sel_attrs, _ = solver.solve(
                lambda_E=lambda_E, lambda_A=lambda_A, lambda_R=lambda_R, min_entities=3
            )
            out_data = {"entity": sel_ents, "attribut": sel_attrs, "relationship": sel_rels}
            return evaluate_er(out_data, ref_data)
        except Exception:
            return None

    scores = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fut in as_completed({ex.submit(process_one, i): i for i in ex_ids}):
            v = fut.result()
            if v is not None: scores.append(v)
    return float(np.mean(scores)) if scores else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_dir", required=True)
    parser.add_argument("--ref_dir",  required=True)
    parser.add_argument("--n_trials", type=int, default=60)
    parser.add_argument("--workers",  type=int, default=16)
    parser.add_argument("--sample",   type=int, default=50)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out", default=os.path.join(project_root, "lambda_tuning/best_lambdas_old_ilp.json"))
    args = parser.parse_args()

    random.seed(args.seed)
    all_ids = [i for i in range(251, 376)
               if os.path.exists(os.path.join(args.prob_dir, f"{i}.json"))
               and os.path.exists(os.path.join(args.ref_dir, f"exercise{i}-baseline.txt"))]
    ex_ids = random.sample(all_ids, min(args.sample, len(all_ids)))
    print(f"Tuning OLD ILP (lambda_E, lambda_A, lambda_R) | {len(ex_ids)} exercises/trial\n")

    def objective(trial):
        # Floor 0.4: ensures log_odds(p) > lambda only when p > sigmoid(0.4) ≈ 0.60
        lE = trial.suggest_float("lambda_E", 0.4, 3.0, log=True)
        lA = trial.suggest_float("lambda_A", 0.4, 3.0, log=True)
        lR = trial.suggest_float("lambda_R", 0.4, 3.0, log=True)
        return run_trial(ex_ids, args.prob_dir, args.ref_dir, lE, lA, lR, args.workers)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    # Baseline: current hand-picked values
    study.enqueue_trial({"lambda_E": 1.2, "lambda_A": 1.0, "lambda_R": 1.0})

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"\n{'='*50}")
    print(f"Best mean F1 : {best.value:.4f}  ({elapsed:.1f}s)")
    print(f"Best lambdas :")
    for k, v in best.params.items():
        print(f"  {k:12s} = {v:.4f}")
    print(f"{'='*50}")

    with open(args.out, "w") as f:
        json.dump({"best_f1": best.value, "lambdas": best.params}, f, indent=2)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
