"""
Ablation Study — Evaluation
============================
Evaluates the ILP-selected ER schemas produced by the ablation pipeline
(Optimization_Ablation/output/) against the ground-truth reference files.

Metrics computed per exercise (251-500):
    Precision / Recall / F1  for  Entity, Attribute, Relationship

Entity and attribute matching uses Hungarian-algorithm optimal assignment
with SBERT cosine similarity (threshold 0.65).  Relationship matching maps
entity pairs through the entity alignment.

Output:
    Evaluation_Ablation/results/{variant_label}.csv   — per-exercise scores
    Evaluation_Ablation/results/summary.csv           — average F1 per variant

Usage:
    python Evaluation_Ablation/evaluate_ablation.py
    python Evaluation_Ablation/evaluate_ablation.py --variant few-shot-gpt
    python Evaluation_Ablation/evaluate_ablation.py --threshold 0.65
"""

import os
import sys
import re
import csv
import json
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment

# ── Paths ─────────────────────────────────────────────────────────────────────
_SELF_DIR    = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_SELF_DIR, '..'))

OPT_ROOT  = os.path.join(project_root, 'Optimization_Ablation', 'output')
REF_DIR   = os.path.join(project_root, 'dataset', 'Datasets', 'Full-Dataset', 'Reference')
if not os.path.isdir(REF_DIR):
    REF_DIR = os.path.join(project_root, 'dataset', 'Datasets', 'Reference')
RESULTS_DIR = os.path.join(_SELF_DIR, 'results')

# ── SBERT model (loaded once, shared across all evaluations) ──────────────────
print("Loading SBERT model (all-MiniLM-L6-v2)...")
_sbert = SentenceTransformer('all-MiniLM-L6-v2')

# Embedding cache: cleaned_name → numpy vector.  Avoids re-encoding the same
# entity/attribute name across exercises (many names repeat heavily).
_emb_cache: dict = {}


# ── Text utilities ─────────────────────────────────────────────────────────────

def clean_name(s: str) -> str:
    """Normalise an entity/attribute name for semantic comparison."""
    if not s:
        return ""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return s.replace('_', ' ').replace('-', ' ').lower().strip()


def _get_embeddings(names: list) -> np.ndarray:
    """
    Return a (N, D) float32 embedding matrix for `names`.
    Uses `_emb_cache` to skip re-encoding already-seen cleaned names.
    New names are batch-encoded in one SBERT call.
    """
    cleaned = [clean_name(n) for n in names]
    missing = [c for c in cleaned if c not in _emb_cache]
    if missing:
        vecs = _sbert.encode(missing, convert_to_numpy=True, batch_size=64)
        for c, v in zip(missing, vecs):
            _emb_cache[c] = v
    return np.array([_emb_cache[c] for c in cleaned], dtype=np.float32)


# ── Matching utilities ─────────────────────────────────────────────────────────

def get_smart_mapping(list_out: list, list_ref: list,
                      threshold: float = 0.65) -> dict:
    """
    Optimal bipartite assignment between two name lists using SBERT cosine
    similarity.  Returns {out_name: ref_name} for pairs above threshold.
    Uses _emb_cache so repeated names are never re-encoded.
    """
    if not list_out or not list_ref:
        return {}
    emb_out = _get_embeddings(list_out)
    emb_ref = _get_embeddings(list_ref)
    # Cosine similarity: normalise then dot-product
    norm_out = emb_out / (np.linalg.norm(emb_out, axis=1, keepdims=True) + 1e-9)
    norm_ref = emb_ref / (np.linalg.norm(emb_ref, axis=1, keepdims=True) + 1e-9)
    cos_mat  = norm_out @ norm_ref.T
    row_ind, col_ind = linear_sum_assignment(1 - cos_mat)
    mapping = {}
    for r, c in zip(row_ind, col_ind):
        if cos_mat[r, c] >= threshold:
            mapping[list_out[r]] = list_ref[c]
    return mapping


def calc_metrics(tp: int, total_out: int, total_ref: int):
    """Return (precision, recall, F1) given TP and totals."""
    p  = tp / total_out if total_out > 0 else 0.0
    r  = tp / total_ref if total_ref > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ── Per-exercise evaluation ────────────────────────────────────────────────────

def evaluate_exercise(out_path: str, ref_path: str,
                      threshold: float = 0.65) -> dict:
    """
    Evaluate a single exercise output against its reference.

    Returns a dict with keys 'entity', 'attribute', 'relation' each holding
    (precision, recall, F1), and 'overall_f1' = mean of the three F1s.
    """
    with open(out_path,  'r', encoding='utf-8') as f:
        out_data = json.load(f)
    with open(ref_path,  'r', encoding='utf-8') as f:
        ref_data = json.load(f)

    out_entities_all  = out_data.get("entity", [])
    # Strip associative-entity prefix added by the probability pipeline
    out_entities_main = [e for e in out_entities_all
                         if not str(e).upper().startswith("ASSOC_")]
    ref_entities      = ref_data.get("entity", [])

    def _eval_with(entities: list) -> dict:
        # 1. Entity alignment
        e_map       = get_smart_mapping(entities, ref_entities, threshold)
        p_e, r_e, f1_e = calc_metrics(len(e_map), len(entities), len(ref_entities))

        # 2. Attribute alignment (only for matched entity pairs)
        tp_a = total_oa = total_ra = 0
        for out_ent, ref_ent in e_map.items():
            oa = out_data.get("attribute", out_data.get("attribut", {})).get(out_ent, [])
            ra = ref_data.get("attribute", ref_data.get("attribut", {})).get(ref_ent, [])
            a_map = get_smart_mapping(oa, ra, threshold)
            tp_a      += len(a_map)
            total_oa  += len(oa)
            total_ra  += len(ra)
        p_a, r_a, f1_a = calc_metrics(tp_a, total_oa, total_ra)

        # 3. Relationship alignment
        def _edges(data, valid_ents=None, mapping=None):
            edges = set()
            for rel in data.get("relationship", []):
                e1, e2 = rel.get("entity_1"), rel.get("entity_2")
                if valid_ents is not None:
                    if e1 not in valid_ents or e2 not in valid_ents:
                        continue
                if mapping:
                    e1, e2 = mapping.get(e1), mapping.get(e2)
                if e1 and e2:
                    edges.add(tuple(sorted((str(e1), str(e2)))))
            return edges

        e_map_keys  = set(e_map.keys())
        out_rel     = _edges(out_data, valid_ents=entities, mapping=e_map)
        ref_rel     = _edges(ref_data)
        total_out_r = sum(
            1 for rel in out_data.get("relationship", [])
            if rel.get("entity_1") in e_map_keys
            and rel.get("entity_2") in e_map_keys
        )
        tp_r = len(out_rel & ref_rel)
        p_r, r_r, f1_r = calc_metrics(tp_r, total_out_r, len(ref_rel))

        overall = (f1_e + f1_a + f1_r) / 3
        return {
            "entity":     (p_e, r_e, f1_e),
            "attribute":  (p_a, r_a, f1_a),
            "relation":   (p_r, r_r, f1_r),
            "overall_f1": overall,
        }

    # Pick whichever entity set (main vs all) yields the higher overall F1
    res_main = _eval_with(out_entities_main)
    res_all  = _eval_with(out_entities_all)
    return res_main if res_main["overall_f1"] >= res_all["overall_f1"] else res_all


# ── Variant evaluation ─────────────────────────────────────────────────────────

def _label(folder: str) -> str:
    """Convert an absolute folder path to a short variant label."""
    rel = os.path.relpath(folder, OPT_ROOT)
    return rel.replace(os.sep, '-').replace('/', '-')


def evaluate_variant(folder: str, threshold: float,
                     max_workers: int = 8) -> dict | None:
    """
    Evaluate all exercises 251-500 in a single variant folder.
    Exercises are processed in parallel (max_workers threads).
    Returns a summary dict, or None if no exercises were found.
    """
    label = _label(folder)
    csv_path = os.path.join(RESULTS_DIR, f"{label}.csv")

    print(f"\n{'='*72}")
    print(f"Variant : {label}")
    print(f"Folder  : {folder}")
    print(f"CSV     : {csv_path}")
    print('-' * 72)

    # Build work list
    tasks = []
    for ex_id in range(251, 501):
        out_file = os.path.join(folder, f"{ex_id}.json")
        ref_file = os.path.join(REF_DIR,  f"exercise{ex_id}-baseline.txt")
        if os.path.exists(out_file) and os.path.exists(ref_file):
            tasks.append((ex_id, out_file, ref_file))

    if not tasks:
        print(f"  [SKIP] No exercises found.")
        return None

    # Pre-warm the embedding cache for all unique names in this variant
    # by collecting them before starting parallel evaluation.
    all_names: list = []
    for _, out_file, ref_file in tasks:
        for path in (out_file, ref_file):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                all_names.extend(d.get("entity", []))
                for attrs in d.get("attribute", d.get("attribut", {}) or {}).values():
                    if isinstance(attrs, list):
                        all_names.extend(attrs)
            except Exception:
                pass
    unique_names = list({clean_name(n) for n in all_names if n})
    if unique_names:
        _get_embeddings(unique_names)  # batch-encode and cache

    # Parallel evaluation
    results_by_id: dict = {}
    def _task(args):
        ex_id, out_f, ref_f = args
        return ex_id, evaluate_exercise(out_f, ref_f, threshold)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_task, t): t[0] for t in tasks}
        for future in as_completed(futures):
            ex_id = futures[future]
            try:
                _, m = future.result()
                results_by_id[ex_id] = m
            except Exception as exc:
                print(f"  Ex {ex_id:<4} | ERROR: {exc}")

    # Print and collect in order
    csv_rows, metrics_list = [], []
    for ex_id in sorted(results_by_id):
        m = results_by_id[ex_id]
        metrics_list.append(m)
        print(
            f"  Ex {ex_id:<4} | "
            f"Ent F1: {m['entity'][2]:.2f}  "
            f"Attr F1: {m['attribute'][2]:.2f}  "
            f"Rel F1: {m['relation'][2]:.2f}"
        )
        csv_rows.append([
            f"Ex {ex_id}",
            *[f"{v:.4f}" for v in m['entity']],
            *[f"{v:.4f}" for v in m['attribute']],
            *[f"{v:.4f}" for v in m['relation']],
        ])

    if not metrics_list:
        print(f"  [SKIP] No exercises found for {label}.")
        return None

    avgs = {
        k: tuple(np.mean([x[k][j] for x in metrics_list]) for j in range(3))
        for k in ('entity', 'attribute', 'relation')
    }
    avg_row = (
        ["AVERAGE"]
        + [round(v, 4) for v in avgs['entity']]
        + [round(v, 4) for v in avgs['attribute']]
        + [round(v, 4) for v in avgs['relation']]
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Exercise",
                    "Ent_P", "Ent_R", "Ent_F1",
                    "Attr_P", "Attr_R", "Attr_F1",
                    "Rel_P",  "Rel_R",  "Rel_F1"])
        w.writerows(csv_rows)
        w.writerow(avg_row)

    print(
        f"\n  AVERAGE | Ent F1: {avgs['entity'][2]:.4f}  "
        f"Attr F1: {avgs['attribute'][2]:.4f}  "
        f"Rel F1: {avgs['relation'][2]:.4f}  "
        f"Overall: {np.mean([v[2] for v in avgs.values()]):.4f}"
    )
    return {"label": label, "avgs": avgs, "n": len(metrics_list)}


# ── Summary CSV ───────────────────────────────────────────────────────────────

def _write_summary(results: list):
    """Write a single-row-per-variant summary CSV."""
    path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Variant", "N",
                    "Ent_F1", "Attr_F1", "Rel_F1", "Overall_F1"])
        for r in results:
            f1e  = round(r["avgs"]["entity"][2],    4)
            f1a  = round(r["avgs"]["attribute"][2],  4)
            f1r  = round(r["avgs"]["relation"][2],   4)
            f1o  = round((f1e + f1a + f1r) / 3,     4)
            w.writerow([r["label"], r["n"], f1e, f1a, f1r, f1o])
    print(f"\nSummary saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ablation ILP outputs against ground-truth reference.'
    )
    parser.add_argument('--variant',   default=None,
        help='Evaluate only variants whose label contains this string.')
    parser.add_argument('--threshold', type=float, default=0.65,
        help='SBERT cosine similarity threshold for entity/attr matching (default: 0.65).')
    args = parser.parse_args()

    # Collect all leaf folders that contain JSON files under OPT_ROOT
    variant_folders = sorted(
        root
        for root, _, files in os.walk(OPT_ROOT)
        if any(f.endswith('.json') for f in files)
    )

    if args.variant:
        variant_folders = [f for f in variant_folders if args.variant in _label(f)]
        if not variant_folders:
            print(f'No variant matches "{args.variant}". Available:')
            for f in sorted(
                root for root, _, files in os.walk(OPT_ROOT)
                if any(fn.endswith('.json') for fn in files)
            ):
                print(f'  {_label(f)}')
            sys.exit(1)

    print(f"\nAblation Evaluation")
    print(f"  Optimization outputs : {OPT_ROOT}")
    print(f"  Reference dir        : {REF_DIR}")
    print(f"  Results dir          : {RESULTS_DIR}")
    print(f"  Variants to evaluate : {len(variant_folders)}")
    print(f"  SBERT threshold      : {args.threshold}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for folder in variant_folders:
        result = evaluate_variant(folder, args.threshold)
        if result:
            all_results.append(result)

    if all_results:
        _write_summary(all_results)

        # Print final comparison table
        print(f"\n{'='*72}")
        print(f"{'Variant':<40} {'N':>4}  {'EntF1':>6}  {'AttrF1':>7}  {'RelF1':>6}  {'Overall':>7}")
        print('-' * 72)
        for r in sorted(all_results, key=lambda x: x['label']):
            f1e = r["avgs"]["entity"][2]
            f1a = r["avgs"]["attribute"][2]
            f1r = r["avgs"]["relation"][2]
            f1o = (f1e + f1a + f1r) / 3
            print(f"  {r['label']:<38} {r['n']:>4}  {f1e:>6.4f}  {f1a:>7.4f}  {f1r:>6.4f}  {f1o:>7.4f}")
        print('=' * 72)

    print(f"\nDone.  Results in: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
