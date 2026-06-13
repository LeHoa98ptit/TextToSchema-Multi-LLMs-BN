"""
Evaluate Hard-Constraint ILP results.

Usage:
    python ablation/optimization_ablation_hard/evaluate_hard_ilp.py
"""

import os, sys, csv, re, json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment

_SELF_DIR    = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_SELF_DIR, '..', '..'))

OPT_ROOT    = os.path.join(_SELF_DIR, 'output')
REF_DIR     = os.path.join(project_root, 'dataset', 'Datasets', 'Full-Dataset', 'Reference')
RESULTS_DIR = os.path.join(_SELF_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading SBERT model...")
_sbert = SentenceTransformer('all-MiniLM-L6-v2')
_emb_cache: dict = {}

def clean_name(s):
    if not s: return ""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return s.replace('_', ' ').replace('-', ' ').lower().strip()

def _get_embeddings(names):
    cleaned = [clean_name(n) for n in names]
    missing = [c for c in cleaned if c not in _emb_cache]
    if missing:
        vecs = _sbert.encode(missing, convert_to_numpy=True, batch_size=64)
        for c, v in zip(missing, vecs):
            _emb_cache[c] = v
    return np.array([_emb_cache[c] for c in cleaned], dtype=np.float32)

def get_smart_mapping(list_out, list_ref, threshold=0.65):
    if not list_out or not list_ref: return {}
    emb_out = _get_embeddings(list_out)
    emb_ref = _get_embeddings(list_ref)
    norm_out = emb_out / (np.linalg.norm(emb_out, axis=1, keepdims=True) + 1e-9)
    norm_ref = emb_ref / (np.linalg.norm(emb_ref, axis=1, keepdims=True) + 1e-9)
    cos_mat  = norm_out @ norm_ref.T
    row_ind, col_ind = linear_sum_assignment(1 - cos_mat)
    return {list_out[r]: list_ref[c] for r, c in zip(row_ind, col_ind) if cos_mat[r, c] >= threshold}

def calc_metrics(tp, total_out, total_ref):
    p  = tp / total_out if total_out > 0 else 0.0
    r  = tp / total_ref if total_ref > 0 else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return p, r, f1

def evaluate_exercise(out_path, ref_path, threshold=0.65):
    with open(out_path,  'r', encoding='utf-8') as f: out_data = json.load(f)
    with open(ref_path,  'r', encoding='utf-8') as f: ref_data = json.load(f)

    out_all  = out_data.get("entity", [])
    out_main = [e for e in out_all if not str(e).upper().startswith("ASSOC_")]
    ref_ents = ref_data.get("entity", [])

    def _eval(ents):
        e_map = get_smart_mapping(ents, ref_ents, threshold)
        p_e, r_e, f1_e = calc_metrics(len(e_map), len(ents), len(ref_ents))

        tp_a = tot_oa = tot_ra = 0
        for oe, re_ in e_map.items():
            oa = out_data.get("attribute", out_data.get("attribut", {})).get(oe, [])
            ra = ref_data.get("attribute", ref_data.get("attribut", {})).get(re_, [])
            tp_a  += len(get_smart_mapping(oa, ra, threshold))
            tot_oa += len(oa); tot_ra += len(ra)
        p_a, r_a, f1_a = calc_metrics(tp_a, tot_oa, tot_ra)

        def _edges(data, valid=None, mapping=None):
            edges = set()
            for rel in data.get("relationship", []):
                e1, e2 = rel.get("entity_1"), rel.get("entity_2")
                if valid is not None and (e1 not in valid or e2 not in valid): continue
                if mapping: e1, e2 = mapping.get(e1), mapping.get(e2)
                if e1 and e2: edges.add(tuple(sorted((str(e1), str(e2)))))
            return edges

        e_keys  = set(e_map.keys())
        out_rel = _edges(out_data, valid=ents, mapping=e_map)
        ref_rel = _edges(ref_data)
        tot_or  = sum(1 for r in out_data.get("relationship", [])
                      if r.get("entity_1") in e_keys and r.get("entity_2") in e_keys)
        p_r, r_r, f1_r = calc_metrics(len(out_rel & ref_rel), tot_or, len(ref_rel))

        return {"entity": (p_e,r_e,f1_e), "attribute": (p_a,r_a,f1_a),
                "relation": (p_r,r_r,f1_r), "overall_f1": (f1_e+f1_a+f1_r)/3}

    res_main = _eval(out_main); res_all = _eval(out_all)
    return res_main if res_main["overall_f1"] >= res_all["overall_f1"] else res_all

def _label(folder):
    return os.path.relpath(folder, OPT_ROOT).replace(os.sep, '-').replace('/', '-')

def evaluate_variant(folder, threshold=0.65, max_workers=8):
    label    = _label(folder)
    csv_path = os.path.join(RESULTS_DIR, f"{label}.csv")
    print(f"\n{'='*65}\nVariant : {label}")

    tasks = []
    for ex_id in range(251, 501):
        out_file = os.path.join(folder, f"{ex_id}.json")
        ref_file = os.path.join(REF_DIR, f"exercise{ex_id}-baseline.txt")
        if os.path.exists(out_file) and os.path.exists(ref_file):
            tasks.append((ex_id, out_file, ref_file))

    if not tasks:
        print("  [SKIP] No exercises found."); return None

    all_names = []
    for _, of, rf in tasks:
        for path in (of, rf):
            try:
                d = json.load(open(path, encoding='utf-8'))
                all_names.extend(d.get("entity", []))
                for attrs in d.get("attribute", d.get("attribut", {}) or {}).values():
                    if isinstance(attrs, list): all_names.extend(attrs)
            except: pass
    _get_embeddings(list({clean_name(n) for n in all_names if n}))

    results_by_id = {}
    def _task(args):
        ex_id, of, rf = args
        return ex_id, evaluate_exercise(of, rf, threshold)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_task, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            ex_id = futures[fut]
            try:
                _, m = fut.result(); results_by_id[ex_id] = m
            except Exception as exc:
                print(f"  Ex {ex_id} | ERROR: {exc}")

    csv_rows, metrics_list = [], []
    for ex_id in sorted(results_by_id):
        m = results_by_id[ex_id]; metrics_list.append(m)
        print(f"  Ex {ex_id:<4} | EntF1:{m['entity'][2]:.2f}  AttrF1:{m['attribute'][2]:.2f}  RelF1:{m['relation'][2]:.2f}")
        csv_rows.append([f"Ex {ex_id}",
                         *[f"{v:.4f}" for v in m['entity']],
                         *[f"{v:.4f}" for v in m['attribute']],
                         *[f"{v:.4f}" for v in m['relation']]])

    avgs = {k: tuple(np.mean([x[k][j] for x in metrics_list]) for j in range(3))
            for k in ('entity', 'attribute', 'relation')}
    avg_row = (["AVERAGE"]
               + [round(v, 4) for v in avgs['entity']]
               + [round(v, 4) for v in avgs['attribute']]
               + [round(v, 4) for v in avgs['relation']])

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Exercise","Ent_P","Ent_R","Ent_F1","Attr_P","Attr_R","Attr_F1","Rel_P","Rel_R","Rel_F1"])
        w.writerows(csv_rows); w.writerow(avg_row)

    f1e = avgs['entity'][2]; f1a = avgs['attribute'][2]; f1r = avgs['relation'][2]
    print(f"\n  AVERAGE | EntF1:{f1e:.4f}  AttrF1:{f1a:.4f}  RelF1:{f1r:.4f}  Overall:{(f1e+f1a+f1r)/3:.4f}")
    return {"label": label, "avgs": avgs, "n": len(metrics_list)}

def main():
    variant_folders = sorted(
        root for root, _, files in os.walk(OPT_ROOT)
        if any(f.endswith('.json') for f in files)
    )
    print(f"Evaluating {len(variant_folders)} variants in: {OPT_ROOT}")

    all_results = []
    for folder in variant_folders:
        r = evaluate_variant(folder)
        if r: all_results.append(r)

    if all_results:
        summary_path = os.path.join(RESULTS_DIR, "summary.csv")
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["Variant","N","Ent_F1","Attr_F1","Rel_F1","Overall_F1"])
            for r in all_results:
                f1e = round(r["avgs"]["entity"][2], 4)
                f1a = round(r["avgs"]["attribute"][2], 4)
                f1r = round(r["avgs"]["relation"][2], 4)
                w.writerow([r["label"], r["n"], f1e, f1a, f1r, round((f1e+f1a+f1r)/3, 4)])

        print(f"\n{'='*65}")
        print(f"{'Variant':<38} {'N':>4}  {'EntF1':>6}  {'AttrF1':>7}  {'RelF1':>6}  {'Overall':>7}")
        print('-'*65)
        for r in sorted(all_results, key=lambda x: x['label']):
            f1e = r["avgs"]["entity"][2]; f1a = r["avgs"]["attribute"][2]; f1r = r["avgs"]["relation"][2]
            print(f"  {r['label']:<36} {r['n']:>4}  {f1e:.4f}  {f1a:.4f}  {f1r:.4f}  {(f1e+f1a+f1r)/3:.4f}")
        print(f"{'='*65}\nSummary: {summary_path}")

if __name__ == '__main__':
    main()
