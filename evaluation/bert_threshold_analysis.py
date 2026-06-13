#!/usr/bin/env python3
"""
bert_threshold_analysis.py
===========================
Statistical analysis justifying bert_threshold = 0.50 in the ER evaluation pipeline.

Analysis A – Score Distribution (100 exercises)
  • After removing easy exact/WordNet matches (steps 1–2), compute BERT cosine
    similarities for all remaining candidate pairs.
  • Hungarian-assigned pairs  → "BERT-zone True Pairs"  (model should match)
  • Non-assigned cross-pairs  → "BERT-zone False Pairs" (model should reject)
  • Reports mean, std, percentiles; KS-test; plots KDE + histogram overlay.

Analysis B – Threshold Sensitivity (80 exercises × 11 thresholds: 0.30–0.80)
  • BERT embedding matrices are pre-computed ONCE per exercise, then reused.
  • Reports Entity / Attribute / Relation / Average F1 at each threshold.
  • Plots F1 curves with the selected threshold marked.

Outputs → results/bert_threshold_analysis/
  bert_score_distributions.png
  threshold_sensitivity.png
  bert_threshold_report.txt
"""

import os
import sys
import json
import re
import random
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from collections import defaultdict

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util as st_util
from scipy.optimize import linear_sum_assignment
import spacy

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

OPT_DIR = os.path.join(
    project_root,
    "output/optimization/multi-llms/opt_fewshot_llama_0.5_1.0-(1.2-1.0-1.0)"
)
REF_DIR = os.path.join(project_root, "dataset/Datasets/Full-Dataset/Reference")
if not os.path.exists(REF_DIR):
    REF_DIR = os.path.join(project_root, "dataset/Datasets/Reference")

OUTPUT_DIR = os.path.join(project_root, "results/bert_threshold_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DIST_SAMPLE      = 100                              # exercises for distribution analysis
SENS_SAMPLE      = 80                               # exercises for sensitivity analysis
THRESHOLDS       = np.round(np.arange(0.30, 0.81, 0.05), 2)   # [0.30 … 0.80]
CHOSEN_THRESHOLD = 0.50
SEED             = 42
random.seed(SEED)

# ─────────────────────────────────────────────────────────────────
# MODEL INIT
# ─────────────────────────────────────────────────────────────────
print("Loading BERT model (all-MiniLM-L6-v2)…")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading spaCy (en_core_web_sm)…")
nlp = spacy.load("en_core_web_sm")
nlp.Defaults.stop_words.add("record")
print("Models ready.\n")


# ─────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS  (identical to evaluate_result_DSL_ToT_DM_ver2)
# ─────────────────────────────────────────────────────────────────

def clean_name(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return s.replace('_', ' ').replace('-', ' ').lower().strip()


def are_synonyms(w1: str, w2: str) -> bool:
    if w1.lower() == w2.lower():
        return True
    for s1 in wordnet.synsets(w1):
        for s2 in wordnet.synsets(w2):
            if s1 == s2:
                return True
    return False


def are_synonyms_phrase(p1: str, p2: str) -> bool:
    def norm(p):
        s = clean_name(p)
        s = re.sub(r'\bnumber\b', 'id', s)
        return [w for w in s.split() if w]
    t1, t2 = norm(p1), norm(p2)
    if not t1 or not t2 or len(t1) != len(t2):
        return False
    return all(are_synonyms(a, b) for a, b in zip(t1, t2))


def char_lcs_score(s1: str, s2: str) -> float:
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    best = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])
    return best / min(m, n)


def strict_word_overlap(s1: str, s2: str) -> bool:
    d1, d2 = nlp(s1), nlp(s2)
    w1 = {t.lemma_ for t in d1 if not (t.is_stop or t.is_punct)}
    w2 = {t.lemma_ for t in d2 if not (t.is_stop or t.is_punct)}
    if not w1 or not w2:
        return False
    if len(w1) == 1 and len(w2) == 1:
        return char_lcs_score(next(iter(w1)), next(iter(w2))) >= 1.0
    exact = len(w1 & w2)
    near  = sum(1 for a in w1 - w2 for b in w2 - w1 if char_lcs_score(a, b) >= 0.85)
    return (exact + near) >= 2


def bert_sim_matrix(list_a: list, list_b: list) -> np.ndarray:
    if not list_a or not list_b:
        return np.zeros((len(list_a), len(list_b)))
    emb_a = bert_model.encode([clean_name(x) for x in list_a], convert_to_tensor=True)
    emb_b = bert_model.encode([clean_name(x) for x in list_b], convert_to_tensor=True)
    return st_util.cos_sim(emb_a, emb_b).cpu().numpy()


def match_step12(list_out: list, list_ref: list) -> dict:
    """Steps 1–2 only: exact + WordNet synonym matching."""
    mapping, used_ref = {}, set()
    for o in list_out:
        for r in list_ref:
            if r not in used_ref and clean_name(o) == clean_name(r):
                mapping[o] = r
                used_ref.add(r)
                break
    for o in [x for x in list_out if x not in mapping]:
        for r in [x for x in list_ref if x not in used_ref]:
            if are_synonyms_phrase(o, r):
                mapping[o] = r
                used_ref.add(r)
                break
    return mapping


def full_matching(list_out: list, list_ref: list,
                  bert_threshold: float,
                  e_sim: np.ndarray = None) -> dict:
    """Full 4-step matching.  e_sim may be pre-computed."""
    if not list_out or not list_ref:
        return {}
    mapping, used_ref = {}, set()

    def rem_out(): return [x for x in list_out if x not in mapping]
    def rem_ref(): return [x for x in list_ref if x not in used_ref]

    # Step 1 – exact
    for o in list_out:
        for r in list_ref:
            if r not in used_ref and clean_name(o) == clean_name(r):
                mapping[o] = r; used_ref.add(r); break

    # Step 2 – WordNet
    for o in rem_out():
        for r in rem_ref():
            if are_synonyms_phrase(o, r):
                mapping[o] = r; used_ref.add(r); break

    # Step 3 – BERT + Hungarian
    ro, rr = rem_out(), rem_ref()
    if ro and rr:
        if e_sim is not None and len(ro) == len(list_out) and len(rr) == len(list_ref):
            mat = e_sim
        else:
            mat = bert_sim_matrix(ro, rr)
        rows, cols = linear_sum_assignment(1 - mat)
        for r_idx, c_idx in zip(rows, cols):
            if mat[r_idx, c_idx] >= bert_threshold:
                o_item, r_item = ro[r_idx], rr[c_idx]
                if o_item not in mapping and r_item not in used_ref:
                    mapping[o_item] = r_item; used_ref.add(r_item)

    # Step 4 – word overlap fallback
    for o in rem_out():
        for r in rem_ref():
            if strict_word_overlap(clean_name(o), clean_name(r)):
                mapping[o] = r; used_ref.add(r); break

    return mapping


def calc_f1(tp: int, total_out: int, total_ref: int) -> float:
    p = tp / total_out if total_out > 0 else 0
    r = tp / total_ref if total_ref > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


# ─────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────

def load_exercise_pairs(opt_folder: str, n_sample: int) -> list:
    """Return list of (out_data, ref_data, exercise_id)."""
    candidates = []
    for f in os.listdir(opt_folder):
        if not f.endswith('.json'):
            continue
        nums = re.findall(r'\d+', f)
        if not nums:
            continue
        ex_id = int(nums[-1])
        ref_path = os.path.join(REF_DIR, f"exercise{ex_id}-baseline.txt")
        if os.path.exists(ref_path):
            candidates.append((os.path.join(opt_folder, f), ref_path, ex_id))

    random.shuffle(candidates)
    candidates = candidates[:n_sample]

    result = []
    for out_path, ref_path, ex_id in candidates:
        try:
            with open(out_path, encoding='utf-8') as fp:
                od = json.load(fp)
            with open(ref_path, encoding='utf-8') as fp:
                rd = json.load(fp)
            result.append((od, rd, ex_id))
        except Exception:
            pass
    return result


# ─────────────────────────────────────────────────────────────────
# PRE-COMPUTE BERT MATRICES (used by sensitivity analysis)
# ─────────────────────────────────────────────────────────────────

def precompute_exercise(od: dict, rd: dict) -> dict:
    """
    Pre-compute all BERT similarity matrices for one exercise so they can be
    reused across multiple threshold evaluations.
    """
    out_ents = [e for e in od.get('entity', [])
                if not str(e).upper().startswith('ASSOC_')]
    ref_ents = rd.get('entity', [])

    e_sim = bert_sim_matrix(out_ents, ref_ents) if (out_ents and ref_ents) else None

    # Attribute matrices for EVERY (out_ent, ref_ent) pair – threshold-agnostic
    out_attr = od.get('attribut', od.get('attribute', {}))
    ref_attr = rd.get('attribut', rd.get('attribute', {}))
    attr_sims = {}
    for oe in out_ents:
        oa = out_attr.get(oe, [])
        if not oa:
            continue
        for re_ in ref_ents:
            ra = ref_attr.get(re_, [])
            if ra:
                attr_sims[(oe, re_)] = bert_sim_matrix(oa, ra)

    return {
        'od': od, 'rd': rd,
        'out_ents': out_ents, 'ref_ents': ref_ents,
        'e_sim': e_sim,
        'attr_sims': attr_sims,
    }


# ─────────────────────────────────────────────────────────────────
# ANALYSIS A – SCORE DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────

def collect_score_distributions(pairs: list) -> tuple:
    """
    Collect BERT cosine scores for the BERT-specific step only
    (i.e., items that were NOT resolved by exact/WordNet matching).

    Returns:
        true_scores  – dict{'entity': [...], 'attribute': [...]}
        false_scores – dict{'entity': [...], 'attribute': [...]}
    """
    true_scores  = defaultdict(list)
    false_scores = defaultdict(list)

    for idx, (od, rd, ex_id) in enumerate(pairs):
        out_ents = [e for e in od.get('entity', [])
                    if not str(e).upper().startswith('ASSOC_')]
        ref_ents = rd.get('entity', [])

        # ── Entities: remove step1+2 matches, analyse BERT residuals ──
        e_map12 = match_step12(out_ents, ref_ents)
        rem_out_e = [e for e in out_ents if e not in e_map12]
        rem_ref_e = [e for e in ref_ents if e not in e_map12.values()]

        if rem_out_e and rem_ref_e:
            mat = bert_sim_matrix(rem_out_e, rem_ref_e)
            rows, cols = linear_sum_assignment(1 - mat)
            assigned = set()
            for r, c in zip(rows, cols):
                true_scores['entity'].append(float(mat[r, c]))
                assigned.add((r, c))
            for r in range(mat.shape[0]):
                for c in range(mat.shape[1]):
                    if (r, c) not in assigned:
                        false_scores['entity'].append(float(mat[r, c]))

        # ── Attributes: use step1+2 entity mapping, analyse BERT residuals ──
        out_attr = od.get('attribut', od.get('attribute', {}))
        ref_attr = rd.get('attribut', rd.get('attribute', {}))
        # Use full entity map for attribute pairing (all 4 steps on entities)
        e_map_full = dict(e_map12)  # start from step1+2
        for oe in out_ents:
            if oe in e_map_full:
                continue
            for re_ in ref_ents:
                if re_ not in e_map_full.values():
                    # Quick BERT check to extend mapping for attribute analysis
                    pass
            # Just use step1+2 for cleanliness in this analysis

        for oe, re_ in e_map12.items():
            oa = out_attr.get(oe, [])
            ra = ref_attr.get(re_, [])
            if not oa or not ra:
                continue
            a_map12 = match_step12(oa, ra)
            rem_oa = [a for a in oa if a not in a_map12]
            rem_ra = [a for a in ra if a not in a_map12.values()]
            if rem_oa and rem_ra:
                mat = bert_sim_matrix(rem_oa, rem_ra)
                rows, cols = linear_sum_assignment(1 - mat)
                assigned = set()
                for r, c in zip(rows, cols):
                    true_scores['attribute'].append(float(mat[r, c]))
                    assigned.add((r, c))
                for r in range(mat.shape[0]):
                    for c in range(mat.shape[1]):
                        if (r, c) not in assigned:
                            false_scores['attribute'].append(float(mat[r, c]))

        if (idx + 1) % 20 == 0:
            print(f"  Distribution: processed {idx+1}/{len(pairs)} exercises…")

    return dict(true_scores), dict(false_scores)


def plot_distributions(true_scores: dict, false_scores: dict,
                       threshold: float = 0.50) -> str:
    concepts = [('entity', 'Entity Names'), ('attribute', 'Attribute Names')]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        'BERT Cosine Similarity: True Pairs vs. False Pairs\n'
        '(BERT-zone only — after Exact and WordNet matching)',
        fontsize=12, fontweight='bold', y=1.02
    )

    for ax, (concept, label) in zip(axes, concepts):
        t = np.array(true_scores.get(concept, []))
        f = np.array(false_scores.get(concept, []))
        if len(t) == 0 or len(f) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        bins = np.linspace(0, 1, 41)
        ax.hist(f, bins=bins, density=True, alpha=0.42,
                color='#EF5350', label=f'False pairs  n={len(f):,}',
                edgecolor='white', linewidth=0.3)
        ax.hist(t, bins=bins, density=True, alpha=0.52,
                color='#1E88E5', label=f'True pairs   n={len(t):,}',
                edgecolor='white', linewidth=0.3)

        xs = np.linspace(0, 1, 300)
        if len(t) > 5:
            ax.plot(xs, sp_stats.gaussian_kde(t, bw_method=0.15)(xs),
                    color='#1565C0', lw=2.0)
        if len(f) > 5:
            ax.plot(xs, sp_stats.gaussian_kde(f, bw_method=0.15)(xs),
                    color='#C62828', lw=2.0)

        ax.axvline(threshold, color='#FF8F00', lw=2.2, linestyle='--',
                   label=f'Threshold = {threshold}', zorder=6)

        # Percentile markers
        for pct, style, alpha in [(10, ':', 0.75), (50, '-.', 0.75)]:
            val = np.percentile(t, pct)
            ax.axvline(val, color='#1E88E5', lw=1.1, linestyle=style, alpha=alpha)
            ax.text(val + 0.012, ax.get_ylim()[1] * 0.85,
                    f'P{pct}={val:.2f}', color='#1565C0',
                    fontsize=7.5, rotation=90, va='top')

        fp90 = np.percentile(f, 90)
        ax.axvline(fp90, color='#C62828', lw=1.1, linestyle=':', alpha=0.75)
        ax.text(fp90 + 0.012, ax.get_ylim()[1] * 0.72,
                f'P90={fp90:.2f}', color='#C62828',
                fontsize=7.5, rotation=90, va='top')

        ks, ksp = sp_stats.ks_2samp(t, f)
        recall_at_thr = 100 * np.mean(t >= threshold)
        precision_at_thr = 100 * np.mean(f < threshold)

        stats_text = (
            f"True pairs:\n"
            f"  μ={np.mean(t):.3f}  σ={np.std(t):.3f}\n"
            f"  P10={np.percentile(t,10):.3f}  P50={np.percentile(t,50):.3f}\n"
            f"\nFalse pairs:\n"
            f"  μ={np.mean(f):.3f}  σ={np.std(f):.3f}\n"
            f"  P90={np.percentile(f,90):.3f}\n"
            f"\nKS D={ks:.3f}  p<{ksp:.1e}\n"
            f"Recall@{threshold}:  {recall_at_thr:.1f}%\n"
            f"Rej-rate@{threshold}: {precision_at_thr:.1f}%"
        )
        ax.text(0.01, 0.99, stats_text,
                transform=ax.transAxes, fontsize=7.5,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.55))

        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('BERT Cosine Similarity', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(0.30, 0.99))
        ax.grid(True, alpha=0.30)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'bert_score_distributions.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────
# ANALYSIS B – THRESHOLD SENSITIVITY (with pre-computed BERT matrices)
# ─────────────────────────────────────────────────────────────────

def evaluate_one_exercise_cached(cache: dict, threshold: float) -> dict:
    """Evaluate one exercise using pre-computed BERT matrices."""
    od, rd     = cache['od'], cache['rd']
    out_ents   = cache['out_ents']
    ref_ents   = cache['ref_ents']
    e_sim      = cache['e_sim']
    attr_sims  = cache['attr_sims']
    out_attr   = od.get('attribut', od.get('attribute', {}))
    ref_attr   = rd.get('attribut', rd.get('attribute', {}))

    if not out_ents or not ref_ents:
        return None

    # ── Entity matching (re-run steps 1-4, reuse e_sim for step 3) ──
    e_map = {}
    used_ref = set()

    def rem_o(): return [e for e in out_ents if e not in e_map]
    def rem_r(): return [e for e in ref_ents if e not in used_ref]

    for o in out_ents:
        for r in ref_ents:
            if r not in used_ref and clean_name(o) == clean_name(r):
                e_map[o] = r; used_ref.add(r); break
    for o in rem_o():
        for r in rem_r():
            if are_synonyms_phrase(o, r):
                e_map[o] = r; used_ref.add(r); break

    ro, rr = rem_o(), rem_r()
    if ro and rr and e_sim is not None:
        # Build sub-matrix for remaining items
        ri_out = [out_ents.index(x) for x in ro]
        ri_ref = [ref_ents.index(x) for x in rr]
        sub = e_sim[np.ix_(ri_out, ri_ref)]
        rows, cols = linear_sum_assignment(1 - sub)
        for r_idx, c_idx in zip(rows, cols):
            if sub[r_idx, c_idx] >= threshold:
                o_item, r_item = ro[r_idx], rr[c_idx]
                if o_item not in e_map and r_item not in used_ref:
                    e_map[o_item] = r_item; used_ref.add(r_item)
    for o in rem_o():
        for r in rem_r():
            if strict_word_overlap(clean_name(o), clean_name(r)):
                e_map[o] = r; used_ref.add(r); break

    ent_f1 = calc_f1(len(e_map), len(out_ents), len(ref_ents))

    # ── Attribute matching ──
    tp_a = tot_oa = tot_ra = 0
    for oe, re_ in e_map.items():
        oa = out_attr.get(oe, [])
        ra = ref_attr.get(re_, [])
        if not oa or not ra:
            tot_oa += len(oa); tot_ra += len(ra); continue

        pre_sim = attr_sims.get((oe, re_))  # may be None if sizes changed

        # Run full 4-step attribute matching with pre-computed sim if available
        a_map = {}
        a_used = set()

        def ra_o(): return [a for a in oa if a not in a_map]
        def ra_r(): return [a for a in ra if a not in a_used]

        for ao in oa:
            for ar in ra:
                if ar not in a_used and clean_name(ao) == clean_name(ar):
                    a_map[ao] = ar; a_used.add(ar); break
        for ao in ra_o():
            for ar in ra_r():
                if are_synonyms_phrase(ao, ar):
                    a_map[ao] = ar; a_used.add(ar); break

        rao, rar = ra_o(), ra_r()
        if rao and rar:
            if pre_sim is not None and len(rao) == len(oa) and len(rar) == len(ra):
                mat = pre_sim
            else:
                mat = bert_sim_matrix(rao, rar)
            rows2, cols2 = linear_sum_assignment(1 - mat)
            for r2, c2 in zip(rows2, cols2):
                if mat[r2, c2] >= threshold:
                    ao_item, ar_item = rao[r2], rar[c2]
                    if ao_item not in a_map and ar_item not in a_used:
                        a_map[ao_item] = ar_item; a_used.add(ar_item)
        for ao in ra_o():
            for ar in ra_r():
                if strict_word_overlap(clean_name(ao), clean_name(ar)):
                    a_map[ao] = ar; a_used.add(ar); break

        tp_a += len(a_map); tot_oa += len(oa); tot_ra += len(ra)

    attr_f1 = calc_f1(tp_a, tot_oa, tot_ra)

    # ── Relation matching ──
    e_map_keys = set(e_map.keys())

    def get_edges(data, valid_ents, mapping):
        edges = set()
        for rel in data.get('relationship', []):
            e1, e2 = rel.get('entity_1'), rel.get('entity_2')
            if valid_ents and (e1 not in valid_ents or e2 not in valid_ents):
                continue
            me1 = mapping.get(e1, e1) if mapping else e1
            me2 = mapping.get(e2, e2) if mapping else e2
            if me1 and me2:
                edges.add(tuple(sorted((str(me1), str(me2)))))
        return edges

    out_rel = get_edges(od, e_map_keys, e_map)
    ref_rel = get_edges(rd, set(ref_ents), None)
    total_out_rel = sum(
        1 for r in od.get('relationship', [])
        if r.get('entity_1') in e_map_keys and r.get('entity_2') in e_map_keys
    )
    rel_f1 = calc_f1(len(out_rel & ref_rel), total_out_rel, len(ref_rel))

    return {'entity': ent_f1, 'attribute': attr_f1, 'relation': rel_f1}


def run_sensitivity_analysis(caches: list, thresholds: np.ndarray) -> dict:
    results = {}
    n = len(caches)
    print(f"\nSensitivity analysis: {n} exercises × {len(thresholds)} thresholds")
    for thr in thresholds:
        t0 = time.time()
        e_f1s, a_f1s, r_f1s = [], [], []
        for cache in caches:
            m = evaluate_one_exercise_cached(cache, float(thr))
            if m is None:
                continue
            e_f1s.append(m['entity'])
            a_f1s.append(m['attribute'])
            r_f1s.append(m['relation'])
        ent  = float(np.mean(e_f1s))  if e_f1s  else 0.0
        attr = float(np.mean(a_f1s))  if a_f1s  else 0.0
        rel  = float(np.mean(r_f1s))  if r_f1s  else 0.0
        avg  = (ent + attr + rel) / 3
        results[float(thr)] = {'entity': ent, 'attribute': attr,
                                'relation': rel, 'average': avg}
        mark = ' ←' if abs(thr - CHOSEN_THRESHOLD) < 1e-9 else ''
        print(f"  θ={thr:.2f}  Ent={ent:.4f}  Attr={attr:.4f}  "
              f"Rel={rel:.4f}  Avg={avg:.4f}  ({time.time()-t0:.1f}s){mark}")
    return results


def plot_sensitivity(results: dict, thresholds: np.ndarray,
                     chosen: float = 0.50) -> str:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    palette = {
        'entity':    ('#1565C0', '-',  'o',  1.8),
        'attribute': ('#2E7D32', '-',  's',  1.8),
        'relation':  ('#C62828', '-',  '^',  1.8),
        'average':   ('#6A1B9A', '--', 'D',  2.4),
    }
    labels = {
        'entity':    'Entity F1',
        'attribute': 'Attribute F1',
        'relation':  'Relation F1',
        'average':   'Average F1',
    }
    for metric, (color, ls, marker, lw) in palette.items():
        ys = [results[float(t)][metric] for t in thresholds]
        ax.plot(thresholds, ys, color=color, ls=ls, lw=lw,
                marker=marker, markersize=5, label=labels[metric])

    ax.axvline(chosen, color='#FF8F00', lw=2.2, ls='--',
               label=f'Selected θ = {chosen}', zorder=5)

    # Annotate best average
    avg_ys = [results[float(t)]['average'] for t in thresholds]
    best_idx  = int(np.argmax(avg_ys))
    best_thr  = float(thresholds[best_idx])
    best_val  = avg_ys[best_idx]
    chosen_avg = results[chosen]['average']
    ax.annotate(
        f'Best avg F1\nθ={best_thr:.2f} → {best_val:.4f}',
        xy=(best_thr, best_val),
        xytext=(best_thr + 0.07, best_val - 0.04),
        fontsize=8.5, color='#6A1B9A',
        arrowprops=dict(arrowstyle='->', color='#6A1B9A', lw=1.4)
    )
    ax.annotate(
        f'Selected\nθ={chosen} → {chosen_avg:.4f}',
        xy=(chosen, chosen_avg),
        xytext=(chosen - 0.13, chosen_avg + 0.03),
        fontsize=8.5, color='#FF8F00',
        arrowprops=dict(arrowstyle='->', color='#FF8F00', lw=1.4)
    )

    ax.set_xlabel('BERT Cosine Similarity Threshold (θ)', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('F1 Score vs. BERT Similarity Threshold\n'
                 '(Entity / Attribute / Relation Matching)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(thresholds[0] - 0.025, thresholds[-1] + 0.025)
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9.5, loc='lower right')
    ax.grid(True, alpha=0.32)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'threshold_sensitivity.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────
# TEXT REPORT  (copy-paste ready for paper)
# ─────────────────────────────────────────────────────────────────

def write_report(true_scores: dict, false_scores: dict,
                 sens_results: dict, thresholds: np.ndarray,
                 chosen: float = 0.50) -> str:
    lines = []
    W = 68

    def h(title): lines.extend(['', '=' * W, title, '=' * W])
    def ln(s=''): lines.append(s)

    h("BERT THRESHOLD ANALYSIS REPORT")
    ln(f"Generated   : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    ln(f"Model       : all-MiniLM-L6-v2")
    ln(f"Chosen θ    : {chosen}")
    ln(f"Dist. sample: {DIST_SAMPLE} exercises")
    ln(f"Sens. sample: {SENS_SAMPLE} exercises  ×  {len(thresholds)} thresholds")

    h("A. SCORE DISTRIBUTIONS  (BERT-zone: after Exact + WordNet)")
    for concept in ['entity', 'attribute']:
        t = np.array(true_scores.get(concept, []))
        f = np.array(false_scores.get(concept, []))
        if len(t) == 0 or len(f) == 0:
            ln(f"\n  [{concept.upper()}]  No data."); continue
        ks, ksp = sp_stats.ks_2samp(t, f)
        recall = 100 * np.mean(t >= chosen)
        rej    = 100 * np.mean(f < chosen)
        fp90   = np.percentile(f, 90)
        tp10   = np.percentile(t, 10)
        ln(f"\n  [{concept.upper()}]  (true n={len(t):,}  |  false n={len(f):,})")
        ln(f"  True pairs:")
        ln(f"    mean={np.mean(t):.4f}  std={np.std(t):.4f}  "
           f"min={np.min(t):.4f}  max={np.max(t):.4f}")
        ln(f"    P10={np.percentile(t,10):.4f}  P25={np.percentile(t,25):.4f}  "
           f"P50={np.percentile(t,50):.4f}  P75={np.percentile(t,75):.4f}  "
           f"P90={np.percentile(t,90):.4f}")
        ln(f"  False pairs:")
        ln(f"    mean={np.mean(f):.4f}  std={np.std(f):.4f}  "
           f"min={np.min(f):.4f}  max={np.max(f):.4f}")
        ln(f"    P10={np.percentile(f,10):.4f}  P25={np.percentile(f,25):.4f}  "
           f"P50={np.percentile(f,50):.4f}  P75={np.percentile(f,75):.4f}  "
           f"P90={fp90:.4f}")
        ln(f"  KS-statistic = {ks:.4f}  p-value = {ksp:.2e}"
           f"  ({'well-separated' if ks >= 0.4 else 'partially overlapping'})")
        ln(f"  At θ={chosen}:")
        ln(f"    True-pair recall    = {recall:.1f}%  (% of true pairs ABOVE θ)")
        ln(f"    False-pair rej rate = {rej:.1f}%  (% of false pairs BELOW θ)")
        ln(f"    Gap: false-P90={fp90:.4f}  <  θ={chosen}  <  true-P10={tp10:.4f} "
           f"{'✓' if fp90 < chosen < tp10 else '△ partially in gap'}")

    h("B. THRESHOLD SENSITIVITY  –  F1 vs. θ")
    ln(f"\n  {'θ':>6}  {'Ent F1':>8}  {'Attr F1':>8}  {'Rel F1':>8}  {'Avg F1':>8}")
    ln('  ' + '-' * 46)
    for thr in thresholds:
        r = sens_results[float(thr)]
        mark = '  ← selected' if abs(thr - chosen) < 1e-9 else ''
        ln(f"  {thr:>5.2f}   {r['entity']:>7.4f}   {r['attribute']:>7.4f}"
           f"   {r['relation']:>7.4f}   {r['average']:>7.4f}{mark}")

    ln()
    for metric in ['entity', 'attribute', 'relation', 'average']:
        vals = {float(t): sens_results[float(t)][metric] for t in thresholds}
        best_thr = max(vals, key=vals.get)
        best_val = vals[best_thr]
        sel_val  = sens_results[chosen][metric]
        delta    = sel_val - best_val
        ln(f"  Best {metric:10s}: θ={best_thr:.2f} → {best_val:.4f} | "
           f"selected θ={chosen} → {sel_val:.4f} | Δ={delta:+.4f}")

    h("C. PAPER-READY SUMMARY")
    r_c = sens_results[chosen]
    ln(f"\n  At BERT threshold θ = {chosen}:")
    ln(f"    Entity F1    = {r_c['entity']:.4f}")
    ln(f"    Attribute F1 = {r_c['attribute']:.4f}")
    ln(f"    Relation F1  = {r_c['relation']:.4f}")
    ln(f"    Average F1   = {r_c['average']:.4f}")
    ln()
    t_e = np.array(true_scores.get('entity', []))
    f_e = np.array(false_scores.get('entity', []))
    if len(t_e) and len(f_e):
        ks, ksp = sp_stats.ks_2samp(t_e, f_e)
        ln("  Evidence (entity channel):")
        ln(f"    1. True-pair  P10 = {np.percentile(t_e,10):.3f}  → "
           f"{100*np.mean(t_e>=chosen):.1f}% of true pairs lie above θ={chosen}")
        ln(f"    2. False-pair P90 = {np.percentile(f_e,90):.3f}  → "
           f"{100*np.mean(f_e<chosen):.1f}% of false pairs lie below θ={chosen}")
        ln(f"    3. KS D={ks:.3f}, p={ksp:.2e}  → distributions are statistically distinct")
        avg_ys = [sens_results[float(t)]['average'] for t in thresholds]
        best_avg_thr = float(thresholds[int(np.argmax(avg_ys))])
        ln(f"    4. Optimal average F1 occurs at θ={best_avg_thr:.2f};  "
           f"θ={chosen} is within Δ={sens_results[chosen]['average']-max(avg_ys):+.4f}")

    report_text = '\n'.join(lines)
    out_path = os.path.join(OUTPUT_DIR, 'bert_threshold_report.txt')
    with open(out_path, 'w', encoding='utf-8') as fp:
        fp.write(report_text)
    print(f"Saved: {out_path}")
    return report_text


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_start = time.time()

    # Resolve opt folder (may be flat or single level)
    opt_folder = OPT_DIR
    if not os.path.isdir(opt_folder):
        sys.exit(f"ERROR: OPT_DIR not found: {opt_folder}")

    # If JSON files are nested one level deeper
    if not any(f.endswith('.json') for f in os.listdir(opt_folder)):
        for sub in sorted(os.listdir(opt_folder)):
            sp = os.path.join(opt_folder, sub)
            if os.path.isdir(sp) and any(f.endswith('.json') for f in os.listdir(sp)):
                opt_folder = sp
                break

    print(f"Optimization folder : {opt_folder}")
    print(f"Reference folder    : {REF_DIR}")
    print(f"Output directory    : {OUTPUT_DIR}\n")

    # ── Load data ──────────────────────────────────────────────
    print(f"Loading up to {max(DIST_SAMPLE, SENS_SAMPLE)} exercise pairs…")
    all_pairs = load_exercise_pairs(opt_folder, max(DIST_SAMPLE, SENS_SAMPLE) + 20)
    random.shuffle(all_pairs)
    dist_pairs = all_pairs[:DIST_SAMPLE]
    sens_pairs_raw = all_pairs[:SENS_SAMPLE]
    print(f"  Loaded {len(all_pairs)} total  |  "
          f"distribution={len(dist_pairs)}  sensitivity={len(sens_pairs_raw)}\n")

    # ── Analysis A: Score Distributions ────────────────────────
    print('=' * 68)
    print('ANALYSIS A: BERT Score Distributions')
    print('=' * 68)
    true_scores, false_scores = collect_score_distributions(dist_pairs)
    for concept in ['entity', 'attribute']:
        t = true_scores.get(concept, [])
        f = false_scores.get(concept, [])
        print(f"  {concept:10s}: true_pairs={len(t):5d}  false_pairs={len(f):6d}")
    plot_distributions(true_scores, false_scores, threshold=CHOSEN_THRESHOLD)

    # ── Pre-compute BERT matrices for sensitivity analysis ──────
    print(f"\nPre-computing BERT matrices for {len(sens_pairs_raw)} exercises…")
    t_pre = time.time()
    caches = []
    for i, (od, rd, ex_id) in enumerate(sens_pairs_raw):
        caches.append(precompute_exercise(od, rd))
        if (i + 1) % 20 == 0:
            print(f"  Pre-computed {i+1}/{len(sens_pairs_raw)}…")
    print(f"  Done in {time.time()-t_pre:.1f}s\n")

    # ── Analysis B: Threshold Sensitivity ──────────────────────
    print('=' * 68)
    print('ANALYSIS B: F1 Threshold Sensitivity')
    print('=' * 68)
    sens_results = run_sensitivity_analysis(caches, THRESHOLDS)
    plot_sensitivity(sens_results, THRESHOLDS, chosen=CHOSEN_THRESHOLD)

    # ── Report ──────────────────────────────────────────────────
    report = write_report(true_scores, false_scores,
                          sens_results, THRESHOLDS, chosen=CHOSEN_THRESHOLD)

    print('\n' + '=' * 68)
    print(f'ALL DONE  –  total elapsed: {(time.time()-t_start)/60:.1f} min')
    print(f'Outputs saved to: {OUTPUT_DIR}')
    print('=' * 68)
    print('\nFiles generated:')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        print(f'  {f}  ({os.path.getsize(fp)//1024} KB)')
