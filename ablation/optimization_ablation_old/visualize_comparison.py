"""
3-Way Ablation Comparison Visualisation
=========================================
Compares three conditions across all 10 generation variants:

    (A) Original   — full system WITH Wikidata, complex ILP
    (B) Ablation   — WITHOUT Wikidata, complex ILP (lambda_isolated + lambda_noattr)
    (C) Simple ILP — WITHOUT Wikidata, simple ILP (structural constraints only)

Figures saved to: ablation/optimization_ablation_old/plots/
    1. overall_3way.png        — Overall F1 grouped bar (3 bars per variant)
    2. metric_3way.png         — Entity / Attr / Rel F1 side-by-side
    3. delta_vs_original.png   — Δ vs Original for both ablation conditions
    4. heatmap_3way.png        — Full metric table as coloured heatmap

Usage:
    python ablation/optimization_ablation_old/visualize_comparison.py
"""

import os, csv, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
_SELF_DIR    = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_SELF_DIR, '..', '..'))

SIMPLE_CSV   = os.path.join(_SELF_DIR, 'results', 'summary.csv')
COMPLEX_CSV  = os.path.join(_SELF_DIR, '..', 'Evaluation_Ablation', 'results', 'summary.csv')
ORIGINAL_DIR = os.path.join(project_root, 'Output_final', 'results', 'F1Score_2')
PLOTS_DIR    = os.path.join(_SELF_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Colors ─────────────────────────────────────────────────────────────────────
C_ORIG    = '#1565C0'   # dark blue  — original (with Wikidata)
C_COMPLEX = '#FF6F00'   # dark orange — ablation complex ILP
C_SIMPLE  = '#2E7D32'   # dark green  — ablation simple ILP

# ── Original result files mapping ─────────────────────────────────────────────
ORIG_FILES = {
    "multi-llms-few-shot-gpt":         os.path.join(ORIGINAL_DIR, "Multi-LLMs-withBN", "opt_fewshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
    "multi-llms-few-shot-llama":       os.path.join(ORIGINAL_DIR, "Multi-LLMs-withBN", "opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0).csv"),
    "multi-llms-zero-shot-gpt":        os.path.join(ORIGINAL_DIR, "Multi-LLMs-withBN", "opt_zeroshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
    "multi-llms-zero-shot-llama":      os.path.join(ORIGINAL_DIR, "Multi-LLMs-withBN", "opt_zeroshot_llama_0.5_1.0-(1.2-1.0-1.0).csv"),
    "one-llm-one_llm_few_shot_gpt":    os.path.join(ORIGINAL_DIR, "One-LLMs-withBN", "opt_one_fewshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
    "one-llm-one_llm_few_shot_llama":  os.path.join(ORIGINAL_DIR, "One-LLMs-withBN", "opt_one_fewshot_llama_0.5_1.0-(1.2-1.0-1.0).csv"),
    "one-llm-one_llm_zero_shot_gpt":   os.path.join(ORIGINAL_DIR, "One-LLMs-withBN", "opt_one_zeroshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
    "one-llm-one_llm_zero_shot_llama": os.path.join(ORIGINAL_DIR, "One-LLMs-withBN", "opt_one_zeroshot_llama_0.5_1.0-(1.2-1.0-1.0).csv"),
    "ToT-gpt":                         os.path.join(ORIGINAL_DIR, "ToT-withBN", "opt_ToT_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
    "ToT-llama":                       os.path.join(ORIGINAL_DIR, "ToT-withBN", "opt_ToT_llama_0.5_1.0-(1.2-1.0-1.0).csv"),
}

DISPLAY_LABELS = {
    "multi-llms-few-shot-gpt":         "Multi-LLMs\nFew-GPT",
    "multi-llms-few-shot-llama":       "Multi-LLMs\nFew-LLaMA",
    "multi-llms-zero-shot-gpt":        "Multi-LLMs\nZero-GPT",
    "multi-llms-zero-shot-llama":      "Multi-LLMs\nZero-LLaMA",
    "one-llm-one_llm_few_shot_gpt":    "One-LLM\nFew-GPT",
    "one-llm-one_llm_few_shot_llama":  "One-LLM\nFew-LLaMA",
    "one-llm-one_llm_zero_shot_gpt":   "One-LLM\nZero-GPT",
    "one-llm-one_llm_zero_shot_llama": "One-LLM\nZero-LLaMA",
    "ToT-gpt":                         "ToT\nGPT",
    "ToT-llama":                       "ToT\nLLaMA",
}

GROUPS = {
    "multi-llms-few-shot-gpt":         "Multi-LLMs",
    "multi-llms-few-shot-llama":       "Multi-LLMs",
    "multi-llms-zero-shot-gpt":        "Multi-LLMs",
    "multi-llms-zero-shot-llama":      "Multi-LLMs",
    "one-llm-one_llm_few_shot_gpt":    "One-LLM",
    "one-llm-one_llm_few_shot_llama":  "One-LLM",
    "one-llm-one_llm_zero_shot_gpt":   "One-LLM",
    "one-llm-one_llm_zero_shot_llama": "One-LLM",
    "ToT-gpt":                         "ToT",
    "ToT-llama":                       "ToT",
}

ORDER = [
    "multi-llms-few-shot-gpt", "multi-llms-few-shot-llama",
    "multi-llms-zero-shot-gpt", "multi-llms-zero-shot-llama",
    "one-llm-one_llm_few_shot_gpt", "one-llm-one_llm_few_shot_llama",
    "one-llm-one_llm_zero_shot_gpt", "one-llm-one_llm_zero_shot_llama",
    "ToT-gpt", "ToT-llama",
]

# ── Data loading ───────────────────────────────────────────────────────────────

def _read_avg(csv_path):
    """Read AVERAGE row from a standard Ent/Attr/Rel CSV."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('Exercise', row.get('Variant', '')).strip().upper() == 'AVERAGE':
                    return {k: float(v) for k, v in row.items()
                            if k not in ('Exercise', 'Variant', 'N')}
    except Exception as e:
        print(f"  [WARN] {csv_path}: {e}")
    return None

def _read_summary(csv_path):
    """Read a summary.csv into {variant: {Ent_F1, Attr_F1, Rel_F1, Overall_F1}}."""
    result = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = row['Variant']
                result[key] = {
                    'Ent_F1':     float(row['Ent_F1']),
                    'Attr_F1':    float(row['Attr_F1']),
                    'Rel_F1':     float(row['Rel_F1']),
                    'Overall_F1': float(row['Overall_F1']),
                }
    except Exception as e:
        print(f"  [WARN] {csv_path}: {e}")
    return result

def load_all():
    simple_map  = _read_summary(SIMPLE_CSV)
    complex_map = _read_summary(COMPLEX_CSV)

    data = {}
    for key in ORDER:
        # Original
        orig_row = _read_avg(ORIG_FILES.get(key, ''))
        if orig_row:
            f1e = orig_row.get('Ent_F1', 0)
            f1a = orig_row.get('Attr_F1', 0)
            f1r = orig_row.get('Rel_F1', 0)
            orig = {'Ent_F1': f1e, 'Attr_F1': f1a, 'Rel_F1': f1r,
                    'Overall_F1': (f1e + f1a + f1r) / 3}
        else:
            orig = None

        simple  = simple_map.get(key)
        complex_ = complex_map.get(key)

        if orig and simple and complex_:
            data[key] = {'orig': orig, 'simple': simple, 'complex': complex_}
        else:
            print(f"  [SKIP] {key}: missing data")
    return data


# ── Style helpers ──────────────────────────────────────────────────────────────

def _style(ax, title, ylabel='F1 Score', ylim=(0.3, 0.95)):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=7)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def _dividers(ax, keys):
    prev = None
    for i, k in enumerate(keys):
        g = GROUPS[k]
        if prev and g != prev:
            ax.axvline(x=i - 0.5, color='gray', lw=0.8, ls=':', alpha=0.5)
        prev = g

def _bar_labels(ax, bars, fontsize=7):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=fontsize, color='#333')

def _group_headers(ax, keys, y=0.91):
    gpos = {}
    for i, k in enumerate(keys):
        gpos.setdefault(GROUPS[k], []).append(i)
    for g, idxs in gpos.items():
        ax.text(np.mean(idxs), y, g, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#444',
                bbox=dict(boxstyle='round,pad=0.2', fc='#eee', ec='none', alpha=0.7),
                transform=ax.get_xaxis_transform())


# ── Figure 1: Overall F1 – 3 conditions ───────────────────────────────────────

def plot_overall(data):
    keys   = list(data.keys())
    labels = [DISPLAY_LABELS[k] for k in keys]
    n = len(keys); x = np.arange(n); w = 0.26

    fig, ax = plt.subplots(figsize=(15, 6))
    b1 = ax.bar(x - w,   [data[k]['orig']['Overall_F1']    for k in keys], w, label='(A) Original + Wikidata',         color=C_ORIG,    alpha=0.87, edgecolor='white')
    b2 = ax.bar(x,       [data[k]['complex']['Overall_F1'] for k in keys], w, label='(B) Ablation + Complex ILP',      color=C_COMPLEX, alpha=0.87, edgecolor='white')
    b3 = ax.bar(x + w,   [data[k]['simple']['Overall_F1']  for k in keys], w, label='(C) Ablation + Simple ILP',       color=C_SIMPLE,  alpha=0.87, edgecolor='white')

    for bars in (b1, b2, b3):
        _bar_labels(ax, bars, fontsize=7)

    _dividers(ax, keys)
    _group_headers(ax, keys)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    _style(ax, 'Overall F1 Score — 3-Way Comparison\n'
               '(A) With Wikidata  |  (B) No Wikidata + λ_isolated  |  (C) No Wikidata + Simple ILP')
    ax.legend(fontsize=9, framealpha=0.9, loc='lower right')

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'overall_3way.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"  Saved: {path}")


# ── Figure 2: Per-metric breakdown ────────────────────────────────────────────

def plot_metric_breakdown(data):
    keys   = list(data.keys())
    labels = [DISPLAY_LABELS[k] for k in keys]
    n = len(keys); x = np.arange(n); w = 0.26

    metrics = [('Ent_F1', 'Entity F1'), ('Attr_F1', 'Attribute F1'), ('Rel_F1', 'Relationship F1')]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), sharey=False)
    fig.suptitle('Per-Metric F1 — (A) Original  vs  (B) No Wikidata Complex ILP  vs  (C) No Wikidata Simple ILP',
                 fontsize=12, fontweight='bold', y=1.01)

    for ax, (mk, ml) in zip(axes, metrics):
        ax.bar(x - w, [data[k]['orig'][mk]    for k in keys], w, label='(A) Original',    color=C_ORIG,    alpha=0.87, edgecolor='white')
        ax.bar(x,     [data[k]['complex'][mk] for k in keys], w, label='(B) Complex Abl', color=C_COMPLEX, alpha=0.87, edgecolor='white')
        ax.bar(x + w, [data[k]['simple'][mk]  for k in keys], w, label='(C) Simple Abl',  color=C_SIMPLE,  alpha=0.87, edgecolor='white')
        _dividers(ax, keys)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        _style(ax, ml, ylim=(0.0, 1.0))
        ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'metric_3way.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"  Saved: {path}")


# ── Figure 3: Delta vs Original ───────────────────────────────────────────────

def plot_delta(data):
    keys   = list(data.keys())
    labels = [DISPLAY_LABELS[k] for k in keys]
    n = len(keys); x = np.arange(n); w = 0.35

    metrics   = ['Ent_F1', 'Attr_F1', 'Rel_F1', 'Overall_F1']
    met_label = ['Entity', 'Attribute', 'Relationship', 'Overall']
    colors    = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    fig.suptitle('Δ F1 vs Original (With Wikidata)\n'
                 'Left: (B) No Wikidata + Complex ILP  |  Right: (C) No Wikidata + Simple ILP',
                 fontsize=12, fontweight='bold')

    for ax, cond_key, title in [
        (axes[0], 'complex', '(B) No Wikidata + Complex ILP (λ_isolated, λ_noattr)'),
        (axes[1], 'simple',  '(C) No Wikidata + Simple ILP (structural only)'),
    ]:
        offsets = [-1.5*w/4, -0.5*w/4, 0.5*w/4, 1.5*w/4]
        for mk, ml, off, col in zip(metrics, met_label, offsets, colors):
            deltas = [data[k][cond_key][mk] - data[k]['orig'][mk] for k in keys]
            bar_colors = ['#4CAF50' if d >= 0 else '#F44336' for d in deltas]
            ax.bar(x + off, deltas, w/4, color=bar_colors, alpha=0.8,
                   label=ml, edgecolor='none')

        ax.axhline(0, color='black', lw=0.9)
        _dividers(ax, keys)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
        _style(ax, title, ylabel='Δ F1', ylim=(-0.30, 0.10))

        patches = [mpatches.Patch(color=c, label=l, alpha=0.8)
                   for c, l in zip(colors, met_label)]
        ax.legend(handles=patches, fontsize=8.5, framealpha=0.9, loc='lower right')

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'delta_vs_original.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"  Saved: {path}")


# ── Figure 4: Heatmap 3-way ───────────────────────────────────────────────────

def plot_heatmap(data):
    keys       = list(data.keys())
    short_lbl  = [DISPLAY_LABELS[k].replace('\n', ' ') for k in keys]
    metrics    = ['Ent_F1', 'Attr_F1', 'Rel_F1', 'Overall_F1']
    col_short  = ['Ent', 'Attr', 'Rel', 'Ovrl']

    # Build matrices: rows=variants, cols=metrics
    mat_orig    = np.array([[data[k]['orig'][m]    for m in metrics] for k in keys])
    mat_complex = np.array([[data[k]['complex'][m] for m in metrics] for k in keys])
    mat_simple  = np.array([[data[k]['simple'][m]  for m in metrics] for k in keys])
    mat_delta_c = mat_complex - mat_orig
    mat_delta_s = mat_simple  - mat_orig

    fig, axes = plt.subplots(1, 5, figsize=(22, 5),
                             gridspec_kw={'width_ratios': [4, 4, 4, 4, 4]})
    fig.suptitle('Full Metric Heatmap — 3-Way Comparison (A) Original | (B) Complex Ablation | (C) Simple Ablation',
                 fontsize=12, fontweight='bold')

    panels = [
        (axes[0], mat_orig,    col_short, 'Blues',   '(A) Original\n(With Wikidata)',         0.35, 0.92),
        (axes[1], mat_complex, col_short, 'Oranges', '(B) No Wikidata\nComplex ILP',           0.35, 0.92),
        (axes[2], mat_simple,  col_short, 'Greens',  '(C) No Wikidata\nSimple ILP',            0.35, 0.92),
        (axes[3], mat_delta_c, col_short, 'RdYlGn',  'Δ (B) − (A)\nComplex − Original',      -0.25, 0.10),
        (axes[4], mat_delta_s, col_short, 'RdYlGn',  'Δ (C) − (A)\nSimple − Original',       -0.25, 0.10),
    ]

    for ax, mat, cols, cmap, title, vmin, vmax in panels:
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=9)
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(short_lbl, fontsize=8)
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=5)
        is_delta = title.startswith('Δ')
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                tc = 'white' if (not is_delta and v > 0.78) else 'black'
                pref = '+' if (is_delta and v > 0) else ''
                ax.text(j, i, f'{pref}{v:.3f}', ha='center', va='center',
                        fontsize=7.5, color=tc, fontweight='bold')

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'heatmap_3way.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig); print(f"  Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data = load_all()
    print(f"  {len(data)} variants loaded.\n")

    print("Generating figures...")
    plot_overall(data)
    plot_metric_breakdown(data)
    plot_delta(data)
    plot_heatmap(data)
    print(f"\nAll plots → {PLOTS_DIR}")

    # Summary table
    print(f"\n{'Variant':<38}  {'Orig':>6}  {'Cmplx':>6}  {'Smpl':>6}  {'ΔC':>7}  {'ΔS':>7}")
    print('-' * 75)
    for k in data:
        o = data[k]['orig']['Overall_F1']
        c = data[k]['complex']['Overall_F1']
        s = data[k]['simple']['Overall_F1']
        lbl = DISPLAY_LABELS[k].replace('\n', ' ')
        print(f"  {lbl:<36}  {o:.4f}  {c:.4f}  {s:.4f}  {c-o:+.4f}  {s-o:+.4f}")

    # Averages
    o_avg = np.mean([data[k]['orig']['Overall_F1']    for k in data])
    c_avg = np.mean([data[k]['complex']['Overall_F1'] for k in data])
    s_avg = np.mean([data[k]['simple']['Overall_F1']  for k in data])
    print('-' * 75)
    print(f"  {'AVERAGE':<36}  {o_avg:.4f}  {c_avg:.4f}  {s_avg:.4f}  {c_avg-o_avg:+.4f}  {s_avg-o_avg:+.4f}")


if __name__ == '__main__':
    main()
