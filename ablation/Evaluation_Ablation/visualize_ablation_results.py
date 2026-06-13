"""
Ablation Study — Result Visualisation
======================================
Compares the full system (with Wikidata x_type feature) against the ablation
variant (without Wikidata) across all 10 generation configurations.

Metrics:  Entity F1 / Attribute F1 / Relationship F1 / Overall F1

Output figures (saved to ablation/Evaluation_Ablation/plots/):
  1. overall_f1_comparison.png   — Overall F1: Original vs Ablation (grouped bars)
  2. metric_breakdown.png        — Entity / Attr / Rel F1 side-by-side for both conditions
  3. delta_impact.png            — Δ F1 = Ablation − Original (positive → Wikidata hurts)
  4. heatmap_comparison.png      — Full metric table as coloured heatmap

Usage:
    python ablation/Evaluation_Ablation/visualize_ablation_results.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')           # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths ─────────────────────────────────────────────────────────────────────
_SELF_DIR    = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_SELF_DIR, '..', '..'))

ABLATION_CSV  = os.path.join(_SELF_DIR,  'results', 'summary.csv')
ORIGINAL_DIR  = os.path.join(project_root, 'Output_final', 'results', 'F1Score_2')
PLOTS_DIR     = os.path.join(_SELF_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_ORIG  = '#2196F3'   # blue  — original (with Wikidata)
C_ABL   = '#FF9800'   # orange — ablation (without Wikidata)
C_POS   = '#4CAF50'   # green  — positive delta (Wikidata hurts)
C_NEG   = '#F44336'   # red    — negative delta (Wikidata helps)

# ── Variant metadata ──────────────────────────────────────────────────────────
# Each entry: (ablation_key, original_csv_path, display_label, group)
VARIANTS = [
    (
        "multi-llms-few-shot-gpt",
        os.path.join(ORIGINAL_DIR, "Multi-LLMs-withBN",
                     "opt_fewshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
        "Multi-LLMs\nFew-GPT",
        "Multi-LLMs",
    ),
    (
        "multi-llms-few-shot-llama",
        os.path.join(ORIGINAL_DIR, "Multi-LLMs-withBN",
                     "opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0).csv"),
        "Multi-LLMs\nFew-LLaMA",
        "Multi-LLMs",
    ),
    (
        "multi-llms-zero-shot-gpt",
        os.path.join(ORIGINAL_DIR, "Multi-LLMs-withBN",
                     "opt_zeroshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
        "Multi-LLMs\nZero-GPT",
        "Multi-LLMs",
    ),
    (
        "multi-llms-zero-shot-llama",
        os.path.join(ORIGINAL_DIR, "Multi-LLMs-withBN",
                     "opt_zeroshot_llama_0.5_1.0-(1.2-1.0-1.0).csv"),
        "Multi-LLMs\nZero-LLaMA",
        "Multi-LLMs",
    ),
    (
        "one-llm-one_llm_few_shot_gpt",
        os.path.join(ORIGINAL_DIR, "One-LLMs-withBN",
                     "opt_one_fewshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
        "One-LLM\nFew-GPT",
        "One-LLM",
    ),
    (
        "one-llm-one_llm_few_shot_llama",
        os.path.join(ORIGINAL_DIR, "One-LLMs-withBN",
                     "opt_one_fewshot_llama_0.5_1.0-(1.2-1.0-1.0).csv"),
        "One-LLM\nFew-LLaMA",
        "One-LLM",
    ),
    (
        "one-llm-one_llm_zero_shot_gpt",
        os.path.join(ORIGINAL_DIR, "One-LLMs-withBN",
                     "opt_one_zeroshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
        "One-LLM\nZero-GPT",
        "One-LLM",
    ),
    (
        "one-llm-one_llm_zero_shot_llama",
        os.path.join(ORIGINAL_DIR, "One-LLMs-withBN",
                     "opt_one_zeroshot_llama_0.5_1.0-(1.2-1.0-1.0).csv"),
        "One-LLM\nZero-LLaMA",
        "One-LLM",
    ),
    (
        "ToT-gpt",
        os.path.join(ORIGINAL_DIR, "ToT-withBN",
                     "opt_ToT_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
        "ToT\nGPT",
        "ToT",
    ),
    (
        "ToT-llama",
        os.path.join(ORIGINAL_DIR, "ToT-withBN",
                     "opt_ToT_llama_0.5_1.0-(1.2-1.0-1.0).csv"),
        "ToT\nLLaMA",
        "ToT",
    ),
]

# ── Data loading ──────────────────────────────────────────────────────────────

def _read_average_row(csv_path: str) -> dict | None:
    """Return the AVERAGE row of a standard Ent/Attr/Rel CSV, or None."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('Exercise', '').strip().upper() == 'AVERAGE':
                    return {k: float(v) for k, v in row.items() if k != 'Exercise'}
    except Exception as e:
        print(f"  [WARN] Cannot read {csv_path}: {e}")
    return None


def load_data() -> dict:
    """
    Returns {variant_key: {'orig': {...}, 'abl': {...}}}
    where each inner dict has keys Ent_F1, Attr_F1, Rel_F1, Overall_F1.
    """
    # Load ablation summary
    abl_map: dict = {}
    with open(ABLATION_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = row['Variant']
            abl_map[key] = {
                'Ent_F1':     float(row['Ent_F1']),
                'Attr_F1':    float(row['Attr_F1']),
                'Rel_F1':     float(row['Rel_F1']),
                'Overall_F1': float(row['Overall_F1']),
            }

    data = {}
    for abl_key, orig_csv, label, group in VARIANTS:
        orig_row = _read_average_row(orig_csv)
        if orig_row is None:
            print(f"  [SKIP] {abl_key}: original CSV not found.")
            continue
        abl_row = abl_map.get(abl_key)
        if abl_row is None:
            print(f"  [SKIP] {abl_key}: ablation results not found.")
            continue

        orig_f1e  = orig_row.get('Ent_F1',  orig_row.get('Ent_F1',  0.0))
        orig_f1a  = orig_row.get('Attr_F1', 0.0)
        orig_f1r  = orig_row.get('Rel_F1',  0.0)
        orig_ovrl = (orig_f1e + orig_f1a + orig_f1r) / 3

        data[abl_key] = {
            'label': label,
            'group': group,
            'orig': {
                'Ent_F1':     orig_f1e,
                'Attr_F1':    orig_f1a,
                'Rel_F1':     orig_f1r,
                'Overall_F1': orig_ovrl,
            },
            'abl': abl_row,
        }
    return data


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _ax_style(ax, title: str, ylabel: str = 'F1 Score',
              ylim: tuple = (0, 1), grid: bool = True):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    if grid:
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _group_dividers(ax, groups: list, n: int):
    """Draw faint vertical lines between groups of variants."""
    prev = None
    for i, g in enumerate(groups):
        if prev is not None and g != prev:
            ax.axvline(x=i - 0.5, color='gray', linewidth=0.8,
                       linestyle=':', alpha=0.6)
        prev = g


# ── Figure 1: Overall F1 comparison ──────────────────────────────────────────

def plot_overall_f1(data: dict):
    keys   = list(data.keys())
    labels = [data[k]['label'] for k in keys]
    groups = [data[k]['group'] for k in keys]
    orig   = [data[k]['orig']['Overall_F1'] for k in keys]
    abl    = [data[k]['abl']['Overall_F1']  for k in keys]

    n     = len(keys)
    x     = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_o = ax.bar(x - width/2, orig, width, label='With Wikidata (Original)',
                    color=C_ORIG, alpha=0.85, edgecolor='white')
    bars_a = ax.bar(x + width/2, abl,  width, label='Without Wikidata (Ablation)',
                    color=C_ABL,  alpha=0.85, edgecolor='white')

    # Value labels on bars
    for bar in bars_o:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color='#333')
    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color='#333')

    _group_dividers(ax, groups, n)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _ax_style(ax, 'Overall F1 Score: With Wikidata vs Without Wikidata (Ablation)',
              ylim=(0.3, 0.85))
    ax.legend(fontsize=10, framealpha=0.9)

    # Group labels at top
    group_positions = {}
    for i, g in enumerate(groups):
        group_positions.setdefault(g, []).append(i)
    for g, idxs in group_positions.items():
        mid = np.mean(idxs)
        ax.text(mid, 0.82, g, ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='#555',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#eee',
                          edgecolor='none', alpha=0.7))

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'overall_f1_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 2: Per-metric breakdown ───────────────────────────────────────────

def plot_metric_breakdown(data: dict):
    keys   = list(data.keys())
    labels = [data[k]['label'] for k in keys]
    groups = [data[k]['group'] for k in keys]
    n     = len(keys)
    x     = np.arange(n)
    width = 0.35

    metrics = [
        ('Ent_F1',  'Entity F1'),
        ('Attr_F1', 'Attribute F1'),
        ('Rel_F1',  'Relationship F1'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle('Per-Metric F1: With Wikidata vs Without Wikidata (Ablation)',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        orig_vals = [data[k]['orig'][metric_key] for k in keys]
        abl_vals  = [data[k]['abl'][metric_key]  for k in keys]

        ax.bar(x - width/2, orig_vals, width, label='With Wikidata',
               color=C_ORIG, alpha=0.85, edgecolor='white')
        ax.bar(x + width/2, abl_vals,  width, label='Without Wikidata',
               color=C_ABL,  alpha=0.85, edgecolor='white')

        _group_dividers(ax, groups, n)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        _ax_style(ax, metric_label, ylim=(0.0, 1.0))
        ax.legend(fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'metric_breakdown.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 3: Delta impact ────────────────────────────────────────────────────

def plot_delta_impact(data: dict):
    keys   = list(data.keys())
    labels = [data[k]['label'] for k in keys]
    groups = [data[k]['group'] for k in keys]
    n      = len(keys)
    x      = np.arange(n)
    width  = 0.2

    metric_keys   = ['Ent_F1', 'Attr_F1', 'Rel_F1', 'Overall_F1']
    metric_labels = ['Entity F1', 'Attr F1', 'Rel F1', 'Overall F1']
    offsets       = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    colors        = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']

    fig, ax = plt.subplots(figsize=(16, 6))

    for mk, ml, off, col in zip(metric_keys, metric_labels, offsets, colors):
        deltas = [data[k]['abl'][mk] - data[k]['orig'][mk] for k in keys]
        bar_colors = [C_POS if d >= 0 else C_NEG for d in deltas]
        bars = ax.bar(x + off, deltas, width, label=ml,
                      color=bar_colors, alpha=0.75, edgecolor='white')
        # Value label
        for bar, d in zip(bars, deltas):
            y_pos = d + 0.003 if d >= 0 else d - 0.012
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{d:+.3f}', ha='center', va='bottom' if d >= 0 else 'top',
                    fontsize=6.5, color='#333')

    ax.axhline(0, color='black', linewidth=0.8)
    _group_dividers(ax, groups, n)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _ax_style(ax, 'Impact of Removing Wikidata  (Δ F1 = Ablation − Original)\n'
                  'Green = Wikidata hurts  |  Red = Wikidata helps',
              ylabel='Δ F1', ylim=(-0.25, 0.25), grid=True)

    # Custom legend for metrics
    legend_patches = [
        mpatches.Patch(color=col, label=ml, alpha=0.75)
        for col, ml in zip(colors, metric_labels)
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9,
              loc='upper right')

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'delta_impact.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Figure 4: Heatmap comparison ──────────────────────────────────────────────

def plot_heatmap(data: dict):
    keys        = list(data.keys())
    short_lbls  = [data[k]['label'].replace('\n', ' ') for k in keys]
    metrics_o   = ['Ent_F1', 'Attr_F1', 'Rel_F1', 'Overall_F1']
    col_headers = (
        ['Orig Ent', 'Orig Attr', 'Orig Rel', 'Orig Overall']
        + ['Abl Ent', 'Abl Attr', 'Abl Rel', 'Abl Overall']
        + ['Δ Ent', 'Δ Attr', 'Δ Rel', 'Δ Overall']
    )

    rows = []
    for k in keys:
        row = (
            [data[k]['orig'][m] for m in metrics_o]
            + [data[k]['abl'][m]  for m in metrics_o]
            + [data[k]['abl'][m] - data[k]['orig'][m] for m in metrics_o]
        )
        rows.append(row)
    matrix = np.array(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             gridspec_kw={'width_ratios': [4, 4, 4]})
    fig.suptitle('Full Metric Table: Original vs Ablation (Without Wikidata)',
                 fontsize=13, fontweight='bold')

    sections = [
        (axes[0], matrix[:, :4],  col_headers[:4],  'Blues',  'Original (With Wikidata)'),
        (axes[1], matrix[:, 4:8], col_headers[4:8], 'Oranges','Ablation (Without Wikidata)'),
        (axes[2], matrix[:, 8:],  col_headers[8:],  'RdYlGn', 'Δ (Ablation − Original)'),
    ]

    for ax, mat, cols, cmap, title in sections:
        is_delta = title.startswith('Δ')
        vmin = -0.2 if is_delta else 0.3
        vmax =  0.2 if is_delta else 0.9
        im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=30, ha='right', fontsize=8)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(short_lbls, fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)

        # Cell text
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                text_color = 'white' if abs(v) > 0.7 else 'black'
                prefix = '+' if (is_delta and v > 0) else ''
                ax.text(j, i, f'{prefix}{v:.3f}', ha='center', va='center',
                        fontsize=7.5, color=text_color, fontweight='bold')

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'heatmap_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data = load_data()
    if not data:
        print("No data loaded. Check paths.")
        return

    print(f"  {len(data)} variants loaded.\n")
    print("Generating figures...")
    plot_overall_f1(data)
    plot_metric_breakdown(data)
    plot_delta_impact(data)
    plot_heatmap(data)

    print(f"\nAll plots saved to: {PLOTS_DIR}")

    # Print quick summary table
    print(f"\n{'Variant':<30} {'Orig':>6}  {'Abl':>6}  {'Δ':>7}")
    print('-' * 55)
    for k, v in data.items():
        o = v['orig']['Overall_F1']
        a = v['abl']['Overall_F1']
        lbl = v['label'].replace('\n', ' ')
        print(f"  {lbl:<28} {o:.4f}  {a:.4f}  {a-o:+.4f}")


if __name__ == '__main__':
    main()
