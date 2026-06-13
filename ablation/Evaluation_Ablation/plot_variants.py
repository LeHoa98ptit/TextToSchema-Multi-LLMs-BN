"""
Ablation Study — Focused Variant Comparison
============================================
Plots only three model groups:
  • Multi-LLMs-BN-LLaMA3   (few-shot + zero-shot)
  • One-LLM-BN-LLaMA3      (few-shot + zero-shot)
  • One-LLM-BN-GPT          (few-shot + zero-shot)

Figures saved to ablation/Evaluation_Ablation/plots/:
  1. variants_overall.png       — Overall F1: Original vs Ablation
  2. variants_metrics.png       — Entity / Attr / Rel F1 breakdown
  3. variants_delta.png         — Δ F1 = Ablation − Original
  4. variants_heatmap.png       — Full metric table heatmap

Usage:
    python ablation/Evaluation_Ablation/plot_variants.py
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
_SELF_DIR    = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(_SELF_DIR, '..', '..'))

ABLATION_CSV = os.path.join(_SELF_DIR, 'results', 'summary.csv')
ORIGINAL_DIR = os.path.join(project_root, 'Output_final', 'results', 'F1Score_2')
PLOTS_DIR    = os.path.join(_SELF_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Colours ────────────────────────────────────────────────────────────────────
C_ORIG = '#2196F3'   # blue  — original (with Wikidata)
C_ABL  = '#FF9800'   # orange — ablation (without Wikidata)
C_POS  = '#4CAF50'   # green  — positive delta
C_NEG  = '#F44336'   # red    — negative delta

# ── Selected variants only ─────────────────────────────────────────────────────
# (ablation_key, original_csv, display_label, group)
VARIANTS = [
    (
        "multi-llms-few-shot-llama",
        os.path.join(ORIGINAL_DIR, "Multi-LLMs-withBN",
                     "opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0).csv"),
        "Multi-LLMs\nFew-LLaMA3",
        "Multi-LLMs\nLLaMA3",
    ),
    (
        "multi-llms-zero-shot-llama",
        os.path.join(ORIGINAL_DIR, "Multi-LLMs-withBN",
                     "opt_zeroshot_llama_0.5_1.0-(1.2-1.0-1.0).csv"),
        "Multi-LLMs\nZero-LLaMA3",
        "Multi-LLMs\nLLaMA3",
    ),
    (
        "one-llm-one_llm_few_shot_llama",
        os.path.join(ORIGINAL_DIR, "One-LLMs-withBN",
                     "opt_one_fewshot_llama_0.5_1.0-(1.2-1.0-1.0).csv"),
        "One-LLM\nFew-LLaMA3",
        "One-LLM\nLLaMA3",
    ),
    (
        "one-llm-one_llm_zero_shot_llama",
        os.path.join(ORIGINAL_DIR, "One-LLMs-withBN",
                     "opt_one_zeroshot_llama_0.5_1.0-(1.2-1.0-1.0).csv"),
        "One-LLM\nZero-LLaMA3",
        "One-LLM\nLLaMA3",
    ),
    (
        "one-llm-one_llm_few_shot_gpt",
        os.path.join(ORIGINAL_DIR, "One-LLMs-withBN",
                     "opt_one_fewshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
        "One-LLM\nFew-GPT",
        "One-LLM\nGPT",
    ),
    (
        "one-llm-one_llm_zero_shot_gpt",
        os.path.join(ORIGINAL_DIR, "One-LLMs-withBN",
                     "opt_one_zeroshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv"),
        "One-LLM\nZero-GPT",
        "One-LLM\nGPT",
    ),
]

# ── Data loading ───────────────────────────────────────────────────────────────

def _read_average_row(csv_path: str) -> dict | None:
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('Exercise', '').strip().upper() == 'AVERAGE':
                    return {k: float(v) for k, v in row.items() if k != 'Exercise'}
    except Exception as e:
        print(f"  [WARN] Cannot read {csv_path}: {e}")
    return None


def load_data() -> dict:
    abl_map = {}
    with open(ABLATION_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            abl_map[row['Variant']] = {
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

        f1e  = orig_row.get('Ent_F1',  0.0)
        f1a  = orig_row.get('Attr_F1', 0.0)
        f1r  = orig_row.get('Rel_F1',  0.0)

        data[abl_key] = {
            'label': label,
            'group': group,
            'orig': {
                'Ent_F1':     f1e,
                'Attr_F1':    f1a,
                'Rel_F1':     f1r,
                'Overall_F1': (f1e + f1a + f1r) / 3,
            },
            'abl': abl_row,
        }
    return data

# ── Helpers ────────────────────────────────────────────────────────────────────

def _ax_style(ax, title, ylabel='F1 Score', ylim=(0, 1)):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _group_dividers(ax, groups):
    prev = None
    for i, g in enumerate(groups):
        if prev is not None and g != prev:
            ax.axvline(x=i - 0.5, color='gray', linewidth=1.0,
                       linestyle=':', alpha=0.7)
        prev = g


def _bar_labels(ax, bars, fontsize=8):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.008,
                f'{h:.3f}', ha='center', va='bottom', fontsize=fontsize, color='#333')

# ── Figure 1: Overall F1 ──────────────────────────────────────────────────────

def plot_overall(data: dict):
    keys   = list(data.keys())
    labels = [data[k]['label'] for k in keys]
    groups = [data[k]['group'] for k in keys]
    orig   = [data[k]['orig']['Overall_F1'] for k in keys]
    abl    = [data[k]['abl']['Overall_F1']  for k in keys]

    n     = len(keys)
    x     = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bo = ax.bar(x - width/2, orig, width, label='With Wikidata (Original)',
                color=C_ORIG, alpha=0.85, edgecolor='white')
    ba = ax.bar(x + width/2, abl,  width, label='Without Wikidata (Ablation)',
                color=C_ABL,  alpha=0.85, edgecolor='white')

    _bar_labels(ax, bo)
    _bar_labels(ax, ba)
    _group_dividers(ax, groups)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _ax_style(ax, 'Overall F1: With Wikidata vs Without Wikidata', ylim=(0.35, 0.90))
    ax.legend(fontsize=10, framealpha=0.9)

    # Group labels
    gpos = {}
    for i, g in enumerate(groups):
        gpos.setdefault(g, []).append(i)
    for g, idxs in gpos.items():
        mid = np.mean(idxs)
        ax.text(mid, 0.87, g.replace('\n', ' '), ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#444',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#eee',
                          edgecolor='none', alpha=0.75))

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'variants_overall.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# ── Figure 2: Per-metric breakdown ────────────────────────────────────────────

def plot_metrics(data: dict):
    keys   = list(data.keys())
    labels = [data[k]['label'] for k in keys]
    groups = [data[k]['group'] for k in keys]
    n      = len(keys)
    x      = np.arange(n)
    width  = 0.35

    metrics = [
        ('Ent_F1',  'Entity F1'),
        ('Attr_F1', 'Attribute F1'),
        ('Rel_F1',  'Relationship F1'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    fig.suptitle('Per-Metric F1: With Wikidata vs Without Wikidata',
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, (mk, ml) in zip(axes, metrics):
        ov = [data[k]['orig'][mk] for k in keys]
        av = [data[k]['abl'][mk]  for k in keys]
        bo = ax.bar(x - width/2, ov, width, label='With Wikidata',
                    color=C_ORIG, alpha=0.85, edgecolor='white')
        ba = ax.bar(x + width/2, av, width, label='Without Wikidata',
                    color=C_ABL,  alpha=0.85, edgecolor='white')
        _bar_labels(ax, bo, fontsize=7.5)
        _bar_labels(ax, ba, fontsize=7.5)
        _group_dividers(ax, groups)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        _ax_style(ax, ml, ylim=(0.0, 1.0))
        ax.legend(fontsize=8.5, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'variants_metrics.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# ── Figure 3: Delta ───────────────────────────────────────────────────────────

def plot_delta(data: dict):
    keys   = list(data.keys())
    labels = [data[k]['label'] for k in keys]
    groups = [data[k]['group'] for k in keys]
    n      = len(keys)
    x      = np.arange(n)
    width  = 0.18

    metric_keys   = ['Ent_F1', 'Attr_F1', 'Rel_F1', 'Overall_F1']
    metric_labels = ['Entity F1', 'Attr F1', 'Rel F1', 'Overall F1']
    offsets       = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]
    colors        = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']

    fig, ax = plt.subplots(figsize=(13, 6))

    for mk, ml, off, col in zip(metric_keys, metric_labels, offsets, colors):
        deltas = [data[k]['abl'][mk] - data[k]['orig'][mk] for k in keys]
        bar_colors = [C_POS if d >= 0 else C_NEG for d in deltas]
        bars = ax.bar(x + off, deltas, width, label=ml,
                      color=bar_colors, alpha=0.75, edgecolor='white')
        for bar, d in zip(bars, deltas):
            yp = d + 0.003 if d >= 0 else d - 0.012
            ax.text(bar.get_x() + bar.get_width() / 2, yp,
                    f'{d:+.3f}', ha='center',
                    va='bottom' if d >= 0 else 'top',
                    fontsize=6.5, color='#333')

    ax.axhline(0, color='black', linewidth=0.8)
    _group_dividers(ax, groups)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    _ax_style(ax,
              'Impact of Removing Wikidata  (Δ F1 = Ablation − Original)\n'
              'Green = Wikidata hurts  |  Red = Wikidata helps',
              ylabel='Δ F1', ylim=(-0.25, 0.15))

    legend_patches = [
        mpatches.Patch(color=col, label=ml, alpha=0.75)
        for col, ml in zip(colors, metric_labels)
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9, loc='lower right')

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'variants_delta.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# ── Figure 4: Heatmap ─────────────────────────────────────────────────────────

def plot_heatmap(data: dict):
    keys       = list(data.keys())
    short_lbls = [data[k]['label'].replace('\n', ' ') for k in keys]
    met_keys   = ['Ent_F1', 'Attr_F1', 'Rel_F1', 'Overall_F1']
    col_hdrs   = (
        ['Orig Ent', 'Orig Attr', 'Orig Rel', 'Orig Ovrl']
        + ['Abl Ent',  'Abl Attr',  'Abl Rel',  'Abl Ovrl']
        + ['Δ Ent',    'Δ Attr',    'Δ Rel',     'Δ Ovrl']
    )

    rows = []
    for k in keys:
        row = (
            [data[k]['orig'][m] for m in met_keys]
            + [data[k]['abl'][m]  for m in met_keys]
            + [data[k]['abl'][m] - data[k]['orig'][m] for m in met_keys]
        )
        rows.append(row)
    matrix = np.array(rows)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4),
                             gridspec_kw={'width_ratios': [4, 4, 4]})
    fig.suptitle('Full Metric Table: Original vs Ablation (Selected Variants)',
                 fontsize=13, fontweight='bold')

    sections = [
        (axes[0], matrix[:, :4],  col_hdrs[:4],  'Blues',   'Original (With Wikidata)'),
        (axes[1], matrix[:, 4:8], col_hdrs[4:8], 'Oranges', 'Ablation (Without Wikidata)'),
        (axes[2], matrix[:, 8:],  col_hdrs[8:],  'RdYlGn',  'Δ (Ablation − Original)'),
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
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                tc = 'white' if abs(v) > 0.7 else 'black'
                prefix = '+' if (is_delta and v > 0) else ''
                ax.text(j, i, f'{prefix}{v:.3f}', ha='center', va='center',
                        fontsize=7.5, color=tc, fontweight='bold')

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, 'variants_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...")
    data = load_data()
    if not data:
        print("No data loaded. Check paths.")
        return
    print(f"  {len(data)} variants loaded.\n")

    print("Generating figures...")
    plot_overall(data)
    plot_metrics(data)
    plot_delta(data)
    plot_heatmap(data)
    print(f"\nAll plots saved to: {PLOTS_DIR}")

    print(f"\n{'Variant':<30} {'Orig':>6}  {'Abl':>6}  {'Δ':>7}")
    print('-' * 52)
    for k, v in data.items():
        o   = v['orig']['Overall_F1']
        a   = v['abl']['Overall_F1']
        lbl = v['label'].replace('\n', ' ')
        print(f"  {lbl:<28} {o:.4f}  {a:.4f}  {a-o:+.4f}")


if __name__ == '__main__':
    main()
