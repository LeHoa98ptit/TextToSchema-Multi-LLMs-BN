"""
Cost analysis: Inference Time vs Token Consumption per pipeline.
Two panels stacked vertically:
  - Top:    Baselines vs Ours
  - Bottom: Variants  vs Ours
Outputs: figures/cost_analysis.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
})

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
# category: "ours" | "baseline" | "variant"
OURS = ("Multi-LLM-BN\nLlama3 [ours]", 23.4, 6.2, "ours")

BASELINES = [
    ("Text-To-ERD\nLlama",  4.66,  4.154, "baseline"),
    ("Text-To-ERD\nGPT",    7.29,  6.454, "baseline"),
    ("DSL-ToT-DM",          194.6, 23.6,  "baseline"),
    ("SchemaAgent",         322.6, 28.9,  "baseline"),
]

VARIANTS = [
    ("One-LLM-BN\nLlama3",         3.4,   1.3,   "variant"),
    ("One-LLM-BN-NoWiki\nLlama3",  3.37,  1.3,   "variant"),
    ("One-LLM-noBN\nLlama3",       2.7,  1.3, "variant"),
    ("Multi-LLM-noBN\nLlama3",     21.0,  6.2,   "variant"),
    ("Multi-LLM-BN\nLlama3 (noWiki)", 22.8,  6.2, "variant"),
]

COLOR = {
    "ours":     "#E65100",
    "baseline": "#1565C0",
    "variant":  "#2E7D32",
}
LABEL = {
    "ours":     "Our Approach",
    "baseline": "Baseline",
    "variant":  "Variant (ablation)",
}


def draw_panel(axes_time, axes_tok, rows, title_suffix):
    """Draw a (time, token) pair of horizontal-bar axes for the given rows."""
    labels   = [r[0] for r in rows]
    t_vals   = [r[1] for r in rows]
    tok_vals = [r[2] for r in rows]
    colors   = [COLOR[r[3]] for r in rows]
    y = np.arange(len(rows))
    h = 0.55

    # ── Time ──────────────────────────────────────────────────────────────────
    bars = axes_time.barh(y, t_vals, color=colors, edgecolor="white",
                          linewidth=0.5, height=h)
    axes_time.set_yticks(y)
    axes_time.set_yticklabels(labels, fontsize=10)
    axes_time.set_xlabel("Inference Time (s / example)")
    axes_time.set_title(f"Inference Time — {title_suffix}")
    axes_time.xaxis.grid(True, linestyle="--", alpha=0.45)
    axes_time.set_axisbelow(True)
    pad = max(t_vals) * 0.03
    for bar, val in zip(bars, t_vals):
        axes_time.text(val + pad, bar.get_y() + bar.get_height() / 2,
                       f"{val}s", va="center", fontsize=9)
    axes_time.set_xlim(0, max(t_vals) * 1.22)

    # ── Tokens ────────────────────────────────────────────────────────────────
    bars2 = axes_tok.barh(y, tok_vals, color=colors, edgecolor="white",
                          linewidth=0.5, height=h)
    axes_tok.set_yticks(y)
    axes_tok.set_yticklabels(labels, fontsize=10)
    axes_tok.set_xlabel("Token Consumption (×10³ / example)")
    axes_tok.set_title(f"Token Consumption — {title_suffix}")
    axes_tok.xaxis.grid(True, linestyle="--", alpha=0.45)
    axes_tok.set_axisbelow(True)
    pad2 = max(tok_vals) * 0.03
    for bar, val in zip(bars2, tok_vals):
        axes_tok.text(val + pad2, bar.get_y() + bar.get_height() / 2,
                      f"{val}k", va="center", fontsize=9)
    axes_tok.set_xlim(0, max(tok_vals) * 1.22)


# ── Build combined rows (ours appended at bottom of each panel) ───────────────
baseline_rows = BASELINES + [OURS]
variant_rows  = VARIANTS  + [OURS]

# ── Figure 1: Baselines vs Ours ───────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5))
fig1.subplots_adjust(wspace=0.42)
draw_panel(axes1[0], axes1[1], baseline_rows, "Baselines vs Ours")
legend_patches = [mpatches.Patch(color=COLOR[c], label=LABEL[c])
                  for c in ("ours", "baseline")]
fig1.legend(handles=legend_patches, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, -0.04), framealpha=0.7, fontsize=11)
fig1.suptitle("Computational Cost — Baselines vs Our Approach",
              fontsize=13, y=1.01)
out1 = os.path.join(OUT_DIR, "cost_analysis_baselines.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ── Figure 2: Variants vs Ours ────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.subplots_adjust(wspace=0.42)
draw_panel(axes2[0], axes2[1], variant_rows, "Variants vs Ours")
legend_patches2 = [mpatches.Patch(color=COLOR[c], label=LABEL[c])
                   for c in ("ours", "variant")]
fig2.legend(handles=legend_patches2, loc="lower center", ncol=2,
            bbox_to_anchor=(0.5, -0.04), framealpha=0.7, fontsize=11)
fig2.suptitle("Computational Cost — Variants (Ablation) vs Our Approach",
              fontsize=13, y=1.01)
out2 = os.path.join(OUT_DIR, "cost_analysis_variants.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ── Summary ───────────────────────────────────────────────────────────────────
ours_t, ours_tok = OURS[1], OURS[2]
print(f"\n{'Pipeline':<32} {'Cat':<10} {'Time (s)':>9} {'Tokens (k)':>11}")
print("-" * 65)
for row in BASELINES + VARIANTS + [OURS]:
    tag = " ★" if row[3] == "ours" else ""
    print(f"{row[0].replace(chr(10),' '):<32} {row[3]:<10} {row[1]:>9.2f} {row[2]:>11.2f}{tag}")
print(f"\nText-To-ERD-GPT vs ours: ×{7.29/ours_t:.2f} time, ×{6.454/ours_tok:.2f} tokens")
print(f"DSL-ToT-DM vs ours:      ×{194.6/ours_t:.1f} slower, ×{23.6/ours_tok:.1f} more tokens")
print(f"SchemaAgent vs ours:     ×{322.6/ours_t:.1f} slower, ×{28.9/ours_tok:.1f} more tokens")
