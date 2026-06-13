"""
Precision vs Recall breakdown — Multi-LLM+BN Llama fewshot (best config).
Two figures:
  - figures/precision_recall_bar.png     — grouped bar (P / R / F1) per metric
  - figures/precision_recall_scatter.png — P vs R scatter with iso-F1 contours
    (one point per metric: Entity, Attribute, Relation)
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
})

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F1_DIR  = os.path.join(ROOT, "results/F1Score")
OUT_DIR = os.path.join(ROOT, "plots_extend/figures")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(
    F1_DIR, "Multi-LLMs-withBN/opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0).csv")
MODEL_LABEL = "Multi-LLM+BN Llama fewshot (best)"

METRIC_GROUPS = [
    ("Ent_P",  "Ent_R",  "Ent_F1",  "Entity"),
    ("Attr_P", "Attr_R", "Attr_F1", "Attribute"),
    ("Rel_P",  "Rel_R",  "Rel_F1",  "Relation"),
]


def load_csv(path):
    result = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            ex = row["Exercise"]
            if not ex.startswith("Ex "):
                continue
            try:
                num = int(ex.replace("Ex ", ""))
            except ValueError:
                continue
            result[num] = {k: float(v) for k, v in row.items() if k != "Exercise"}
    return result


ex_data = load_csv(CSV_PATH)
rows = list(ex_data.values())
n = len(rows)

means = {}
for p_col, r_col, f_col, title in METRIC_GROUPS:
    means[title] = {
        "P":  np.mean([r[p_col]  for r in rows]),
        "R":  np.mean([r[r_col]  for r in rows]),
        "F1": np.mean([r[f_col]  for r in rows]),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Grouped bar — x = metric (Entity/Attr/Rel), bars = P / R / F1
# ─────────────────────────────────────────────────────────────────────────────
metric_labels = [g[3] for g in METRIC_GROUPS]
p_vals  = [means[m]["P"]  for m in metric_labels]
r_vals  = [means[m]["R"]  for m in metric_labels]
f1_vals = [means[m]["F1"] for m in metric_labels]

x = np.arange(len(metric_labels))
w = 0.26
colors = {"P": "#5C9BD6", "R": "#F4A261", "F1": "#2E7D32"}

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - w, p_vals,  w, label="Precision", color=colors["P"],
       edgecolor="white", linewidth=0.5)
ax.bar(x,     r_vals,  w, label="Recall",    color=colors["R"],
       edgecolor="white", linewidth=0.5)
ax.bar(x + w, f1_vals, w, label="F1",        color=colors["F1"],
       edgecolor="white", linewidth=0.5)

# Value labels on top of bars
for xi, (pv, rv, fv) in enumerate(zip(p_vals, r_vals, f1_vals)):
    for offset, val in zip((-w, 0, w), (pv, rv, fv)):
        ax.text(xi + offset, val + 0.01, f"{val:.2f}",
                ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylabel("Score")
ax.set_title(f"Precision / Recall / F1 Breakdown\n({MODEL_LABEL})")
ax.set_ylim(0, 1.12)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(loc="lower right", framealpha=0.7)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
out_bar = os.path.join(OUT_DIR, "precision_recall_bar.png")
plt.savefig(out_bar, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_bar}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: P vs R scatter with iso-F1 contours
# ─────────────────────────────────────────────────────────────────────────────
marker_styles = {"Entity": "o", "Attribute": "s", "Relation": "^"}
point_colors  = {"Entity": "#1565C0", "Attribute": "#E65100", "Relation": "#2E7D32"}
recall_range  = np.linspace(0.01, 1.0, 300)

fig2, ax2 = plt.subplots(figsize=(6, 5.5))

# Iso-F1 curves
for f_target in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
    p_curve = f_target * recall_range / (2 * recall_range - f_target)
    valid = (p_curve >= 0) & (p_curve <= 1)
    ax2.plot(recall_range[valid], p_curve[valid],
             color="lightgray", linestyle="--", linewidth=0.9, zorder=0)
    mid = np.argmin(np.abs(recall_range[valid] - 0.6))
    rx = recall_range[valid][mid]
    py = p_curve[valid][mid]
    ax2.text(rx + 0.01, py, f"F1={f_target}", fontsize=7.5,
             color="gray", va="bottom")

# Points
for m in metric_labels:
    r_val = means[m]["R"]
    p_val = means[m]["P"]
    f_val = means[m]["F1"]
    ax2.scatter(r_val, p_val,
                color=point_colors[m],
                marker=marker_styles[m],
                s=130, zorder=4,
                label=f"{m}  (P={p_val:.3f}, R={r_val:.3f}, F1={f_val:.3f})")
    ax2.annotate(m, (r_val, p_val),
                 textcoords="offset points", xytext=(8, 4),
                 fontsize=10, fontweight="bold",
                 color=point_colors[m])

ax2.set_xlim(0.5, 1.02)
ax2.set_ylim(0.5, 1.02)
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title(f"Precision vs Recall\n({MODEL_LABEL})")
ax2.legend(loc="lower left", fontsize=8, framealpha=0.7)
ax2.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
out_scatter = os.path.join(OUT_DIR, "precision_recall_scatter.png")
plt.savefig(out_scatter, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_scatter}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n── {MODEL_LABEL}  (n={n}) ──")
print(f"{'Metric':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 44)
for m in metric_labels:
    print(f"{m:<12} {means[m]['P']:>10.3f} {means[m]['R']:>10.3f} {means[m]['F1']:>10.3f}")
