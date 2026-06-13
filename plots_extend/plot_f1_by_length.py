"""
F1 by Text Length — Multi-LLM+BN Llama fewshot (best config).
Shows Entity / Attribute / Relation F1 across Short / Medium / Long buckets.
Outputs: figures/f1_by_length.png
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

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT, "dataset/Datasets/Full-Dataset/input")
F1_DIR    = os.path.join(ROOT, "results/F1Score")
OUT_DIR   = os.path.join(ROOT, "plots_extend/figures")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(
    F1_DIR, "Multi-LLMs-withBN/opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0).csv")
MODEL_LABEL = "Multi-LLM+BN Llama (fewshot, best)"

METRICS = [
    ("Ent_F1",  "Entity F1",    "#1565C0"),
    ("Attr_F1", "Attribute F1", "#E65100"),
    ("Rel_F1",  "Relation F1",  "#2E7D32"),
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

# Word counts
wc = {}
for num in ex_data:
    p = os.path.join(INPUT_DIR, f"{num}.txt")
    if os.path.exists(p):
        with open(p) as f:
            wc[num] = len(f.read().split())

sorted_wc = sorted(wc.values())
n = len(sorted_wc)
q33 = sorted_wc[n // 3]
q66 = sorted_wc[2 * n // 3]
print(f"Thresholds — Short: <{q33}w  |  Medium: {q33}–{q66}w  |  Long: >{q66}w")


def bucket(w):
    if w < q33:
        return f"Short (<{q33}w)"
    elif w <= q66:
        return f"Medium ({q33}–{q66}w)"
    else:
        return f"Long (>{q66}w)"


buckets = [f"Short (<{q33}w)", f"Medium ({q33}–{q66}w)", f"Long (>{q66}w)"]

# Aggregate
agg = {b: {col: [] for col, *_ in METRICS} for b in buckets}
for num, metrics in ex_data.items():
    if num not in wc:
        continue
    b = bucket(wc[num])
    for col, *_ in METRICS:
        agg[b][col].append(metrics[col])

bucket_counts = {b: len(agg[b]["Ent_F1"]) for b in buckets}

# ── Grouped bar: x = bucket, groups = Ent/Attr/Rel ───────────────────────────
x = np.arange(len(buckets))
w = 0.26
fig, ax = plt.subplots(figsize=(8, 5.5))

for i, (col, label, color) in enumerate(METRICS):
    vals = [np.mean(agg[b][col]) if agg[b][col] else 0.0 for b in buckets]
    offset = (i - 1) * w
    ax.bar(x + offset, vals, w, label=label, color=color,
           edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(
    [f"{b}\n(n={bucket_counts[b]})" for b in buckets], fontsize=10)
ax.set_ylabel("F1 Score")
ax.set_title(f"F1 Score by Input Text Length\n({MODEL_LABEL})")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(loc="lower left", framealpha=0.7)
ax.yaxis.grid(True, linestyle="--", alpha=0.45)
ax.set_axisbelow(True)

plt.tight_layout()
out = os.path.join(OUT_DIR, "f1_by_length.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print(f"\n{'Bucket':<24} {'Ent_F1':>8} {'Attr_F1':>8} {'Rel_F1':>8} {'n':>5}")
print("-" * 56)
for b in buckets:
    row = f"{b:<24}"
    for col, *_ in METRICS:
        vals = agg[b][col]
        row += f" {np.mean(vals) if vals else 0.0:>8.3f}"
    row += f" {bucket_counts[b]:>5}"
    print(row)
