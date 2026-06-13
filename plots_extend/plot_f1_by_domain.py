"""
F1 per Domain — compares Our Method vs two baselines.
Domain classification: SentenceTransformer (all-MiniLM-L6-v2),
same 18-domain taxonomy as dataset_analysis.py.
Domain assignments are cached to avoid re-running the model.
Focuses on Relation F1 where the BN effect is most visible.
Outputs: figures/f1_by_domain.png
"""

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "hatch.linewidth": 0.3,
})

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT, "dataset/Datasets/Full-Dataset/input")
F1_DIR    = os.path.join(ROOT, "results/F1Score")
OUT_DIR   = os.path.join(ROOT, "plots_extend/figures")
CACHE     = os.path.join(ROOT, "plots_extend/domain_cache.json")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Pipelines to compare ──────────────────────────────────────────────────────
CONFIGS = {
    "Multi-LLM+BN Llama": {
        "csv":   "Multi-LLMs-withBN/opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0).csv",
        "color": "#0072B2",
        "hatch": "",
    },
    "Text-To-ERD-Llama3": {
        "csv":   "Text-To-ERD/text_to_erd_llama.csv",
        "color": "#56B4E9",
        "hatch": "/",
    },
    "Text-To-ERD-GPT": {
        "csv":   "Text-To-ERD/text_to_erd_gpt.csv",
        "color": "#E69F00",
        "hatch": "/",
    },
    "DSL-ToT-DM": {
        "csv":   "DSL-TOT-DM/GPT.csv",
        "color": "#D55E00",
        "hatch": "*",
    },
    "SchemaAgent": {
        "csv":   "SchemaAgent/data.csv",
        "color": "#999999",
        "hatch": "-",
    },
}

# ── Load ground-truth domain cache (built from id_domain.jsonl) ───────────────
print(f"Loading domain assignments from {CACHE}")
with open(CACHE) as f:
    ex_domain = json.load(f)


# ── Load F1 CSVs ──────────────────────────────────────────────────────────────
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


pipeline_data = {}
for label, cfg in CONFIGS.items():
    pipeline_data[label] = load_csv(os.path.join(F1_DIR, cfg["csv"]))

# ── Aggregate avg F1 (Ent+Attr+Rel / 3) per domain per pipeline ──────────────
domain_rel = defaultdict(lambda: defaultdict(list))
for label, ex_data in pipeline_data.items():
    for num, metrics in ex_data.items():
        d = ex_domain.get(str(num), "Other")
        avg = (metrics["Ent_F1"] + metrics["Attr_F1"] + metrics["Rel_F1"]) / 3
        domain_rel[d][label].append(avg)

# Keep domains with >= 5 examples in OURS; sort by ours avg F1 ascending
ours_label = list(CONFIGS.keys())[0]
active = {d: v for d, v in domain_rel.items()
          if len(v[ours_label]) >= 5}
domains_sorted = sorted(active.keys(),
                        key=lambda d: np.mean(active[d][ours_label]))

# ── Compute mean Rel_F1 per pipeline per domain ───────────────────────────────
pipeline_labels = list(CONFIGS.keys())

# Short single-line domain names for x-axis
ABBREV = {
    "IT, Software & Information Services":          "IT & Software",
    "Education":                                    "Education",
    "Culture, Sports & Entertainment":              "Culture/Sports",
    "Wholesale & Retail":                           "Retail",
    "Leasing & Business Services":                  "Business Svcs",
    "Finance":                                      "Finance",
    "Accommodation & Food Services":                "Food & Hospitality",
    "Healthcare & Social Work":                     "Healthcare",
    "Residential Services & Repair":                "Residential",
    "Scientific Research & Technical Services":     "Research & Tech",
    "Transportation, Storage & Postal":             "Transport",
    "Manufacturing":                                "Manufacturing",
    "Water, Environment & Public Facilities":       "Environment",
    "Public Administration & Social Security":      "Public Admin",
    "Agriculture, Forestry, Animal Husbandry & Fishery": "Agriculture",
    "Real Estate":                                  "Real Estate",
    "Energy & Utilities":                           "Energy",
    "Mining":                                       "Mining",
    "Construction":                                 "Construction",
}
short_domains = [ABBREV.get(d, d) for d in domains_sorted]

x      = np.arange(len(domains_sorted))
n_pipe = len(pipeline_labels)
w      = 0.75 / n_pipe          # bar width so all bars fit in width 0.75
offsets = np.linspace(-(n_pipe - 1) / 2, (n_pipe - 1) / 2, n_pipe) * w

fig, ax = plt.subplots(figsize=(9, 2.8))

for i, label in enumerate(pipeline_labels):
    cfg   = CONFIGS[label]
    vals  = [np.mean(active[d][label]) if active[d][label] else 0.0
             for d in domains_sorted]
    hatch = cfg.get("hatch", "")
    ax.bar(x + offsets[i], vals, w,
           label=label,
           color=cfg["color"],
           edgecolor="none", linewidth=0, zorder=3)
    # White hatch overlay (same style as pipeline comparison figure)
    if hatch == "*":
        for xb, vb in zip(x + offsets[i], vals):
            for y_s in np.arange(0.07, vb - 0.03, 0.13):
                ax.text(xb, y_s, "★", ha="center", va="center",
                        fontsize=3.5, color="white", alpha=0.7, zorder=4)
    elif hatch:
        ax.bar(x + offsets[i], vals, w,
               facecolor="none", edgecolor="white",
               hatch=hatch, linewidth=0, alpha=0.55, zorder=4)

labels_with_n = [f"{s}\n(n={len(active[d][ours_label])})"
                 for s, d in zip(short_domains, domains_sorted)]
ax.set_xticks(x)
ax.set_xticklabels(labels_with_n, fontsize=7, rotation=40, ha="right", rotation_mode="anchor")
ax.set_ylabel("Average F1")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.set_title("Average F1 by Domain",
             fontsize=12)
ax.legend(loc="lower right", framealpha=0.7, fontsize=7)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)


plt.tight_layout(pad=0.5)
out1 = os.path.join(OUT_DIR, "f1_by_domain_1.png")
plt.savefig(out1, dpi=150, bbox_inches="tight", pad_inches=0.05)
plt.close()
print(f"\nSaved: {out1}")

# ── Print gap table ───────────────────────────────────────────────────────────
labels_list = list(CONFIGS.keys())
print(f"\n{'Domain':<28}", end="")
for lb in labels_list:
    print(f"  {lb[:22]:>22}", end="")
print(f"  {'Gap(ours-GPT)':>14}")
print("-" * (28 + len(labels_list) * 24 + 16))
for d in domains_sorted:
    row = f"{d:<28}"
    vals_per_label = []
    for lb in labels_list:
        vals = active[d][lb]
        v = np.mean(vals) if vals else float("nan")
        vals_per_label.append(v)
        row += f"  {v:>22.3f}"
    gap = vals_per_label[0] - vals_per_label[1] 
    print(row)
