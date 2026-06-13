import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
DATA_DIR = os.path.join(ROOT, "results/F1Score/One-LLMs-withBN")
OUT_DIR  = os.path.dirname(__file__)

FOLDER_MAP = {
    ("zeroshot", "gpt"):   "opt_one_zeroshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv",
    ("zeroshot", "llama"): "opt_one_zeroshot_llama_0.5_1.0-(1.2-1.0-1.0).csv",
    ("fewshot",  "gpt"):   "opt_one_fewshot_gpt_0.5_1.0-(1.2-1.0-1.0).csv",
    ("fewshot",  "llama"): "opt_one_fewshot_llama_0.5_1.0-(1.2-1.0-1.0).csv",
}

def load_f1(path):
    ent, att, rel = [], [], []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            ex = row.get("Exercise", "")
            if ex.startswith("Ex ") and 376 <= int(ex.split()[1]) <= 500:
                ent.append(float(row["Ent_F1"]))
                att.append(float(row["Attr_F1"]))
                rel.append(float(row["Rel_F1"]))
    return (np.mean(ent), np.mean(att), np.mean(rel)) if ent else (0, 0, 0)

# Load raw values (no scaling)
GROUPS = [
    ("zeroshot", "gpt",   "Zero-shot GPT",   "#4472C4"),
    ("zeroshot", "llama", "Zero-shot Llama",  "#C0392B"),
    ("fewshot",  "gpt",   "Few-shot GPT",     "#7BAFD4"),
    ("fewshot",  "llama", "Few-shot Llama",   "#E59B1A"),
]

vals = {}
for prompt, llm, label, color in GROUPS:
    path = os.path.join(DATA_DIR, FOLDER_MAP[(prompt, llm)])
    vals[(prompt, llm)] = load_f1(path)
    ent, att, rel = vals[(prompt, llm)]
    print(f"  {label:20s} -> Ent={ent:.3f}  Att={att:.3f}  Rel={rel:.3f}")

METRICS = [
    ("Ent_F1",  "Entity F1",       0),
    ("Attr_F1", "Attribute F1",    1),
    ("Rel_F1",  "Relationship F1", 2),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.subplots_adjust(wspace=0.08, bottom=0.18)

x = np.arange(len(GROUPS))
bar_w = 0.62

for ax, (_, title, idx) in zip(axes, METRICS):
    metric_vals = [vals[(p, l)][idx] for p, l, _, _ in GROUPS]
    best_i = int(np.argmax(metric_vals))

    for i, ((prompt, llm, label, color), v) in enumerate(zip(GROUPS, metric_vals)):
        is_best = (i == best_i)
        bar = ax.bar(i, v, width=bar_w, color=color, edgecolor="none", zorder=3)

        # Gold border on best bar
        if is_best:
            bar[0].set_edgecolor("#F5C518")
            bar[0].set_linewidth(2.5)

        # Value label
        ax.text(i, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222")

        # Star on best
        if is_best:
            ax.text(i, v + 0.052, "★",
                    ha="center", va="bottom",
                    fontsize=13, color="#F5C518", zorder=5)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([g[2] for g in GROUPS], fontsize=8.5, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)
    ax.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("F1-Score", fontsize=10)
axes[1].set_ylabel("")
axes[2].set_ylabel("")

fig.suptitle("F1-Score Comparison — One-LLM with BN (Config: 0.5 + 1.0)",
             fontsize=13, fontweight="bold", y=1.01)

# Shared legend at bottom
handles = [plt.Rectangle((0,0),1,1, color=c) for *_, c in GROUPS]
labels  = [g[2] for g in GROUPS]
fig.legend(handles, labels, loc="lower center", ncol=4,
           fontsize=9.5, frameon=False,
           bbox_to_anchor=(0.5, -0.04))

fig_out = os.path.join(OUT_DIR, "compare_one_llm.png")
fig.savefig(fig_out, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {fig_out}")
plt.close()
