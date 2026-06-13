import os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['hatch.linewidth'] = 0.3

_HERE = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
ROOT  = os.path.abspath(os.path.join(_HERE, "../../../.."))
F1    = os.path.join(ROOT, "results/F1Score")
ABL   = os.path.join(ROOT, "ablation/optimization_ablation_hard/results")
ABL_one = os.path.join(ROOT, "ablation/ablation_one_llm/results")
OUT   = _HERE

def darken(hex_color, factor=0.65):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}'

def load_f1(path):
    ent_f1, att_f1, rel_f1 = [], [], []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            if not row.get("Exercise", "").startswith("Ex "):
                continue
            ent_f1.append(float(row["Ent_F1"]))
            att_f1.append(float(row["Attr_F1"]))
            rel_f1.append(float(row["Rel_F1"]))
    if not ent_f1:
        return None
    e, a, r = np.mean(ent_f1), np.mean(att_f1), np.mean(rel_f1)
    return (e, a, r, (e + a + r) / 3)

# Okabe & Ito colorblind-safe palette + subtle hatch patterns (no borders) "Text-To-ERD/",    "text_to_erd_llama.csv"
ALL_PIPELINES = {
    "our":          ("Multi-LLM BN-Llama3",   os.path.join(F1, "Multi-LLMs-withBN",    "opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0).csv"), "#0072B2", ""),
    "text_to_erd_llama": ("Text-To-ERD (Llama3) [15]",      os.path.join(F1, "Text-To-ERD",    "text_to_erd_llama.csv"),                    "#56B4E9", "/"),
    "text_to_erd_gpt":   ("Text-To-ERD (GPT) [15]",         os.path.join(F1, "Text-To-ERD",    "text_to_erd_gpt.csv"),                      "#E69F00", "/"),
    "dsl_tot":      ("DSL-ToT-DM [16]",             os.path.join(F1, "DSL-TOT-DM",           "GPT.csv"),                                       "#D55E00", "*"),
    "schema_agent": ("SchemaAgent [17]",         os.path.join(F1, "SchemaAgent",           "data.csv"),                                      "#999999", "-"),
    "nobn_llama":   ("Multi-LLM-noBN-Llama3",  os.path.join(F1, "Multi-LLMs-withoutBN", "few-shot-llama.csv"),                            "#56B4E9", "\\"),
    "one_bn_llama": ("One-LLM BN-Llama3",      os.path.join(F1, "One-LLMs-withBN",      "opt_one_fewshot_llama_0.5_1.0-(1.2-1.0-1.0).csv"), "#CC79A7", "*"),
    "one_nobn_llama":   ("One-LLM-noBN-Llama3",         os.path.join(F1, "One-LLM-withoutBN",      "one_llm_few_shot_llama.csv"),  "#999999", "*"),
    "our_nowd":     ("Multi-LLM-BN-NoWiki-Llama3", os.path.join(ABL, "multi-llms-few-shot-llama.csv"), "#009E73", "//"),
    "one_nowd":     ("One-LLM-BN-NoWiki-Llama3", os.path.join(ABL, "multi-llms-few-shot-llama.csv"), "#AADB98", "//"),

}

FIGURE_GROUPS = [
    ("baselines", "Comparison with Baselines",  ["our", "text_to_erd_llama", "text_to_erd_gpt", "dsl_tot", "schema_agent"]),
    ("variants",  "Comparison with Variants",   ["our", "our_nowd", "nobn_llama", "one_bn_llama", "one_nowd", "one_nobn_llama"]),
]

loaded = {}
for key, (label, path, color, hatch) in ALL_PIPELINES.items():
    vals = load_f1(path)
    if vals is None:
        print(f"  MISSING: {os.path.basename(path)}")
        vals = (0, 0, 0, 0)
    else:
        print(f"  {label:28s} Ent={vals[0]:.3f} Att={vals[1]:.3f} Rel={vals[2]:.3f} Avg={vals[3]:.3f}")
    loaded[key] = (label, vals, color, hatch)

GROUPS    = ["Entity", "Attribute", "Relationship", "Average"]
GROUP_IDX = [0, 1, 2, 3]

def make_figure(title, keys, out_path):
    data      = [loaded[k] for k in keys]
    N         = len(data)
    bar_w     = 0.7
    group_gap = 1.2
    x_centers = np.array([i * (N * bar_w + group_gap) for i in range(4)])

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    fig.patch.set_facecolor("white")

    for i, (label, vals, color, hatch) in enumerate(data):
        offsets  = x_centers + (i - N / 2 + 0.5) * bar_w
        bar_vals = [vals[g] for g in GROUP_IDX]
        alpha    = 1.0 if i == 0 else 0.82
        # solid base bar
        ax.bar(offsets, bar_vals, width=bar_w,
               color=color, alpha=alpha,
               edgecolor="none", linewidth=0, zorder=3)
        # subtle white hatch overlay — white lines on colored bar, edges blend into white bg
        if hatch == "*":
            # draw tiny ★ text markers at grid positions inside each bar
            for x_b, h_b in zip(offsets, bar_vals):
                for y_s in np.arange(0.07, h_b - 0.03, 0.13):
                    ax.text(x_b, y_s, "★", ha="center", va="center",
                            fontsize=3.5, color="white", alpha=0.7, zorder=4)
        elif hatch:
            ax.bar(offsets, bar_vals, width=bar_w,
                   facecolor="none", edgecolor="white",
                   hatch=hatch, linewidth=0, alpha=0.55, zorder=4)

    # value labels on top of each bar, vertical
    for i, (label, vals, color, hatch) in enumerate(data):
        offsets  = x_centers + (i - N / 2 + 0.5) * bar_w
        bar_vals = [vals[g] for g in GROUP_IDX]
        for x_b, v in zip(offsets, bar_vals):
            ax.text(x_b, v + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=3.8,
                    rotation=90, color="#333333", zorder=6)

    # small star above best bar per group
    for gi in range(4):
        group_vals = [data[i][1][gi] for i in range(N)]
        best_i = int(np.argmax(group_vals))
        best_v = group_vals[best_i]
        x_pos  = x_centers[gi] + (best_i - N / 2 + 0.5) * bar_w
        ax.text(x_pos, best_v + 0.085, "★",
                ha="center", va="bottom", fontsize=5.5,
                color="#E69F00", zorder=5)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(GROUPS, fontsize=6)
    ax.set_ylabel("F1", fontsize=6)
    ax.set_ylim(0, 1.12)
    ax.set_xlim(x_centers[0] - N * bar_w / 2 - 0.4,
                x_centers[-1] + N * bar_w / 2 + 0.4)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.tick_params(bottom=False, left=True, labelsize=5.5, width=0.5, length=2)
    ax.set_title(title, fontsize=6.5, pad=5)

    legend_handles = []
    for i, (label, _, color, hatch) in enumerate(data):
        alpha = 0.9 if i == 0 else 0.82
        patch = mpatches.Patch(
            facecolor=color, alpha=alpha,
            edgecolor="white" if hatch else "none",
            hatch=hatch, linewidth=0, label=label,
        )
        legend_handles.append(patch)

    leg = ax.legend(handles=legend_handles, loc="lower right",
                    ncol=1, fontsize=3.8, frameon=True,
                    framealpha=0.9, handlelength=0.7, handleheight=0.5,
                    borderpad=0.3, labelspacing=0.15,
                    edgecolor="#dddddd")
    leg.get_frame().set_linewidth(0.4)
    for text in leg.get_texts():
        text.set_fontweight("normal")

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

for suffix, title, keys in FIGURE_GROUPS:
    make_figure(title, keys, os.path.join(OUT, f"pipeline_comparison_{suffix}.png"))
