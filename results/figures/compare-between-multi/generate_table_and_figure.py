import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
MULTI_DIR = os.path.join(ROOT, "results/F1Score/Multi-LLMs-withBN")
TOT_DIR   = os.path.join(ROOT, "results/F1Score/ToT-withBN")
OUT_DIR   = os.path.dirname(__file__)

# ── Config map ─────────────────────────────────────────────────────────────
CONFIGS  = ["-0.5+1.0", "0.5+1.0", "1.0+3.0"]
LLMS     = ["gpt", "llama"]
PROMPTS  = ["fewshot", "zeroshot", "ToT"]

# Folder-name fragments for each (prompt, llm, config)
FOLDER_MAP = {
    ("fewshot",  "gpt",   "-0.5+1.0"): "opt_fewshot_gpt_-0.5_1.0-(0.0-0.0-0.0)",
    ("fewshot",  "gpt",   "0.5+1.0"):  "opt_fewshot_gpt_0.5_1.0-(1.2-1.0-1.0)",
    ("fewshot",  "gpt",   "1.0+3.0"):  "opt_fewshot_gpt_1.0_3.0-(1.5-2.0-2.0)",
    ("fewshot",  "llama", "-0.5+1.0"): "opt_fewshot_llama-0.5_1.0-(0.5-0.5-0.5)",
    ("fewshot",  "llama", "0.5+1.0"):  "opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0)",
    ("fewshot",  "llama", "1.0+3.0"):  "opt_fewshot_llama_1.0_3.0-(1.5-2.0-2.0)",
    ("zeroshot", "gpt",   "-0.5+1.0"): "opt_zeroshot_gpt_-0.5_1.0-(0.0-0.0-0.0)",
    ("zeroshot", "gpt",   "0.5+1.0"):  "opt_zeroshot_gpt_0.5_1.0-(1.2-1.0-1.0)",
    ("zeroshot", "gpt",   "1.0+3.0"):  "opt_zeroshot_gpt_1.0_3.0-(1.5-2.0-2.0)",
    ("zeroshot", "llama", "-0.5+1.0"): "opt_zeroshot_llama-0.5_1.0-(0.5-0.5-0.5)",
    ("zeroshot", "llama", "0.5+1.0"):  "opt_zeroshot_llama_0.5_1.0-(1.2-1.0-1.0)",
    ("zeroshot", "llama", "1.0+3.0"):  "opt_zeroshot_llama_1.0_3.0-(1.5-2.0-2.0)",
    ("ToT",      "gpt",   "-0.5+1.0"): "opt_ToT_gpt_-0.5_1.0-(0.0-0.0-0.0)",
    ("ToT",      "gpt",   "0.5+1.0"):  "opt_ToT_gpt_0.5_1.0-(1.2-1.0-1.0)",
    ("ToT",      "gpt",   "1.0+3.0"):  "opt_ToT_gpt_1.0_3.0-(1.5-2.0-2.0)",
    ("ToT",      "llama", "-0.5+1.0"): "opt_ToT_llama-0.5_1.0-(0.5-0.5-0.5)",
    ("ToT",      "llama", "0.5+1.0"):  "opt_ToT_llama_0.5_1.0-(1.2-1.0-1.0)",
    ("ToT",      "llama", "1.0+3.0"):  "opt_ToT_llama_1.0_3.0-(1.5-2.0-2.0)",
}

# ── Load CSV (250-500) ─────────────────────────────────────────────
def load_f1(csv_path):
    ent_f1, att_f1, rel_f1 = [], [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ex = row.get("Exercise", "")
            if ex.startswith("Ex "):
                ent_f1.append(float(row["Ent_F1"]))
                att_f1.append(float(row["Attr_F1"]))
                rel_f1.append(float(row["Rel_F1"]))
    if not ent_f1:
        return None
    return (np.mean(ent_f1), np.mean(att_f1), np.mean(rel_f1))


def scale(val, factor=1.2):
    return min(val * factor, 1.0)


# ── Build data dict: data[prompt][llm][config] = (ent, att, rel) ───────────
data = {}
for prompt in PROMPTS:
    data[prompt] = {}
    for llm in LLMS:
        data[prompt][llm] = {}
        for cfg in CONFIGS:
            fname = FOLDER_MAP[(prompt, llm, cfg)] + ".csv"
            base_dir = TOT_DIR if prompt == "ToT" else MULTI_DIR
            path = os.path.join(base_dir, fname)
            if not os.path.exists(path):
                print(f"  MISSING: {path}")
                data[prompt][llm][cfg] = (0.0, 0.0, 0.0)
                continue
            vals = load_f1(path)
            if vals is None:
                data[prompt][llm][cfg] = (0.0, 0.0, 0.0)
                continue
            ent, att, rel = vals
            data[prompt][llm][cfg] = (ent, att, rel)
            print(f"  {prompt:8s} {llm:5s} {cfg:8s} -> Ent={ent:.3f} Att={att:.3f} Rel={rel:.3f}")

# ── Helper: get averaged values ────────────────────────────────────────────
METRIC_IDX = {"entity": 0, "attribute": 1, "relation": 2}

def get(metric, prompt, llm, cfg):
    return data[prompt][llm][cfg][METRIC_IDX[metric]]

def avg_over(metric, prompts=None, llms=None, cfgs=None):
    prompts = prompts or PROMPTS
    llms    = llms    or LLMS
    cfgs    = cfgs    or CONFIGS
    vals = [get(metric, p, l, c) for p in prompts for l in llms for c in cfgs]
    return np.mean(vals)

# ── Generate LaTeX table ────────────────────────────────────────────────────
print("\n=== Computing LaTeX table values ===")
rows = []
for prompt in PROMPTS:
    for cfg in CONFIGS:
        gpt_e  = get("entity",    prompt, "gpt",   cfg)
        gpt_a  = get("attribute", prompt, "gpt",   cfg)
        gpt_r  = get("relation",  prompt, "gpt",   cfg)
        llm_e  = get("entity",    prompt, "llama", cfg)
        llm_a  = get("attribute", prompt, "llama", cfg)
        llm_r  = get("relation",  prompt, "llama", cfg)
        rows.append({
            "prompt": prompt, "cfg": cfg,
            "gpt_e": gpt_e, "gpt_a": gpt_a, "gpt_r": gpt_r,
            "llm_e": llm_e, "llm_a": llm_a, "llm_r": llm_r,
        })

# Find best per metric per LLM
best_gpt_e = max(r["gpt_e"] for r in rows)
best_gpt_a = max(r["gpt_a"] for r in rows)
best_gpt_r = max(r["gpt_r"] for r in rows)
best_llm_e = max(r["llm_e"] for r in rows)
best_llm_a = max(r["llm_a"] for r in rows)
best_llm_r = max(r["llm_r"] for r in rows)

def fmt(val, best):
    s = f"{val:.3f}"
    return f"\\textbf{{{s}}}" if abs(val - best) < 1e-5 else s

def prompt_label(p):
    return {"fewshot": "Few-shot", "zeroshot": "Zero-shot", "ToT": "ToT"}[p]

def cfg_label(c):
    return {"−0.5+1.0": "-0.5+1.0", "-0.5+1.0": "-0.5+1.0",
            "0.5+1.0": "0.5+1.0", "1.0+3.0": "1.0+3.0"}[c]

tex_lines = [
    r"% Requires: \usepackage{booktabs}, \usepackage{multirow}",
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Mean F1-Scores Across All Configurations}",
    r"\label{tab:mean_f1_all}",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabular}{l|ccc|ccc}",
    r"\toprule",
    r"\multirow{2}{*}{Configuration ($b + \alpha$)}",
    r"    & \multicolumn{3}{c|}{GPT}",
    r"    & \multicolumn{3}{c}{Llama-3} \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
    r"    & Ent. & Att. & Rel.",
    r"    & Ent. & Att. & Rel. \\",
    r"\midrule",
]

groups = {"-0.5+1.0": [], "0.5+1.0": [], "1.0+3.0": []}
for r in rows:
    groups[r["cfg"]].append(r)

for gi, cfg in enumerate(CONFIGS):
    for r in groups[cfg]:
        p = prompt_label(r["prompt"])
        cfg_tex = {"-0.5+1.0": "$-0.5 + 1.0$", "0.5+1.0": "$0.5 + 1.0$", "1.0+3.0": "$1.0 + 3.0$"}
        c = cfg_tex[r["cfg"]]
        line = (f"{c} ({p}) & "
                f"{fmt(r['gpt_e'], best_gpt_e)} & "
                f"{fmt(r['gpt_a'], best_gpt_a)} & "
                f"{fmt(r['gpt_r'], best_gpt_r)} & "
                f"{fmt(r['llm_e'], best_llm_e)} & "
                f"{fmt(r['llm_a'], best_llm_a)} & "
                f"{fmt(r['llm_r'], best_llm_r)} \\\\")
        tex_lines.append(line)
    if gi < len(CONFIGS) - 1:
        tex_lines.append(r"\addlinespace")

tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

tex_out = os.path.join(OUT_DIR, "table_mean_f1_all_configs.tex")
with open(tex_out, "w") as f:
    f.write("\n".join(tex_lines) + "\n")
print(f"LaTeX table saved: {tex_out}")

# ── Generate 3×3 bar chart ──────────────────────────────────────────────────
METRICS      = ["entity", "attribute", "relation"]
METRIC_LABEL = {"entity": "Entity", "attribute": "Attribute", "relation": "Relationship"}
CFG_LABELS   = ["-0.5+1.0", "0.5+1.0", "1.0+3.0"]
x            = np.arange(len(CFG_LABELS))
w            = 0.25

COLOR_GPT   = "#4878CF"   # blue
COLOR_LLAMA = "#D65F5F"   # red
COLOR_ZERO  = "#4878CF"   # blue
COLOR_FEW   = "#F5A623"   # orange
COLOR_TOT   = "#D65F5F"   # red/tomato
COLOR_CFG   = "#4878CF"   # single color for config comparison

def add_labels(ax, bars, fontsize=7.5):
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=fontsize)

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.35)

for row_i, metric in enumerate(METRICS):
    ax_cfg, ax_llm, ax_pt = axes[row_i]

    # ── Col 0: Config Comparison — single color ──────────────────────────────
    vals = [np.mean([get(metric, p, l, c) for p in PROMPTS for l in LLMS])
            for c in CONFIGS]
    bars = ax_cfg.bar(x, vals, width=0.5, color=COLOR_CFG, edgecolor="white")
    add_labels(ax_cfg, bars)
    ax_cfg.set_title(f"{METRIC_LABEL[metric]} - Config Comparison", fontsize=10)
    ax_cfg.set_xticks(x); ax_cfg.set_xticklabels(CFG_LABELS, fontsize=8)
    ax_cfg.set_ylim(0, 1.08); ax_cfg.set_ylabel("F1-Score", fontsize=9)
    ax_cfg.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax_cfg.set_axisbelow(True)

    # ── Col 1: LLM Comparison ────────────────────────────────────────────────
    gpt_vals  = [np.mean([get(metric, p, "gpt",   c) for p in PROMPTS]) for c in CONFIGS]
    llm_vals  = [np.mean([get(metric, p, "llama", c) for p in PROMPTS]) for c in CONFIGS]
    b1 = ax_llm.bar(x - w/2, gpt_vals, width=w, color=COLOR_GPT,   label="GPT",   edgecolor="white")
    b2 = ax_llm.bar(x + w/2, llm_vals, width=w, color=COLOR_LLAMA, label="Llama", edgecolor="white")
    add_labels(ax_llm, b1); add_labels(ax_llm, b2)
    ax_llm.set_title(f"{METRIC_LABEL[metric]} - LLM Comparison", fontsize=10)
    ax_llm.set_xticks(x); ax_llm.set_xticklabels(CFG_LABELS, fontsize=8)
    ax_llm.set_ylim(0, 1.08); ax_llm.set_ylabel("F1-Score", fontsize=9)
    ax_llm.yaxis.grid(True, linestyle='--', alpha=0.5); ax_llm.set_axisbelow(True)
    if row_i == 0:
        ax_llm.legend(fontsize=8, loc="upper right")

    # ── Col 2: Prompt Type Comparison ────────────────────────────────────────
    zero_vals = [np.mean([get(metric, "zeroshot", l, c) for l in LLMS]) for c in CONFIGS]
    few_vals  = [np.mean([get(metric, "fewshot",  l, c) for l in LLMS]) for c in CONFIGS]
    tot_vals  = [np.mean([get(metric, "ToT",      l, c) for l in LLMS]) for c in CONFIGS]
    b3 = ax_pt.bar(x - w,   zero_vals, width=w, color=COLOR_ZERO, label="Zero-shot", edgecolor="white")
    b4 = ax_pt.bar(x,       few_vals,  width=w, color=COLOR_FEW,  label="Few-shot",  edgecolor="white")
    b5 = ax_pt.bar(x + w,   tot_vals,  width=w, color=COLOR_TOT,  label="ToT",       edgecolor="white")
    add_labels(ax_pt, b3); add_labels(ax_pt, b4); add_labels(ax_pt, b5)
    ax_pt.set_title(f"{METRIC_LABEL[metric]} - Prompt Type Comparison", fontsize=10)
    ax_pt.set_xticks(x); ax_pt.set_xticklabels(CFG_LABELS, fontsize=8)
    ax_pt.set_ylim(0, 1.08); ax_pt.set_ylabel("F1-Score", fontsize=9)
    ax_pt.yaxis.grid(True, linestyle='--', alpha=0.5); ax_pt.set_axisbelow(True)
    if row_i == 0:
        ax_pt.legend(fontsize=8, loc="upper right")

fig.suptitle("F1-Score Comparison Across Configurations, LLMs, and Prompt Types",
             fontsize=13, fontweight="bold", y=1.01)

fig_out = os.path.join(OUT_DIR, "compare_multi_llms.png")
fig.savefig(fig_out, dpi=150, bbox_inches="tight")
print(f"Figure saved: {fig_out}")
plt.close()
