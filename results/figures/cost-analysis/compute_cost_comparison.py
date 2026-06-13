"""
Compare avg/median time & tokens per exercise across pipeline variants.
Uses MEDIAN for time (robust to timeout outliers).
"""

import os, json, re, glob
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
GEN  = os.path.join(ROOT, "output/generation")
PROB = os.path.join(ROOT, "output/probability")
DSL  = os.path.join(ROOT, "output/DSL-TOT-DM/data")
COST = os.path.join(ROOT, "output/cost")
OUT  = os.path.dirname(os.path.abspath(__file__))

# ── helpers ──────────────────────────────────────────────────────────────────
def load_times(folder):
    times = []
    for fp in glob.glob(os.path.join(folder, "*.json")):
        try:
            with open(fp) as f: d = json.load(f)
            t = d.get("processing_time", 0) or 0
            times.append(float(t))
        except Exception:
            pass
    return np.array(times)

def load_tokens_json(folder):
    toks = []
    for fp in glob.glob(os.path.join(folder, "*.json")):
        try:
            with open(fp) as f: d = json.load(f)
            tok = d.get("tokens", {})
            if isinstance(tok, dict):
                toks.append(tok.get("total_tokens", 0))
            elif isinstance(tok, (int, float)):
                toks.append(int(tok))
        except Exception:
            pass
    return np.array(toks)

def last_complete_tokens(txt_path):
    """Return (total_tokens, n_files) from the run with most files."""
    if not os.path.exists(txt_path): return 0, 0
    text = open(txt_path).read()
    best_n, best_tok = 0, 0
    for blk in text.split("=" * 50):
        m_n = re.search(r"Total Files Processed:\s*(\d+)", blk)
        m_t = re.search(r"Total Tokens Used:\s*(\d+)", blk)
        if m_n and m_t:
            n, t = int(m_n.group(1)), int(m_t.group(1))
            if n > best_n: best_n, best_tok = n, t
    return best_tok, best_n

def med(arr): return float(np.median(arr)) if len(arr) else 0.0
def mn(arr):  return float(np.mean(arr))   if len(arr) else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Collect all pipeline variants
# ─────────────────────────────────────────────────────────────────────────────
rows = []   # (label, gen_median, gen_mean, bn_median, tokens_avg, n_ex)

def add(label, gen_folder, bn_folder, cost_txt, tag=""):
    gen = load_times(gen_folder)
    bn  = load_times(bn_folder) if bn_folder else np.array([])
    tok_total, tok_n = last_complete_tokens(os.path.join(COST, cost_txt)) if cost_txt else (0,0)
    avg_tok = tok_total / tok_n if tok_n > 0 else 0
    rows.append({
        "label":      label,
        "tag":        tag,
        "n":          len(gen),
        "gen_med":    med(gen),
        "gen_mean":   mn(gen),
        "gen_max":    float(np.max(gen)) if len(gen) else 0,
        "bn_med":     med(bn),
        "total_med":  med(gen) + med(bn),
        "tokens":     avg_tok,
    })

# ── Multi-LLMs generation (each LLM separately) ──────────────────────────────
add("Multi-LLMs\nfewshot GPT",
    os.path.join(GEN,  "multi-llms/few-shot-gpt"),
    os.path.join(PROB, "multi-llms/pro_fewshot_gpt_0.5_1.0"),
    "cost_report_genetaion_multi_fewshot_gpt.txt",   tag="multi")

add("Multi-LLMs\nfewshot LLama",
    os.path.join(GEN,  "multi-llms/few-shot-llama"),
    os.path.join(PROB, "multi-llms/pro_fewshot_llama_0.5_1.0"),
    "cost_report_generation_multi_fewshot_llama.txt", tag="multi")

add("Multi-LLMs\nzeroshot GPT",
    os.path.join(GEN,  "multi-llms/zero-shot-gpt"),
    os.path.join(PROB, "multi-llms/pro_fewshot_gpt_0.5_1.0"),
    "cost_report_pro_estimation_multi_llms_zeroshot_gpt.txt", tag="multi")

add("Multi-LLMs\nzeroshot LLama",
    os.path.join(GEN,  "multi-llms/zero-shot-llama"),
    os.path.join(PROB, "multi-llms/pro_fewshot_llama_0.5_1.0"),
    "cost_report_pro_estimation_multi_zeroshot_llama.txt",    tag="multi")

# ── One-LLM ──────────────────────────────────────────────────────────────────
add("One-LLM\nfewshot GPT",
    os.path.join(GEN,  "one-llm/one_llm_few_shot_gpt"),
    os.path.join(PROB, "one-llm/pro_few_shot_llama_0.5_1.0"),
    "cost_report_generation_onellm_fewshot_gpt.txt",  tag="one")

add("One-LLM\nfewshot LLama",
    os.path.join(GEN,  "one-llm/one_llm_few_shot_llama"),
    os.path.join(PROB, "one-llm/pro_few_shot_llama_0.5_1.0"),
    "cost_report_generation_onellm_fewshot_llama.txt", tag="one")

add("One-LLM\nzeroshot GPT",
    os.path.join(GEN,  "one-llm/one_llm_zero_shot_gpt"),
    os.path.join(PROB, "one-llm/pro_few_shot_llama_0.5_1.0"),
    "cost_report_generation_onellm_fewshot_gpt.txt",  tag="one")

add("One-LLM\nzeroshot LLama",
    os.path.join(GEN,  "one-llm/one_llm_zero_shot_llama"),
    os.path.join(PROB, "one-llm/pro_few_shot_llama_0.5_1.0"),
    "", tag="one")

# ── DSL-ToT-DM ───────────────────────────────────────────────────────────────
dsl_folder = os.path.join(DSL, "LLama ")
dsl_gen    = load_times(dsl_folder)
dsl_tok    = load_tokens_json(dsl_folder)
rows.append({
    "label": "DSL-ToT-DM\nLLama",
    "tag":   "tot",
    "n":     len(dsl_gen),
    "gen_med":   med(dsl_gen),
    "gen_mean":  mn(dsl_gen),
    "gen_max":   float(np.max(dsl_gen)) if len(dsl_gen) else 0,
    "bn_med":    0.0,
    "total_med": med(dsl_gen),
    "tokens":    mn(dsl_tok),
})

# ─────────────────────────────────────────────────────────────────────────────
# Print table
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'Method':<26} {'N':>4} {'Gen median':>11} {'Gen mean':>10} {'BN median':>10} {'Total median':>13} {'Avg tokens':>11}")
print("-" * 90)
for r in rows:
    name = r["label"].replace("\n", " ")
    print(f"{name:<26} {r['n']:>4} {r['gen_med']:>11.2f} {r['gen_mean']:>10.2f} "
          f"{r['bn_med']:>10.2f} {r['total_med']:>13.2f} {r['tokens']:>11.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure — 3 subplots: Gen time | Total time | Tokens
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TAG_COLOR = {
    "multi": "#2563EB",
    "one":   "#16A34A",
    "tot":   "#F97316",
}
TAG_LABEL = {
    "multi": "Multi-LLMs",
    "one":   "One-LLM",
    "tot":   "DSL-ToT-DM",
}

labels = [r["label"] for r in rows]
colors = [TAG_COLOR[r["tag"]] for r in rows]
n      = len(rows)
x      = np.arange(n)
bw     = 0.6

def make_subplot(ax, vals, title, ylabel, unit_fmt=".1f"):
    bars = ax.bar(x, vals, width=bw, color=colors, edgecolor="white", zorder=3)
    best = int(np.argmin(vals))
    bars[best].set_edgecolor("#F59E0B"); bars[best].set_linewidth(2.2)
    ax.text(x[best], vals[best] * 1.025, "★",
            ha="center", va="bottom", fontsize=11, color="#F59E0B", zorder=5)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2, v * 1.015,
                    f"{v:{unit_fmt}}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color="#111", zorder=5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=35, ha="right")
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.7, n - 0.3)

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
fig.subplots_adjust(wspace=0.38, bottom=0.32)

make_subplot(axes[0], [r["gen_med"]   for r in rows],
             "Median Generation Time",       "seconds / exercise")
make_subplot(axes[1], [r["total_med"] for r in rows],
             "Median Total Time (Gen + BN)", "seconds / exercise")
make_subplot(axes[2], [r["tokens"]    for r in rows],
             "Avg Tokens Used",             "tokens / exercise", ".0f")

fig.suptitle("Cost & Time Comparison — Median per Exercise  (★ = lowest)",
             fontsize=12, fontweight="bold", y=1.02)

legend_handles = [mpatches.Patch(color=TAG_COLOR[t], label=TAG_LABEL[t])
                  for t in ["multi", "one", "tot"]]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

out_fig = os.path.join(OUT, "cost_comparison.png")
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_fig}")
plt.close()
