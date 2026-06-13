"""
Compute time, token, and estimated cost per exercise for each pipeline.
Saves LaTeX table to results/Reading/LaTeX_Tables/cost_analysis_pipelines.tex

Time source: processing_time field in JSON output files (median, robust to outliers).
  - Multi-LLM BN: Gen step + BN/prob step (both have separate LLM calls → separate timings).
  - One-LLM BN / noBN / baselines: Gen step only.
Token source:
  - DSL-ToT-DM: tokens field in each JSON (actual).
  - Others: inferred from same-model zero-shot cost reports (similar prompt complexity).
Cost model (USD per 1M tokens, 70% input / 30% output split):
  - GPT   : in=$0.50  out=$1.50  → blended $0.80/1M
  - LLaMA : in=$0.80  out=$0.90  → blended $0.83/1M
"""
import os, json, re, statistics, glob

_HERE = os.path.abspath(os.path.dirname(__file__))
ROOT  = os.path.abspath(os.path.join(_HERE, "../../.."))

def path(*parts):
    return os.path.join(ROOT, *parts)

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_times(folder, hung_thresh=None):
    """Return processing_time values, optionally excluding hung/timeout files."""
    times = []
    for fp in glob.glob(os.path.join(folder, "*.json")):
        if os.path.basename(fp).startswith("."): continue
        try:
            d = json.load(open(fp))
            t = d.get("processing_time", 0)
            if t and float(t) > 0:
                if hung_thresh is None or float(t) <= hung_thresh:
                    times.append(float(t))
        except Exception:
            pass
    return times

def mean_time(folder, hung_thresh=None):
    """Simple mean of processing_time, excluding hung files above threshold."""
    times = get_times(folder, hung_thresh)
    return sum(times) / len(times) if times else 0.0, len(times)

def get_tokens_json(folder):
    toks = []
    for fp in glob.glob(os.path.join(folder, "*.json")):
        if os.path.basename(fp).startswith("."): continue
        try:
            d = json.load(open(fp))
            tok = d.get("tokens", {})
            if isinstance(tok, dict) and tok.get("total_tokens"):
                toks.append(int(tok["total_tokens"]))
        except Exception:
            pass
    return toks

def best_from_cost_report(txt_path):
    """Return (total_tokens, n_files) from the largest completed batch."""
    if not os.path.exists(txt_path):
        return 0, 0
    text = open(txt_path).read()
    best_n, best_tok = 0, 0
    for blk in text.split("=" * 50):
        mn = re.search(r"Total Files Processed:\s*(\d+)", blk)
        mt = re.search(r"Total Tokens Used:\s*(\d+)", blk)
        if mn and mt:
            n, t = int(mn.group(1)), int(mt.group(1))
            if n > best_n:
                best_n, best_tok = n, t
    return best_tok, best_n

def avg_tok_from_report(txt_path):
    tok, n = best_from_cost_report(txt_path)
    return tok / n if n > 0 else 0

def cost_per_ex(avg_tokens, model, avg_input_tok=None, avg_output_tok=None):
    """If input/output split known, use exact pricing; otherwise use blended estimate."""
    if avg_input_tok is not None and avg_output_tok is not None:
        if model == "gpt":
            return (avg_input_tok * 0.50 + avg_output_tok * 1.50) / 1_000_000
        else:
            return (avg_input_tok * 0.80 + avg_output_tok * 0.90) / 1_000_000
    if model == "gpt":
        blended = 0.7 * 0.50 + 0.3 * 1.50  # $0.80/1M
    else:
        blended = 0.7 * 0.80 + 0.3 * 0.90  # $0.83/1M
    return avg_tokens * blended / 1_000_000

# ── Manually provided stats (no JSON data available) ─────────────────────────
MANUAL_STATS = {
    # SchemaAgent: stats from processing logs (250 exercises)
    # Time: mean (no per-file JSON → median unavailable)
    # Tokens: actual input/output split
    "schema_agent": {
        "total_mean_time": 322.61,   # seconds per exercise (mean)
        "avg_input_tok":   19848,
        "avg_output_tok":   9057,
        "avg_total_tok":   28904,
    },
}

# ── Architecture notes ────────────────────────────────────────────────────────
# Multi-LLM-BN:  2 separate LLM calls (generation + probability estimation).
#                Gen step: median 7.30s (~6232 tok).
#                Prob step: median 47.88s (same model + similar prompt → ~6232 tok).
#                Total tokens: ~12464/exercise.
# One-LLM-BN:   1 LLM call (generation) + ILP solver (no extra LLM call).
#                ILP overhead is negligible (<1s, not separately tracked).
#                Time/tokens = same as Single-LLM counterpart.
# Multi-LLM-noBN: 1 LLM call (multi-step generation only, no BN).
ARCH_NOTES = {
    "multi_bn_llama":   "1 LLM call + BN-local + ILP",
    "multi_nobn_gpt":   "1 LLM call",
    "multi_nobn_llama": "1 LLM call",
    "one_bn_llama":     "1 LLM call + BN-local + ILP",
    "one_bn_gpt":       "1 LLM call + BN-local + ILP",
    "single_gpt":       "1 LLM call",
    "single_llama":     "1 LLM call",
    "dsl_tot":          "Multi-step (ToT)",
    "schema_agent":     "Multi-agent",
}

# BN probability step uses SentenceTransformer (local, no API tokens, no API cost).
# The processing_time in probability JSON files is inherited from gen files — NOT real.
# True timing from cost reports (output/cost/cost_report_pro_estimation_*.txt):
BN_PROB_TIME = {
    "multi_bn_llama": 2.41,  # 248 files, avg 2.41s (output/cost/...fewshot_llama.txt)
    "one_bn_llama":   0.63,  # 246 files, avg 0.63s (output/cost/...one_llm_few_shot_llama.txt)
    "one_bn_gpt":     1.36,  # 250 files, avg 1.36s (output/cost/...one_llm_few_shot_gpt.txt)
}

TOKEN_OVERRIDE = {}  # no overrides needed: all BN prob steps are local (0 extra tokens)

# ── Token averages per model from zero-shot cost reports ─────────────────────
# (zero-shot and few-shot use similar prompt sizes; these are good proxies)
AVG_TOK = {
    "multi_gpt":   avg_tok_from_report(path("output/generation/multi-llms/zero-shot-gpt/cost_report_zeroshot_gpt.txt")),
    "multi_llama": avg_tok_from_report(path("output/generation/multi-llms/zero-shot-llama/cost_report_zeroshot_llama.txt")),
    "one_gpt":     avg_tok_from_report(path("output/generation/one-llm/one_llm_zero_shot_gpt/cost_report_onestep_zeroshot_gpt.txt")),
    "one_llama":   avg_tok_from_report(path("output/generation/one-llm/one_llm_zero_shot_llama/cost_report_onestep_zeroshot_llama.txt")),
}
# All BN prob steps use local SentenceTransformer — no extra API tokens needed.

# ── Pipeline definitions ──────────────────────────────────────────────────────
# (label, gen_folder, bn_folder_or_None, token_key_or_folder, model)
PIPELINES = {
    "multi_bn_llama": (
        "Multi-LLM-BN-Llama3",
        path("output/generation/multi-llms/few-shot-llama"),
        path("output/probability/multi-llms/pro_fewshot_llama_0.5_1.0"),
        "multi_llama", "llama",
    ),
    "multi_nobn_gpt": (
        "Multi-LLM-noBN-GPT",
        path("output/generation/multi-llms/few-shot-gpt"),
        None, "multi_gpt", "gpt",
    ),
    "multi_nobn_llama": (
        "Multi-LLM-noBN-Llama3",
        path("output/generation/multi-llms/few-shot-llama"),
        None, "multi_llama", "llama",
    ),
    "one_bn_llama": (
        "One-LLM-BN-Llama3",
        path("output/generation/one-llm/one_llm_few_shot_llama"),
        None, "one_llama", "llama",
    ),
    "one_bn_gpt": (
        "One-LLM-BN-GPT",
        path("output/generation/one-llm/one_llm_few_shot_gpt"),
        None, "one_gpt", "gpt",
    ),
    "single_gpt": (
        "Single-LLM-GPT",
        path("output/generation/one-llm/one_llm_few_shot_gpt"),
        None, "one_gpt", "gpt",
    ),
    "single_llama": (
        "Single-LLM-Llama3",
        path("output/generation/one-llm/one_llm_few_shot_llama"),
        None, "one_llama", "llama",
    ),
    "dsl_tot": (
        "DSL-ToT-DM",
        path("output/DSL-TOT-DM/GPT"),
        None, "dsl_json", "gpt",
    ),
    "schema_agent": (
        "SchemaAgent-GPT",
        path("output/SchemaAgent/data"),
        None, None, "gpt",
    ),
}

GROUPS = [
    ("Our Approach", ["multi_bn_llama"]),
    ("Baselines",    ["single_gpt", "single_llama", "dsl_tot", "schema_agent"]),
    ("Variants",     ["multi_nobn_gpt", "multi_nobn_llama", "one_bn_llama", "one_bn_gpt"]),
]

# ── Hung-file thresholds (files above this are API retries, not real generation) ──
# Llama3: >300s clearly hung; GPT: very stable, no threshold needed; DSL-ToT: keep all
HUNG_THRESH = {
    "multi_bn_llama":   300,   # Llama3 API hung >300s
    "multi_nobn_llama": 300,
    "single_llama":     300,
    "one_bn_llama":     300,
    "dsl_tot":         1000,   # DSL-ToT hung >1000s (36828s, 9751s clearly stuck)
}

# ── Compute ───────────────────────────────────────────────────────────────────
data = {}
print(f"\n{'Pipeline':28s} {'n':>4} {'Gen(avg)':>9} {'BN(avg)':>9} {'Total(avg)':>11}"
      f" {'Avg Tok':>8} {'Est Cost':>10}")
print("-" * 85)

for key, (label, gen_f, bn_f, tok_key, model) in PIPELINES.items():
    if key in MANUAL_STATS:
        ms = MANUAL_STATS[key]
        est_cost = cost_per_ex(
            ms["avg_total_tok"], model,
            ms["avg_input_tok"], ms["avg_output_tok"]
        )
        data[key] = {
            "label": label, "n": 250,
            "gen_avg": ms["total_mean_time"], "bn_avg": 0.0,
            "total_avg": ms["total_mean_time"],
            "avg_tokens": ms["avg_total_tok"], "est_cost": est_cost,
            "model": model, "arch": ARCH_NOTES.get(key, ""),
        }
        print(f"{label:28s} {'250':>4}  {ms['total_mean_time']:>8.2f}s  {0.0:>8.2f}s  "
              f"{ms['total_mean_time']:>10.2f}s  {ms['avg_total_tok']:>7.0f}  ${est_cost:.4f}")
        continue

    thresh = HUNG_THRESH.get(key, None)
    gen_times, n_gen = mean_time(gen_f, thresh) if gen_f and os.path.exists(gen_f) else (0.0, 0)
    gen_avg = gen_times

    bn_avg = BN_PROB_TIME.get(key, 0.0)
    total_avg = gen_avg + bn_avg

    if key in TOKEN_OVERRIDE:
        avg_tokens = TOKEN_OVERRIDE[key]
    elif tok_key == "dsl_json":
        tok_list = get_tokens_json(gen_f)
        avg_tokens = sum(tok_list) / len(tok_list) if tok_list else 0.0
    elif tok_key and tok_key in AVG_TOK:
        avg_tokens = AVG_TOK[tok_key]
    else:
        avg_tokens = 0.0

    est_cost = cost_per_ex(avg_tokens, model)

    data[key] = {
        "label": label, "n": n_gen,
        "gen_avg": gen_avg, "bn_avg": bn_avg, "total_avg": total_avg,
        "avg_tokens": avg_tokens, "est_cost": est_cost, "model": model,
        "arch": ARCH_NOTES.get(key, ""),
    }
    thresh_note = f" (bo treo>{thresh}s)" if thresh else ""
    print(f"{label:28s} {n_gen:>4}  {gen_avg:>8.2f}s  {bn_avg:>8.2f}s  {total_avg:>10.2f}s"
          f"  {avg_tokens:>7.0f}  ${est_cost:.4f}{thresh_note}")

# ── Generate LaTeX ────────────────────────────────────────────────────────────
def fmt(v, dec=2):
    s = f"{v:.{dec}f}".rstrip("0")
    return s + "0" if s.endswith(".") else s

def make_row(d):
    label  = d["label"]
    arch_s = d.get("arch", "")
    gen_s  = fmt(d["gen_avg"])   if d["gen_avg"]   > 0 else "---"
    bn_s   = fmt(d["bn_avg"])    if d["bn_avg"]    > 0 else "---"
    tot_s  = fmt(d["total_avg"]) if d["total_avg"] > 0 else "---"
    tok_s  = f"{d['avg_tokens']:.0f}" if d["avg_tokens"] > 0 else "---"
    cost_s = f"\\${d['est_cost']:.4f}" if d["est_cost"]  > 0 else "---"
    return f"{label} & {arch_s} & {gen_s} & {bn_s} & {tot_s} & {tok_s} & {cost_s} \\\\"

lines = []
lines.append("% Requires: \\usepackage{booktabs}, \\usepackage{colortbl}, \\usepackage{xcolor}")
lines.append("% Time unit: seconds (mean per exercise, hung files excluded). Tokens: avg per exercise.")
lines.append("% Cost: estimated USD per exercise (GPT $0.80/1M blended; LLaMA $0.83/1M blended).")
lines.append("% (*) Token counts estimated from same-model zero-shot runs (similar prompt size).")
lines.append("\\begin{table*}[t]")
lines.append("\\centering")
lines.append("\\caption{\\textcolor{blue}{Efficiency comparison: time, tokens, and estimated cost per exercise.")
lines.append("Time is mean per exercise (Llama3 pipelines exclude API-hung files $>$300s).")
lines.append("Token counts for noBN/single pipelines estimated from same-model zero-shot runs.}}")
lines.append("\\label{tab:cost_analysis_pipelines}")
lines.append("\\scriptsize")
lines.append("\\setlength{\\tabcolsep}{3pt}")
lines.append("\\begin{tabular}{llccccc}")
lines.append("\\toprule")
lines.append("\\textbf{Pipeline}")
lines.append("& \\textbf{Architecture}")
lines.append("& \\textbf{Gen Time (s)} $\\mu$")
lines.append("& \\textbf{BN Time (s)} $\\mu$")
lines.append("& \\textbf{Total Time (s)} $\\mu$")
lines.append("& \\textbf{Avg Tokens}")
lines.append("& \\textbf{Est. Cost (USD)} \\\\")
lines.append("\\midrule")

for gi, (group_name, keys) in enumerate(GROUPS):
    if gi > 0:
        lines.append("\\midrule")
    lines.append("\\rowcolor{gray!15}")
    lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{group_name}}}}} \\\\")
    for key in keys:
        d = data[key]
        lines.append(make_row(d))

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table*}")
lines.append("")

out_path = os.path.join(_HERE, "LaTeX_Tables", "cost_analysis_pipelines.tex")
with open(out_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved: {out_path}")
