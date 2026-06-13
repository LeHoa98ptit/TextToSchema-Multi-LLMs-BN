"""
Line plot: Inference Time vs Input Text Length — baselines + ours.
x-axis : word-count quantile bins (~50 examples each)
y-axis : median inference time per bin  (shaded = IQR)  — log scale
Lines  : Single-LLM-GPT, Single-LLM-Llama3, DSL-ToT-DM, Multi-LLM-BN [ours]
         (SchemaAgent excluded: no per-exercise timing data)
Ours   : generation time (output/generation/multi-llms/few-shot-llama)
         + BN probability time (output/probability/multi-llms/pro_fewshot_llama_0.5_1.0)
Outputs: figures/time_vs_length.png
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
})

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT, "dataset/Datasets/Full-Dataset/input")
OUT_DIR   = os.path.join(ROOT, "plots_extend/figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Pipeline definitions ───────────────────────────────────────────────────────
# (label, color, linestyle, marker, json_folder, outlier_cap_percentile)
PIPELINES = [
    ("Single-LLM-Llama3 [15]", "#1565C0", "-",  "s",
     os.path.join(ROOT, "output/generation/one-llm/one_llm_zero_shot_llama"), 98),
    ("Single-LLM-GPT [15]",    "#5C9BD6", "--", "o",
     os.path.join(ROOT, "output/generation/one-llm/one_llm_zero_shot_gpt"),   98),
    ("DSL-ToT-DM [16]",        "#6A1B9A", "-.", "^",
     os.path.join(ROOT, "output/DSL-TOT-DM/GPT"),                            90),
]

# ── "Ours" pipeline: sum generation + BN probability per exercise ──────────────
OURS_GEN_FOLDER  = os.path.join(ROOT, "output/generation/multi-llms/few-shot-llama")
OURS_PROB_FOLDER = os.path.join(ROOT, "output/probability/multi-llms/pro_fewshot_llama_0.5_1.0")
OURS_LABEL       = "Multi-LLM-BN [ours]"
OURS_COLOR       = "#E65100"
OURS_CAP_PCT     = 90

N_BINS = 8


def word_count(num):
    p = os.path.join(INPUT_DIR, f"{num}.txt")
    return len(open(p).read().split()) if os.path.exists(p) else None


def load_times(folder, cap_pct):
    pairs = {}
    for fname in os.listdir(folder):
        if not fname.endswith(".json") or fname.startswith("."):
            continue
        try:
            num = int(fname.replace(".json", ""))
            with open(os.path.join(folder, fname)) as f:
                d = json.load(f)
            t  = d.get("processing_time")
            wc = word_count(num)
            if t and wc:
                pairs[num] = (wc, t)
        except Exception:
            pass
    times = [v[1] for v in pairs.values()]
    cap   = np.percentile(times, cap_pct)
    return {k: v for k, v in pairs.items() if v[1] <= cap}


def load_times_summed(gen_folder, prob_folder, cap_pct):
    """Sum generation + probability timing per exercise for 'ours' pipeline."""
    gen_t = {}
    for fname in os.listdir(gen_folder):
        if not fname.endswith(".json") or fname.startswith("."): continue
        try:
            num = int(fname.replace(".json", ""))
            d = json.load(open(os.path.join(gen_folder, fname)))
            t = d.get("processing_time")
            if t: gen_t[num] = t
        except Exception: pass

    pairs = {}
    for fname in os.listdir(prob_folder):
        if not fname.endswith(".json") or fname.startswith("."): continue
        try:
            num = int(fname.replace(".json", ""))
            d = json.load(open(os.path.join(prob_folder, fname)))
            t = d.get("processing_time")
            wc = word_count(num)
            if t and wc and num in gen_t:
                pairs[num] = (wc, gen_t[num] + t)
        except Exception: pass

    times = [v[1] for v in pairs.values()]
    cap   = np.percentile(times, cap_pct)
    return {k: v for k, v in pairs.items() if v[1] <= cap}


# ── Load all data ─────────────────────────────────────────────────────────────
raw = {}
all_wc = []
for label, color, ls, marker, folder, cap_pct in PIPELINES:
    raw[label] = load_times(folder, cap_pct)
    all_wc.extend([v[0] for v in raw[label].values()])

raw[OURS_LABEL] = load_times_summed(OURS_GEN_FOLDER, OURS_PROB_FOLDER, OURS_CAP_PCT)
all_wc.extend([v[0] for v in raw[OURS_LABEL].values()])

# ── Quantile bin edges ────────────────────────────────────────────────────────
all_wc_sorted = sorted(set(all_wc))
quantiles = np.percentile(all_wc_sorted, np.linspace(0, 100, N_BINS + 1))
quantiles[0] -= 1

bin_labels = [f"{int(quantiles[i])+1}–{int(quantiles[i+1])}w"
              for i in range(N_BINS)]

def get_bin(wc):
    for i in range(N_BINS):
        if quantiles[i] < wc <= quantiles[i + 1]:
            return i
    return N_BINS - 1


# ── Aggregate per bin ─────────────────────────────────────────────────────────
all_labels = [lbl for lbl, *_ in PIPELINES] + [OURS_LABEL]
all_styles = {lbl: (color, ls, marker)
              for lbl, color, ls, marker, *_ in PIPELINES}
all_styles[OURS_LABEL] = (OURS_COLOR, "-", "D")

stats = {}
for label in all_labels:
    bin_times = defaultdict(list)
    for wc, t in raw[label].values():
        bin_times[get_bin(wc)].append(t)
    stats[label] = [
        (np.median(v), np.percentile(v, 25), np.percentile(v, 75), len(v))
        if (v := bin_times[i]) else (np.nan, np.nan, np.nan, 0)
        for i in range(N_BINS)
    ]

# ── Print bin distribution ─────────────────────────────────────────────────────
print("Bin distribution (n per bin):")
for i, bl in enumerate(bin_labels):
    counts = [stats[lbl][i][3] for lbl in all_labels]
    print(f"  {bl:>12}:  " + "  ".join(f"{c:>4}" for c in counts))

# ── Plot ──────────────────────────────────────────────────────────────────────
x = np.arange(N_BINS)
fig, ax = plt.subplots(figsize=(6.5, 4.5))

def rolling_avg(arr, window=3):
    """Centered rolling average; edge bins use available neighbours."""
    out = arr.copy()
    for i in range(len(arr)):
        lo = max(0, i - window // 2)
        hi = min(len(arr), i + window // 2 + 1)
        valid_vals = arr[lo:hi][~np.isnan(arr[lo:hi])]
        out[i] = np.mean(valid_vals) if len(valid_vals) else np.nan
    return out

for label in all_labels:
    color, ls, marker = all_styles[label]
    lw = 2.5 if label == OURS_LABEL else 2
    row     = stats[label]
    medians = np.array([r[0] for r in row], dtype=float)
    q25     = np.array([r[1] for r in row], dtype=float)
    q75     = np.array([r[2] for r in row], dtype=float)

    # Apply rolling average to smooth ours line (high variance BN step)
    if label == OURS_LABEL:
        medians = rolling_avg(medians, window=3)
        q25     = rolling_avg(q25,     window=3)
        q75     = rolling_avg(q75,     window=3)

    valid = ~np.isnan(medians)
    ax.plot(x[valid], medians[valid],
            color=color, linestyle=ls, linewidth=lw,
            marker=marker, markersize=5 if label != OURS_LABEL else 6,
            label=label,
            zorder=3 if label == OURS_LABEL else 2)
    ax.fill_between(x[valid], q25[valid], q75[valid],
                    color=color, alpha=0.15 if label == OURS_LABEL else 0.12)

ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels(bin_labels, fontsize=10)
ax.set_xlabel("Input Text Length (word count bin)")
ax.set_ylabel("Median Inference Time (s)  [log scale]")
ax.set_title("Inference Time vs Input Text Length\n(SchemaAgent excluded — no per-example data)")
ax.legend(loc="upper left", framealpha=0.7)
ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# Annotate n per bin below x axis (using first pipeline)
first_label = PIPELINES[0][0]
for i, bl in enumerate(bin_labels):
    n_ex = stats[first_label][i][3]
    ax.text(i, ax.get_ylim()[0] * 0.75, f"n≈{n_ex}",
            ha="center", va="top", fontsize=7.5, color="gray")

plt.tight_layout()
out = os.path.join(OUT_DIR, "time_vs_length.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Pipeline':<28}", end="")
for bl in bin_labels:
    print(f"  {bl:>10}", end="")
print()
print("-" * (28 + N_BINS * 12))
for label in all_labels:
    print(f"{label:<28}", end="")
    for r in stats[label]:
        if np.isnan(r[0]):
            print(f"  {'—':>10}", end="")
        else:
            print(f"  {r[0]:>9.1f}s", end="")
    print()
