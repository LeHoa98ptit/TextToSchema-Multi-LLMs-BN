import os, csv
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
F1   = os.path.join(ROOT, "results/F1Score")
OUT  = os.path.join(ROOT, "results/figures/compare-all-pipelines")

# ── Get exercise list from DSL-TOT-DM ────────────────────────────────────────
def get_exercise_set(csv_path):
    exs = set()
    for r in csv.DictReader(open(csv_path)):
        if r["Exercise"].startswith("Ex "):
            exs.add(int(r["Exercise"].split()[1]))
    return exs

llama_exs    = get_exercise_set(os.path.join(F1, "DSL-TOT-DM/LLama.csv"))
together_exs = get_exercise_set(os.path.join(F1, "DSL-TOT-DM/Together.csv"))

# ── Compute average on a specific subset ─────────────────────────────────────
def avg_on_subset(csv_path, subset=None):
    """subset=None → all exercises"""
    ent, att, rel = [], [], []
    for r in csv.DictReader(open(csv_path)):
        if not r["Exercise"].startswith("Ex "):
            continue
        num = int(r["Exercise"].split()[1])
        if subset is not None and num not in subset:
            continue
        ent.append(float(r["Ent_F1"]))
        att.append(float(r["Attr_F1"]))
        rel.append(float(r["Rel_F1"]))
    if not ent:
        return None
    e, a, r = np.mean(ent), np.mean(att), np.mean(rel)
    return {"n": len(ent), "Ent": e, "Att": a, "Rel": r, "Avg": (e+a+r)/3}

# ── Methods to compare ───────────────────────────────────────────────────────
OUR_CSV     = os.path.join(F1, "Multi-LLMs-withBN/opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0).csv")
DSL_LLAMA   = os.path.join(F1, "DSL-TOT-DM/LLama.csv")
DSL_TOGETHER= os.path.join(F1, "DSL-TOT-DM/Together.csv")

METHODS = [
    ("Multi-LLM BN-LLAMA (ours)",   OUR_CSV),
    ("DSL-ToT-DM LLama",            DSL_LLAMA),
    ("DSL-ToT-DM Together",         DSL_TOGETHER),
]

subsets = {
    "All 250 exercises":    None,
    "DSL-LLama subset (36 exercises)":     llama_exs,
    "DSL-Together subset (118 exercises)": together_exs,
}

# ── Print results and save file ──────────────────────────────────────────────
lines = []
header = f"{'Method':<35} {'Subset':<35} {'N':>4}  {'Ent':>6} {'Att':>6} {'Rel':>6} {'Avg':>6}"
lines.append(header)
lines.append("-" * len(header))

for subset_name, subset in subsets.items():
    lines.append(f"\n[{subset_name}]")
    for method_name, csv_path in METHODS:
        if not os.path.exists(csv_path):
            lines.append(f"  {method_name:<33} MISSING")
            continue
        v = avg_on_subset(csv_path, subset)
        if v is None:
            lines.append(f"  {method_name:<33} NO DATA for this subset")
            continue
        row = (f"  {method_name:<33} {v['n']:>4}  "
               f"{v['Ent']:>6.3f} {v['Att']:>6.3f} {v['Rel']:>6.3f} {v['Avg']:>6.3f}")
        lines.append(row)

output = "\n".join(lines)
print(output)

out_path = os.path.join(OUT, "fair_subset_comparison.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(output + "\n")
print(f"\nSaved: {out_path}")
