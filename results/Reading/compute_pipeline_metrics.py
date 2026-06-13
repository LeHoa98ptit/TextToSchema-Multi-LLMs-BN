"""
Compute structural quality metrics for each pipeline folder and generate
results/Reading/LaTeX_Tables/structural_metrics_pipelines.tex
"""
import os, json, statistics

# ── Load evaluator ────────────────────────────────────────────────────────────
_HERE = os.path.abspath(os.path.dirname(__file__))
ROOT  = os.path.abspath(os.path.join(_HERE, "../../.."))

_ns = {}
with open(os.path.join(ROOT, "experiments/Evaluation/Metric_Evaluation")) as _f:
    exec(_f.read(), _ns)
LLMERRawEvaluator = _ns["LLMERRawEvaluator"]

# ── Pipeline definitions (key, folder) ─────────────────────────────────────
PIPELINES = {
    "multi_bn_llama":    "output/add/multi/opt_fewshot_llama_0.5_1.0-(1.2--0.5-1.0)",
    "multi_nobn_gpt":    "output/generation/multi-llms/few-shot-gpt",
    "multi_nobn_llama":  "output/generation/multi-llms/few-shot-llama",
    "one_bn_llama":      "output/add/one-llm/opt_one_fewshot_llama_0.5_1.0-(1.2-1.0-1.0)",
    "one_bn_gpt":        "output/add/one-llm/opt_one_fewshot_gpt_0.5_1.0-(1.2-1.0-1.0)",
    "text-to-erd_gpt":        "output/generation/Text-To-ERD/gpt",
    "text-to-erd-llama":      "output/generation/Text-To-ERD/llama",
    "dsl_tot":           "output/DSL-TOT-DM/GPT",
    "schema_agent":      "output/SchemaAgent/data",
    "multi_bn_llama_nowd": "ablation/optimization_ablation_hard/output/multi-llms/few-shot-llama",
    "one_bn_llama_nowd": "ablation/ablation_one_llm/optimization"
}

# Table layout: groups with header and (key, display_label, bold) rows
GROUPS = [
    ("Our Approach", [
        ("multi_bn_llama",   "\\textbf{Multi-LLM-BN-Llama3}", True),
    ]),
    ("Baselines", [
        ("text-to-erd_gpt",       "TEXT-To-ERD-GPT",   False),
        ("text-to-erd-llama",     "TEXT-To-ERD-Llama3", False),
        ("dsl_tot",          "DSL-ToT-DM",        False),
        ("schema_agent",     "SchemaAgent-GPT",   False),
    ]),
    ("Variants", [
        ("multi_bn_llama_nowd", "Multi-LLM-BN-Llama3 (noWiki)", False),
        ("multi_nobn_gpt",   "Multi-LLM-noBN-GPT",   False),
        ("multi_nobn_llama", "Multi-LLM-noBN-Llama3", False),
        ("one_bn_llama",     "One-LLM-BN-Llama3",    False),
        ("one_bn_llama_nowd", "One-LLM-BN-Llama3 (noWiki)", False),
        ("one_nobn_llama",       "One-LLM-noBN-Llama3",       False),
    ]),
]


def compute_folder(folder):
    results = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".json") or fname.startswith("."):
            continue
        try:
            schema = json.load(open(os.path.join(folder, fname)))
            if "attribute" in schema and "attribut" not in schema:
                schema["attribut"] = schema["attribute"]
            m = LLMERRawEvaluator(schema).evaluate()
            results.append(m)
        except Exception:
            pass
    return results


def summarize(results):
    if not results:
        return None

    def avg(key, subkey):
        vals = [float(r[key][subkey]) for r in results
                if key in r and subkey in r[key]]
        return statistics.mean(vals) if vals else 0.0

    E    = avg("basic_counts", "entities")
    R    = avg("basic_counts", "relationships")
    A    = avg("basic_counts", "attributes")
    Tot  = avg("basic_counts", "total_constructs")
    RE   = avg("complexity_metrics", "R_per_E")
    NM   = avg("redundancy_metrics", "N_M_relationships_percentage")
    AE   = avg("complexity_metrics", "A_per_E")
    Dup  = avg("redundancy_metrics", "duplicate_relationships_percentage")
    Sim  = avg("redundancy_metrics", "similar_entity_pairs_percentage")
    EntA = avg("well_formedness_metrics", "percent_entities_with_attrs")
    VC   = avg("well_formedness_metrics", "percent_valid_cardinality")
    NI   = statistics.mean([
        100.0 - r["well_formedness_metrics"]["isolated_entities_percentage"]
        for r in results
    ])
    return (E, R, A, Tot, RE, NM, AE, Dup, Sim, EntA, VC, NI)


def fmt(v):
    """Format a float: 2 decimal places, strip trailing zeros, keep ≥1 decimal."""
    s = f"{v:.2f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return s


# ── Compute ───────────────────────────────────────────────────────────────────
data = {}
for key, rel_folder in PIPELINES.items():
    folder = os.path.join(ROOT, rel_folder)
    r = compute_folder(folder)
    s = summarize(r)
    data[key] = s
    if s:
        E, R, A, Tot, RE, NM, AE, Dup, Sim, EntA, VC, NI = s
        print(f"{key:20s} n={len(r):3d}  "
              f"{E:.2f}|{R:.2f}|{A:.2f}|{Tot:.2f}  "
              f"{RE:.2f}|{NM:.1f}|{AE:.2f}  "
              f"{Dup:.2f}|{Sim:.2f}  "
              f"{EntA:.2f}|{VC:.2f}|{NI:.2f}")

# ── Generate LaTeX ────────────────────────────────────────────────────────────
def make_data_row(label, s):
    E, R, A, Tot, RE, NM, AE, Dup, Sim, EntA, VC, NI = s
    basic = f"{fmt(E)} | {fmt(R)} | {fmt(A)} | {fmt(Tot)}"
    compl = f"{fmt(RE)} | {fmt(NM)} | {fmt(AE)}"
    redun = f"{fmt(Dup)} | {fmt(Sim)}"
    wellf = f"{fmt(EntA)} | {fmt(VC)} | {fmt(NI)}"
    return f"{label} & {basic} & {compl} & {redun} & {wellf} \\\\"


lines = []
lines.append("% Requires: \\usepackage{booktabs}, \\usepackage{colortbl}, \\usepackage{xcolor}")
lines.append("\\begin{table*}[t]")
lines.append("\\centering")
lines.append("\\caption{\\textcolor{blue}{ER model quality metrics for pipeline comparison.}}")
lines.append("\\label{tab:structural_metrics_pipelines}")
lines.append("\\scriptsize")
lines.append("\\setlength{\\tabcolsep}{2.5pt}")
lines.append("\\begin{tabular}{lcccc}")
lines.append("\\toprule")
lines.append("\\textbf{Pipeline}")
lines.append("& \\textbf{Basic Counts}")
lines.append("& \\textbf{Complexity}")
lines.append("& \\textbf{Redundancy (\\%)} $\\downarrow$")
lines.append("& \\textbf{Well-Formedness (\\%)} $\\uparrow$ \\\\")
lines.append("\\midrule")
lines.append("& E | R | A | Total")
lines.append("& R/E | N:M (\\%) | A/E")
lines.append("& Dup | Similar")
lines.append("& Ent w/ A | Valid Card | Not Isolated \\\\")

for gi, (group_name, group_rows) in enumerate(GROUPS):
    if gi > 0:
        lines.append("\\midrule")
    lines.append("\\rowcolor{gray!15}")
    lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{group_name}}}}} \\\\")
    for key, display_label, _ in group_rows:
        s = data.get(key)
        if s:
            lines.append(make_data_row(display_label, s))

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table*}")
lines.append("")

out_path = os.path.join(_HERE, "LaTeX_Tables", "structural_metrics_pipelines.tex")
with open(out_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved: {out_path}")
