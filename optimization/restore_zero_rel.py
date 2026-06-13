"""
Restore exercises in add/multi that were degraded by no_isolated=True.
When both prob AND gen have 0 rels, the constraint makes ILP infeasible → E=1 R=0 fallback.
These exercises are better served by the original opt result (E=N R=0).
Strategy: for any add/multi exercise with E=1 R=0, restore from original opt.
"""
import os, json, shutil

project_root = "/Volumes/T9/PhD_These/Development/Multi-LLMs-WithBN-ERSchema-Generation"
OPT_BASE  = os.path.join(project_root, "output/optimization/multi-llms")
ADD_MULTI = os.path.join(project_root, "output/add/multi")

FOLDERS = [
    "opt_fewshot_gpt_-0.5_1.0-(0.0-0.0-0.0)",
    "opt_fewshot_gpt_0.5_1.0-(1.2-1.0-1.0)",
    "opt_fewshot_gpt_1.0_3.0-(1.5-2.0-2.0)",
    "opt_fewshot_llama-0.5_1.0-(0.5-0.5-0.5)",
    "opt_fewshot_llama_1.0_3.0-(1.5-2.0-2.0)",
    "opt_zeroshot_gpt_-0.5_1.0-(0.0-0.0-0.0)",
    "opt_zeroshot_gpt_0.5_1.0-(1.2-1.0-1.0)",
    "opt_zeroshot_gpt_1.0_3.0-(1.5-2.0-2.0)",
    "opt_zeroshot_llama-0.5_1.0-(0.5-0.5-0.5)",
]

total_restored = total_already_ok = total_no_orig = 0

for opt_name in FOLDERS:
    add_dir  = os.path.join(ADD_MULTI, opt_name)
    orig_dir = os.path.join(OPT_BASE,  opt_name)
    if not os.path.exists(add_dir): continue

    restored = []
    for f in sorted(os.listdir(add_dir)):
        if not f.endswith('.json') or f.startswith('._'): continue
        add_path  = os.path.join(add_dir,  f)
        orig_path = os.path.join(orig_dir, f)

        d = json.load(open(add_path))
        ents = d.get('entity', [])
        rels = d.get('relationship', [])

        # Only restore if degraded: 1 entity and 0 rels
        if len(ents) == 1 and len(rels) == 0:
            if not os.path.exists(orig_path):
                total_no_orig += 1
                continue
            orig = json.load(open(orig_path))
            orig_ents = orig.get('entity', [])
            orig_rels  = orig.get('relationship', [])
            # Only restore if original has more entities
            if len(orig_ents) > 1:
                shutil.copy2(orig_path, add_path)
                restored.append(f.replace('.json',''))
                total_restored += 1
            else:
                total_already_ok += 1

    if restored:
        print(f"{opt_name}: restored {len(restored)} exercises: {restored}")

print(f"\nTotal restored: {total_restored}  no_orig: {total_no_orig}")

# Final quality check
print("\nFinal quality per folder:")
for opt_name in FOLDERS:
    add_dir = os.path.join(ADD_MULTI, opt_name)
    if not os.path.exists(add_dir): continue
    files = [f for f in os.listdir(add_dir) if f.endswith('.json') and not f.startswith('._')]
    n_iso = n_noattr = n_1ent = 0
    for fname in files:
        d = json.load(open(os.path.join(add_dir, fname)))
        ents  = d.get('entity', [])
        attrs = d.get('attribut', d.get('attribute', {}))
        rels  = d.get('relationship', [])
        conn  = {r.get('entity_1') for r in rels} | {r.get('entity_2') for r in rels}
        if any(e not in conn for e in ents): n_iso += 1
        if any(not (attrs.get(e) if isinstance(attrs, dict) else []) for e in ents): n_noattr += 1
        if len(ents) == 1: n_1ent += 1
    print(f"  {opt_name:<55}  iso={n_iso:3d}  noattr={n_noattr:3d}  1ent={n_1ent}  total={len(files)}")
