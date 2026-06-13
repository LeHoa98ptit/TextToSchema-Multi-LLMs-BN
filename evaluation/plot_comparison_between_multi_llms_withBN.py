import os
import csv
import re
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

pattern = re.compile(r'opt_(fewshot|zeroshot)_(gpt|llama)[_]?(-?[\d.]+_[\d.]+)')

def read_average(filepath):
    with open(filepath, 'r') as f:
        for row in csv.reader(f):
            if row[0] == 'AVERAGE':
                return float(row[3]), float(row[6]), float(row[9])
    return None

data = []
for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.endswith('.csv'):
        continue
    m = pattern.match(fname)
    if not m:
        continue
    prompt, llm, config_raw = m.group(1), m.group(2), m.group(3)
    avg = read_average(os.path.join(RESULTS_DIR, fname))
    if avg:
        data.append({
            'prompt': prompt,
            'llm': llm.upper(),
            'config': config_raw.replace('_', '+'),
            'ent': avg[0], 'attr': avg[1], 'rel': avg[2]
        })

print(f"Loaded {len(data)} files")
for d in data:
    print(f"  {d['prompt']:9} | {d['llm']:5} | {d['config']:8} | Ent={d['ent']:.3f} Attr={d['attr']:.3f} Rel={d['rel']:.3f}")

configs = ['-0.5+1.0', '0.5+1.0', '1.0+3.0']
metrics = [('ent', 'Entity'), ('attr', 'Attribute'), ('rel', 'Relationship')]

def avg_for(key, **filters):
    vals = [d[key] for d in data if all(d[k] == v for k, v in filters.items())]
    return np.mean(vals) if vals else 0.0

x = np.arange(len(configs))
w = 0.35

BLUE   = '#4472C4'
RED    = '#C0392B'
ORANGE = '#E67E22'
GREEN  = '#70AD47'
LBLUE  = '#5B9BD5'

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('F1-Score Comparison - Multi-LLMs with BN', fontsize=14, fontweight='bold', y=1.01)

col_titles = ['Config Comparison', 'LLM Comparison', 'Prompt Type Comparison']

for row, (mk, mname) in enumerate(metrics):
    for col in range(3):
        ax = axes[row][col]
        ax.set_ylim(0, 1)
        ax.set_ylabel('F1-Score' if col == 0 else '')
        ax.set_xticks(x)
        ax.set_xticklabels(configs)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.set_title(f'{mname} - {col_titles[col]}', fontsize=10)

        if col == 0:
            vals = [avg_for(mk, config=c) for c in configs]
            for i, (v, clr) in enumerate(zip(vals, [LBLUE, ORANGE, GREEN])):
                ax.bar(x[i], v, width=0.5, color=clr, zorder=3)

        elif col == 1:
            gpt_vals   = [avg_for(mk, config=c, llm='GPT')   for c in configs]
            llama_vals = [avg_for(mk, config=c, llm='LLAMA') for c in configs]
            ax.bar(x - w/2, gpt_vals,   width=w, label='GPT',   color=BLUE, zorder=3)
            ax.bar(x + w/2, llama_vals, width=w, label='Llama', color=RED,  zorder=3)
            ax.legend(fontsize=8)

        else:
            zero_vals = [avg_for(mk, config=c, prompt='zeroshot') for c in configs]
            few_vals  = [avg_for(mk, config=c, prompt='fewshot')  for c in configs]
            ax.bar(x - w/2, zero_vals, width=w, label='Zero-shot', color=BLUE,   zorder=3)
            ax.bar(x + w/2, few_vals,  width=w, label='Few-shot',  color=ORANGE, zorder=3)
            ax.legend(fontsize=8)

plt.tight_layout()
out = ""
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out}")
plt.show()
