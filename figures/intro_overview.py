"""
Small intro figure: high-level concept of Multi-LLMs → ER Model
Suitable for paper introduction section (≈ half-column width)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── colours ──────────────────────────────────────────────────────────
C_INPUT  = "#2C3E50"
C_GPT    = "#2980B9"
C_LLAMA  = "#C0392B"
C_BN     = "#27AE60"
C_ER     = "#7D3C98"
C_ARROW  = "#7F8C8D"
GREY_BG  = "#F2F3F4"

fig, ax = plt.subplots(figsize=(8.5, 3.2))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 10.3)
ax.set_ylim(0, 3.2)
ax.axis("off")

def rbox(ax, x, y, w, h, fc, ec, lw=1.4, alpha=1.0, radius=0.25, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0.0,rounding_size={radius}",
                         facecolor=fc, edgecolor=ec,
                         linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)

def arrow(ax, x0, y, x1, color=C_ARROW):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.6, mutation_scale=13),
                zorder=3)

# ── 1. Input text box ────────────────────────────────────────────────
rbox(ax, 0.15, 0.55, 1.55, 2.1, GREY_BG, C_INPUT, lw=1.8)
ax.text(0.925, 2.82, "Natural Language\nSpecification",
        ha="center", va="center", fontsize=8.5, fontweight="bold",
        color=C_INPUT)
ax.text(0.925, 1.90,
        '"Design a database\n for a pet grooming\n salon that tracks\n'
        ' groomers, pets,\n customers, and\n services..."',
        ha="center", va="center", fontsize=7.8,
        color="#555", style="italic", linespacing=1.55)

# arrow input → LLMs
arrow(ax, 1.72, 1.60, 2.35, color=C_INPUT)

# ── 2. LLM boxes (stacked) ───────────────────────────────────────────
LLM_X = 2.38
llms = [
    (2.85, "GPT-4\n(Few-shot)",  C_GPT),
    (2.05, "GPT-4\n(Zero-shot)", C_GPT),
    (1.25, "Llama-3\n(Few-shot)", C_LLAMA),
    (0.45, "Llama-3\n(Zero-shot)",C_LLAMA),
]
for cy, label, col in llms:
    rbox(ax, LLM_X, cy - 0.27, 1.35, 0.52, col, col, lw=1.5,
         alpha=0.88, radius=0.18)
    ax.text(LLM_X + 0.675, cy, label, ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

ax.text(LLM_X + 0.675, 3.02, "Multiple LLMs",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=C_INPUT)

# bracket / brace left of LLMs
bx = LLM_X - 0.08
ax.annotate("", xy=(bx, 0.42), xytext=(bx, 2.92),
            arrowprops=dict(arrowstyle="-", color=C_ARROW,
                            lw=1.0, connectionstyle="bar,fraction=0.15"))

# arrows LLMs → BN
for cy, _, _ in llms:
    arrow(ax, LLM_X + 1.37, cy, 4.45, color=C_ARROW)

# ── 3. BN / Aggregation box ──────────────────────────────────────────
BN_X = 4.48
rbox(ax, BN_X, 0.40, 1.50, 2.40, C_BN + "18", C_BN, lw=2.0, radius=0.25)
ax.text(BN_X + 0.75, 3.02, "Bayesian Network",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=C_INPUT)

ax.text(BN_X + 0.75, 1.95,
        "Consensus\nProbability\nEstimation",
        ha="center", va="center", fontsize=8.5,
        fontweight="bold", color=C_BN)

# mini prob table
rows = [("PET",     0.82, True),
        ("GROOMER", 0.82, True),
        ("SERVICE", 0.82, True),
        ("SALON",   0.21, False)]
for k, (ent, p, keep) in enumerate(rows):
    ry = 1.55 - k * 0.30
    fc = "#EBF5FB" if keep else "#FDEDEC"
    ec = C_BN    if keep else "#E74C3C"
    rbox(ax, BN_X + 0.05, ry - 0.13, 1.42, 0.26, fc, ec, lw=0.8,
         radius=0.08)
    col = C_INPUT if keep else "#E74C3C"
    sym = "✓" if keep else "✗"
    ax.text(BN_X + 0.10, ry, ent, ha="left", va="center",
            fontsize=8, color=col, fontweight="bold")
    ax.text(BN_X + 0.90, ry, f"p={p:.2f}", ha="left", va="center",
            fontsize=8, color=col)
    ax.text(BN_X + 1.37, ry, sym, ha="center", va="center",
            fontsize=10, color=ec, fontweight="bold")

# arrow BN → ILP
arrow(ax, BN_X + 1.52, 1.60, 6.52, color=C_BN)

# ── 4. ILP box ───────────────────────────────────────────────────────
ILP_X = 6.55
rbox(ax, ILP_X, 0.40, 1.48, 2.40, C_ER + "18", C_ER, lw=2.0, radius=0.25)
ax.text(ILP_X + 0.74, 3.02, "ILP Optimisation",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=C_INPUT)

ax.text(ILP_X + 0.74, 1.92,
        "Optimal\nER Selection",
        ha="center", va="center", fontsize=8.5,
        fontweight="bold", color=C_ER)

# mini entity list
for k, ent in enumerate(["PET", "GROOMER", "CUSTOMER", "SERVICE"]):
    ey = 1.56 - k * 0.30
    rbox(ax, ILP_X + 0.10, ey - 0.125, 1.28, 0.24, GREY_BG, C_ER,
         lw=0.9, radius=0.08)
    ax.text(ILP_X + 0.74, ey, ent, ha="center", va="center",
            fontsize=8, color=C_ER, fontweight="bold")

# arrow ILP → ER Schema
arrow(ax, ILP_X + 1.50, 1.60, 8.52, color=C_ER)

# ── 5. Final ER Schema box ───────────────────────────────────────────
ER_X = 8.55
rbox(ax, ER_X, 0.35, 1.35, 2.50, C_ER + "12", C_ER, lw=2.0, radius=0.25)
ax.text(ER_X + 0.675, 3.02, "ER Schema",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color=C_INPUT)

# mini ER diagram: 4 entity nodes + lines
ent_pos = {
    "PET":      (ER_X + 0.33, 2.55),
    "GROOMER":  (ER_X + 1.02, 2.55),
    "CUSTOMER": (ER_X + 0.33, 1.65),
    "SERVICE":  (ER_X + 1.02, 1.65),
}
# relationships lines
for e1, e2 in [("PET","SERVICE"), ("GROOMER","SERVICE"),
               ("CUSTOMER","PET"), ("CUSTOMER","SERVICE")]:
    x1, y1 = ent_pos[e1]
    x2, y2 = ent_pos[e2]
    ax.plot([x1, x2], [y1, y2], color=C_ER, lw=1.1,
            alpha=0.5, zorder=1)

for ent, (ex, ey) in ent_pos.items():
    rbox(ax, ex - 0.32, ey - 0.175, 0.64, 0.34, "white", C_ER,
         lw=1.3, radius=0.10, zorder=3)
    ax.text(ex, ey, ent, ha="center", va="center",
            fontsize=6.8, fontweight="bold", color=C_ER, zorder=4)

# bottom note
ax.text(ER_X + 0.675, 0.95,
        "Entities\nAttributes\nRelationships",
        ha="center", va="center", fontsize=7.5,
        color="#555", linespacing=1.4)

# ── Step labels at bottom ────────────────────────────────────────────
for txt, cx, col in [
    ("① Input",       0.925, C_INPUT),
    ("② Generation",  3.055, C_INPUT),
    ("③ Consensus",   5.230, C_BN),
    ("④ Optimise",    7.290, C_ER),
    ("⑤ Output",      9.225, C_ER),
]:
    ax.text(cx, 0.16, txt, ha="center", va="center",
            fontsize=8, color=col, fontweight="bold")

plt.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    p = os.path.join(OUT_DIR, f"intro_overview.{ext}")
    plt.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}")
plt.close()
