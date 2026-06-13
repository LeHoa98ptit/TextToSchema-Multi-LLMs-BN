"""
Figure: Multi-LLMs pipeline running example (Exercise 251 – Pet Grooming Salon)
Saved to: figures/pipeline_example.pdf  (and .png)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np, os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── colour palette ───────────────────────────────────────────────────
C_GPT   = "#4472C4"
C_LLAMA = "#C0392B"
C_BN    = "#27AE60"
C_ILP   = "#8E44AD"
C_INPUT = "#2C3E50"
C_GREY  = "#ECF0F1"
C_HIGH  = "#F9EBEA"   # light red for high-prob rows
C_GOLD  = "#F39C12"

# ── real data from exercise 251 ──────────────────────────────────────
INPUT_TEXT = (
    "This is a pet grooming salon management system.\n"
    "The shop needs to manage pet groomers, customers,\n"
    "and their pets' information. Groomer information\n"
    "includes name, specialization, years of experience,\n"
    "and rating. Pets need to have basic information\n"
    "recorded including breed, age, weight, and fur\n"
    "characteristics. Each grooming service needs to\n"
    "record the service items, duration, pricing, and\n"
    "customer ratings."
)

LLM_OUTPUTS = [
    {
        "label": "GPT-4\n(Few-shot)",
        "color": C_GPT,
        "entities": ["PET", "GROOMER", "CUSTOMER", "SERVICE"],
        "rels": [("CUSTOMER","PET","1:N"),
                 ("CUSTOMER","SERVICE","1:N"),
                 ("PET","SERVICE","1:N"),
                 ("GROOMER","SERVICE","1:N")],
    },
    {
        "label": "GPT-4\n(Zero-shot)",
        "color": C_GPT,
        "entities": ["PET", "GROOMER", "CUSTOMER", "SERVICE"],
        "rels": [("CUSTOMER","PET","1:N"),
                 ("PET","SERVICE","1:N"),
                 ("GROOMER","SERVICE","1:N"),
                 ("CUSTOMER","SERVICE","1:N")],
    },
    {
        "label": "Llama-3\n(Few-shot)",
        "color": C_LLAMA,
        "entities": ["PET", "GROOMER", "CUSTOMER", "SERVICE"],
        "rels": [("PET","CUSTOMER","N:M"),
                 ("PET","SERVICE","N:M"),
                 ("GROOMER","SERVICE","N:M"),
                 ("CUSTOMER","GROOMER","N:M")],
    },
    {
        "label": "Llama-3\n(Zero-shot)",
        "color": C_LLAMA,
        "entities": ["PET_GROOMING\nSALON", "PET_GROOMER",
                     "CUSTOMER", "PET", "GROOM\nSERVICE"],
        "rels": [("SALON","GROOMER","1:N"),
                 ("CUSTOMER","PET","1:N"),
                 ("GROOMER","SERVICE","1:N"),
                 ("PET","SERVICE","1:N"),
                 ("CUSTOMER","SERVICE","1:N")],
        "extra": True,   # has extra entity
    },
]

BN_ROWS = [
    # (entity display, p_entity, kept)
    ("PET",                 0.818, True),
    ("GROOMER",             0.818, True),
    ("CUSTOMER",            0.818, True),
    ("SERVICE",             0.818, True),
    ("PET_GROOMING_SALON",  0.205, False),
]

FINAL_ENTITIES = ["PET", "GROOMER", "CUSTOMER", "SERVICE"]
FINAL_RELS     = [
    ("CUSTOMER",  "PET",     "1:N"),
    ("CUSTOMER",  "SERVICE", "1:N"),
    ("PET",       "SERVICE", "1:N"),
    ("GROOMER",   "SERVICE", "1:N"),
]
FINAL_ATTRS = {
    "PET":      ["pet_id","breed","age","weight","fur_char."],
    "GROOMER":  ["groomer_id","name","speciali.","exp.","rating"],
    "CUSTOMER": ["customer_id","name","phone"],
    "SERVICE":  ["service_id","items","duration","price","rating"],
}


# ────────────────────────────────────────────────────────────────────
def rounded_box(ax, x, y, w, h, color, alpha=1.0, lw=1.5,
                edgecolor="white", zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.01",
                         facecolor=color, edgecolor=edgecolor,
                         linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box


def draw_arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.5):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14))


def draw_entity_box(ax, cx, cy, name, color, width=0.13, height=0.055,
                    fontsize=7.5, bold=False):
    x, y = cx - width/2, cy - height/2
    rounded_box(ax, x, y, width, height, color, alpha=0.25,
                edgecolor=color, lw=1.5)
    ax.text(cx, cy, name, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            color=color)


# ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.patch.set_facecolor("white")

# We use a single large axes spanning everything and draw manually
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# ── Section headers helper ──────────────────────────────────────────
def section_header(ax, x, y, text, color, width=0.18, height=0.032):
    rounded_box(ax, x - width/2, y - height/2, width, height, color,
                alpha=1.0, edgecolor=color, zorder=3)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=4)

# ═══════════════════════════════════════════════════════════════════
# 0.  TITLE
# ═══════════════════════════════════════════════════════════════════
ax.text(0.5, 0.975, "Multi-LLMs with BN Pipeline — Running Example (Exercise 251)",
        ha="center", va="top", fontsize=12, fontweight="bold", color=C_INPUT)

# ═══════════════════════════════════════════════════════════════════
# 1.  INPUT TEXT  (left column, top)
# ═══════════════════════════════════════════════════════════════════
INPUT_X, INPUT_Y = 0.085, 0.62
INPUT_W, INPUT_H = 0.155, 0.30

rounded_box(ax, INPUT_X - INPUT_W/2, INPUT_Y - INPUT_H/2,
            INPUT_W, INPUT_H, C_INPUT, alpha=0.06,
            edgecolor=C_INPUT, lw=2)

section_header(ax, INPUT_X, INPUT_Y + INPUT_H/2 + 0.017,
               "Input Text", C_INPUT, width=0.13)

ax.text(INPUT_X, INPUT_Y + 0.02, INPUT_TEXT,
        ha="center", va="center", fontsize=7.2,
        linespacing=1.55, color="#2C3E50",
        family="monospace")

# ═══════════════════════════════════════════════════════════════════
# 2.  PHASE 1 — Four LLM boxes  (centre-left)
# ═══════════════════════════════════════════════════════════════════
section_header(ax, 0.355, 0.955, "Phase 1 — LLM Generation", C_INPUT, width=0.24)

LLM_XS = [0.245, 0.345, 0.445, 0.545]
LLM_Y_TOP = 0.88

for i, llm in enumerate(LLM_OUTPUTS):
    bx = LLM_XS[i]
    by_top = LLM_Y_TOP
    bw, bh = 0.09, 0.74

    # Outer box
    rounded_box(ax, bx - bw/2, by_top - bh, bw, bh,
                llm["color"], alpha=0.06, edgecolor=llm["color"], lw=2)

    # Header
    rounded_box(ax, bx - bw/2, by_top - 0.068, bw, 0.065,
                llm["color"], alpha=0.85, edgecolor=llm["color"], lw=1)
    ax.text(bx, by_top - 0.035, llm["label"],
            ha="center", va="center", fontsize=8,
            fontweight="bold", color="white")

    # Entities
    ent_label_y = by_top - 0.10
    ax.text(bx, ent_label_y, "Entities", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=llm["color"])

    for j, ent in enumerate(llm["entities"]):
        ey = ent_label_y - 0.055 - j * 0.062
        is_extra = llm.get("extra") and (ent in ("PET_GROOMING\nSALON",))
        fc = "#FDEDEC" if is_extra else C_GREY
        ec = "#E74C3C" if is_extra else llm["color"]
        rounded_box(ax, bx - 0.037, ey - 0.020, 0.074, 0.038,
                    fc, alpha=1.0, edgecolor=ec, lw=1.2)
        ax.text(bx, ey, ent.replace("_", "_\n") if len(ent) > 10 else ent,
                ha="center", va="center",
                fontsize=6.5, color=ec,
                fontweight="bold" if is_extra else "normal",
                linespacing=1.1)

    # Relationships
    rel_y_start = ent_label_y - 0.055 - len(llm["entities"]) * 0.062 - 0.025
    ax.text(bx, rel_y_start, "Relationships", ha="center", va="center",
            fontsize=7.5, fontweight="bold", color=llm["color"])
    for k, (e1, e2, card) in enumerate(llm["rels"]):
        ry = rel_y_start - 0.042 - k * 0.052
        is_nm = card == "N:M"
        ax.text(bx, ry,
                f"{e1[:5]}→{e2[:5]}\n{card}",
                ha="center", va="center",
                fontsize=5.8,
                color="#C0392B" if is_nm else "#555",
                linespacing=1.2)

# Arrow: Input → Phase1
draw_arrow(ax, INPUT_X + INPUT_W/2, INPUT_Y + 0.05,
           LLM_XS[0] - 0.045 - 0.005, INPUT_Y + 0.05,
           color=C_INPUT, lw=2)

# ═══════════════════════════════════════════════════════════════════
# 3.  PHASE 2 — Bayesian Network probabilities (centre)
# ═══════════════════════════════════════════════════════════════════
BN_X = 0.685
section_header(ax, BN_X, 0.955, "Phase 2 — Bayesian Network", C_BN, width=0.24)

bn_box_x = BN_X - 0.115
bn_box_y = 0.14
bn_box_w = 0.23
bn_box_h = 0.77

rounded_box(ax, bn_box_x, bn_box_y, bn_box_w, bn_box_h,
            C_BN, alpha=0.05, edgecolor=C_BN, lw=2)

# Table header
header_y = bn_box_y + bn_box_h - 0.045
rounded_box(ax, bn_box_x, header_y, bn_box_w, 0.045,
            C_BN, alpha=0.85, edgecolor=C_BN, lw=1)
for txt, xoff in [("Entity", -0.055), ("P(select)", 0.055), ("Selected?", 0.13)]:
    ax.text(bn_box_x + bn_box_w/2 + xoff - 0.04, header_y + 0.022,
            txt, ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

# Rows
row_h = 0.072
# Attributes sample above entity rows
attr_note_y = header_y - 0.04
ax.text(BN_X, attr_note_y,
        "Entity & Attribute probabilities estimated\nvia consensus across 4 LLM outputs",
        ha="center", va="center", fontsize=7.5,
        style="italic", color="#27AE60")

row_start_y = attr_note_y - 0.06
for idx, (ent, prob, kept) in enumerate(BN_ROWS):
    ry = row_start_y - idx * row_h
    fc = "#FDEDEC" if not kept else "#EBF5FB"
    rounded_box(ax, bn_box_x + 0.005, ry - row_h + 0.005,
                bn_box_w - 0.01, row_h - 0.008,
                fc, alpha=1.0,
                edgecolor="#E74C3C" if not kept else C_BN,
                lw=1.2)
    ax.text(bn_box_x + 0.055, ry - row_h/2, ent,
            ha="center", va="center",
            fontsize=8, color="#E74C3C" if not kept else C_INPUT,
            fontweight="bold")

    # Probability bar
    bar_x = bn_box_x + 0.115
    bar_w_max = 0.075
    bar_w = prob * bar_w_max
    rounded_box(ax, bar_x, ry - row_h + 0.015,
                bar_w_max, row_h - 0.025,
                "#D5DBDB", alpha=1.0, edgecolor="#AAB7B8", lw=0.5)
    rounded_box(ax, bar_x, ry - row_h + 0.015,
                bar_w, row_h - 0.025,
                C_BN if kept else "#E74C3C", alpha=0.7,
                edgecolor="none", lw=0)
    ax.text(bar_x + bar_w_max/2, ry - row_h/2,
            f"{prob:.3f}",
            ha="center", va="center",
            fontsize=8, color=C_INPUT, fontweight="bold")

    # Checkmark / cross
    sym = "✓" if kept else "✗"
    col = C_BN if kept else "#E74C3C"
    ax.text(bn_box_x + bn_box_w - 0.016, ry - row_h/2,
            sym, ha="center", va="center",
            fontsize=12, color=col, fontweight="bold")

# Threshold annotation
thresh_y = row_start_y - (len(BN_ROWS) - 0.5) * row_h - 0.01
ax.axhline(thresh_y + 0.005 * 0 , xmin=0, xmax=0, color="red")  # dummy
ax.annotate("",
            xy=(bn_box_x + bn_box_w - 0.005, thresh_y),
            xytext=(bn_box_x + 0.005, thresh_y),
            arrowprops=dict(arrowstyle="-", color="#E74C3C",
                            linestyle="dashed", lw=1.5))
ax.text(BN_X, thresh_y - 0.025,
        "threshold θ = 0.5 (ILP λ-based)",
        ha="center", va="center", fontsize=7.5,
        color="#E74C3C", style="italic")

# Arrow: Phase1 → BN
draw_arrow(ax, LLM_XS[-1] + 0.045 + 0.005, 0.52,
           bn_box_x - 0.008, 0.52,
           color=C_BN, lw=2)

# ═══════════════════════════════════════════════════════════════════
# 4.  PHASE 3 — ILP output (right)
# ═══════════════════════════════════════════════════════════════════
ILP_X = 0.905
section_header(ax, ILP_X, 0.955, "Phase 3 — ILP Output", C_ILP, width=0.18)

ilp_box_x = ILP_X - 0.085
ilp_box_y = 0.10
ilp_box_w = 0.17
ilp_box_h = 0.80

rounded_box(ax, ilp_box_x, ilp_box_y, ilp_box_w, ilp_box_h,
            C_ILP, alpha=0.05, edgecolor=C_ILP, lw=2)

# Entity positions in final ER
ent_positions = {
    "PET":      (ILP_X,        0.74),
    "GROOMER":  (ILP_X,        0.57),
    "CUSTOMER": (ILP_X - 0.05, 0.40),
    "SERVICE":  (ILP_X + 0.05, 0.25),
}

# Draw entities with attributes
for ent, (ex, ey) in ent_positions.items():
    # Entity name box
    rounded_box(ax, ex - 0.055, ey + 0.005, 0.11, 0.038,
                C_ILP, alpha=0.85, edgecolor=C_ILP, lw=1.5, zorder=3)
    ax.text(ex, ey + 0.024, ent, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white", zorder=4)

    # Attribute list
    for ai, attr in enumerate(FINAL_ATTRS[ent]):
        ay = ey - 0.003 - ai * 0.028
        rounded_box(ax, ex - 0.052, ay - 0.012, 0.104, 0.024,
                    C_GREY, alpha=1.0, edgecolor="#BDC3C7", lw=0.6, zorder=2)
        ax.text(ex, ay, attr, ha="center", va="center",
                fontsize=6, color="#555", zorder=3)

# Draw relationships
for e1, e2, card in FINAL_RELS:
    x1, y1 = ent_positions[e1]
    x2, y2 = ent_positions[e2]
    mx, my = (x1 + x2)/2, (y1 + y2)/2
    ax.annotate("", xy=(x2, y2 + 0.04 + 0.01),
                xytext=(x1, y1 - 0.15 + 0.01),
                arrowprops=dict(arrowstyle="-|>", color=C_ILP,
                                lw=1.2, mutation_scale=10), zorder=1)
    ax.text(mx + 0.012, my + 0.015, card,
            ha="left", va="center", fontsize=6.5,
            color=C_ILP, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.1", fc="white",
                      ec=C_ILP, lw=0.6, alpha=0.85))

# Arrow: BN → ILP
draw_arrow(ax, bn_box_x + bn_box_w + 0.005, 0.52,
           ilp_box_x - 0.005, 0.52,
           color=C_ILP, lw=2)

# ═══════════════════════════════════════════════════════════════════
# 5.  Phase labels at bottom
# ═══════════════════════════════════════════════════════════════════
for txt, xp, col in [
    ("① Input",              0.085, C_INPUT),
    ("② Multi-LLM Generation", 0.395, C_INPUT),
    ("③ BN Consensus",         0.685, C_BN),
    ("④ ILP Optimisation",     0.905, C_ILP),
]:
    ax.text(xp, 0.045, txt, ha="center", va="center",
            fontsize=9, color=col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=C_GREY,
                      ec=col, lw=1.5, alpha=0.9))

# ── Save ─────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    fpath = os.path.join(OUT_DIR, f"pipeline_example.{ext}")
    plt.savefig(fpath, dpi=200, bbox_inches="tight",
                facecolor="white")
    print(f"Saved: {fpath}")

plt.close()
