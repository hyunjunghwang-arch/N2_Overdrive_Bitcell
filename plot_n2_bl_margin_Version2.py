"""
N2 Overdrive Bitcell Sim Result Visualiser
==========================================
Reads N2_newbitcelltb_ssm40_taccess_vnegsweep.csv and produces a
professional multi-panel figure for circuit design review.

Usage:
    python plot_n2_bl_margin.py

Adjust THRESHOLDS below to match your review criteria.
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
CSV_FILE   = "N2_newbitcelltb_ssm40_taccess_vnegsweep.csv"
TITLE      = "N2 Overdrive Bitcell sim result - ssm40"
THRESHOLDS = [0.150, 0.100, 0.050]          # V  — edit freely
THRESH_LABELS = ["≥150 mV", "≥100 mV", "≥50 mV"]
THRESH_COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]

TACCESS_UNIT = 1e10   # display in units of ×10⁻¹⁰ s  (0.25 ns → 2.5)
TWRITE_UNIT  = 1e10

# ─────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN
# ─────────────────────────────────────────────────────────────────
# The exported Spectre CSV has several header comment lines before
# the actual column-name row.  We skip everything until we hit the
# line that starts with "writetr0_vth1".
with open(CSV_FILE, "r") as fh:
    lines = fh.readlines()

header_idx = next(
    i for i, ln in enumerate(lines) if ln.strip().startswith("writetr0_vth1")
)
from io import StringIO
df = pd.read_csv(StringIO("".join(lines[header_idx:])), na_values=["NaN", "nan"])

# Convert timing to display units
df["taccess_disp"] = df["taccess"] * TACCESS_UNIT   # e.g. 2.5, 3.0, 4.0
df["twrite_disp"]  = df["twrite"]  * TWRITE_UNIT

# Round to avoid floating-point grouping noise
df["vb_r"]      = df["vb"].round(3)
df["vneg_r"]    = df["vneg"].round(3)
df["ta_r"]      = df["taccess_disp"].round(2)
df["tw_r"]      = df["twrite_disp"].round(2)

# Unique sorted sweep values
vb_vals     = sorted(df["vb_r"].unique())
vneg_vals   = sorted(df["vneg_r"].unique())
ta_vals     = sorted(df["ta_r"].unique())
tw_vals     = sorted(df["tw_r"].unique())

# ─────────────────────────────────────────────────────────────────
# 2. FIGURE LAYOUT
# ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 26), facecolor="#0d1117")
fig.suptitle(
    TITLE,
    fontsize=18, fontweight="bold", color="white",
    y=0.985
)

n_vb   = len(vb_vals)
n_vneg = len(vneg_vals)

# Grid: top section = heatmaps (n_vb rows × n_vneg cols),
#       bottom left = scatter,  bottom right = threshold table
gs = GridSpec(
    n_vb + 2, n_vneg + 1,
    figure=fig,
    hspace=0.55, wspace=0.35,
    left=0.07, right=0.97, top=0.97, bottom=0.04
)

cmap    = plt.cm.RdYlGn
norm    = TwoSlopeNorm(vmin=-0.05, vcenter=0.10, vmax=0.25)
ax_cb   = None   # will hold the colour-bar axis

# ─────────────────────────────────────────────────────────────────
# 3. HEATMAPS  (one per vb × vneg combination)
# ─────────────────────────────────────────────────────────────────
for ri, vb in enumerate(vb_vals):
    for ci, vneg in enumerate(vneg_vals):

        sub = df[(df["vb_r"] == vb) & (df["vneg_r"] == vneg)]
        pivot = sub.pivot_table(
            index="ta_r", columns="tw_r",
            values="bl_voltage_margin", aggfunc="mean"
        )
        pivot = pivot.reindex(index=ta_vals, columns=tw_vals)

        ax = fig.add_subplot(gs[ri, ci])
        im = ax.imshow(
            pivot.values * 1e3,          # → mV
            cmap=cmap,
            norm=mcolors.Normalize(vmin=-50, vmax=250),
            aspect="auto", origin="lower"
        )

        # annotate cells
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    txt_color = "black" if 0.04 < val < 0.18 else "white"
                    ax.text(
                        c, r, f"{val*1e3:.0f}",
                        ha="center", va="center",
                        fontsize=6.5, color=txt_color, fontweight="bold"
                    )

        ax.set_xticks(range(len(tw_vals)))
        ax.set_yticks(range(len(ta_vals)))
        ax.set_xticklabels(
            [f"{v:.2f}" for v in tw_vals], fontsize=7, color="lightgray"
        )
        ax.set_yticklabels(
            [f"{v:.2f}" for v in ta_vals], fontsize=7, color="lightgray"
        )
        ax.set_facecolor("#1a1d23")
        ax.tick_params(colors="gray", length=2)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        if ri == 0:
            ax.set_title(
                f"vneg = {vneg} V",
                fontsize=8.5, color="#aad4f5", fontweight="bold", pad=4
            )
        if ci == 0:
            ax.set_ylabel(
                f"vb = {vb} V\ntaccess (×10⁻¹⁰s)",
                fontsize=7.5, color="#aad4f5", labelpad=3
            )
        if ri == n_vb - 1:
            ax.set_xlabel("twrite (×10⁻¹⁰s)", fontsize=7.5, color="lightgray")

        ax_cb = ax   # keep last axis for colour-bar placement

# Shared colour bar
cbar_ax = fig.add_axes([0.96, 0.52, 0.012, 0.40])
sm = plt.cm.ScalarMappable(
    cmap=cmap, norm=mcolors.Normalize(vmin=-50, vmax=250)
)
sm.set_array([])
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("BL Voltage Margin (mV)", color="lightgray", fontsize=8)
cb.ax.yaxis.set_tick_params(color="lightgray", labelcolor="lightgray", labelsize=7)
cb.outline.set_edgecolor("#444")
for thr in THRESHOLDS:
    cb.ax.axhline(thr * 1e3, color="white", linewidth=1.2, linestyle="--")

# ─────────────────────────────────────────────────────────────────
# 4. SCATTER PLOT  — taccess vs bl_voltage_margin
# ─────────────────────────────────────────────────────────────────
ax_sc = fig.add_subplot(gs[n_vb : n_vb + 2, : n_vneg // 2 + 1])
ax_sc.set_facecolor("#1a1d23")

vb_cmap  = plt.cm.plasma
vb_norm  = mcolors.Normalize(vmin=min(vb_vals), vmax=max(vb_vals))
tw_sizes = {v: 30 + 60 * i / max(1, len(tw_vals) - 1) for i, v in enumerate(tw_vals)}

for _, row in df.iterrows():
    ax_sc.scatter(
        row["taccess_disp"],
        row["bl_voltage_margin"] * 1e3,
        color=vb_cmap(vb_norm(row["vb_r"])),
        s=tw_sizes.get(row["tw_r"], 40),
        alpha=0.72,
        edgecolors="none"
    )

for thr, lbl, col in zip(THRESHOLDS, THRESH_LABELS, THRESH_COLORS):
    ax_sc.axhline(
        thr * 1e3, color=col, linewidth=1.5,
        linestyle="--", label=f"Threshold {lbl}"
    )

ax_sc.set_xlabel("taccess (×10⁻¹⁰ s)", color="lightgray", fontsize=9)
ax_sc.set_ylabel("BL Voltage Margin (mV)", color="lightgray", fontsize=9)
ax_sc.set_title(
    "BL Margin vs taccess  (colour = vb, size = twrite)",
    color="white", fontsize=10, fontweight="bold"
)
ax_sc.tick_params(colors="gray", labelsize=8)
for sp in ax_sc.spines.values():
    sp.set_edgecolor("#444")
ax_sc.legend(
    fontsize=8, facecolor="#1a1d23", edgecolor="#555",
    labelcolor="lightgray", loc="upper right"
)

# vb colour bar inside scatter
sm2   = plt.cm.ScalarMappable(cmap=vb_cmap, norm=vb_norm)
sm2.set_array([])
cbar2 = fig.colorbar(sm2, ax=ax_sc, pad=0.01, fraction=0.03)
cbar2.set_label("vb (V)", color="lightgray", fontsize=7)
cbar2.ax.yaxis.set_tick_params(color="lightgray", labelcolor="lightgray", labelsize=7)
cbar2.outline.set_edgecolor("#444")

# ─────────────────────────────────────────────────────────────────
# 5. THRESHOLD TABLE  — passing param combos per threshold
# ─────────────────────────────────────────────────────────────────
ax_tb = fig.add_subplot(gs[n_vb : n_vb + 2, n_vneg // 2 + 1 :])
ax_tb.set_facecolor("#1a1d23")
ax_tb.axis("off")

ax_tb.set_title(
    "Passing Parameter Combinations per Threshold",
    color="white", fontsize=10, fontweight="bold", pad=6
)

col_labels = ["vb (V)", "vneg (V)", "taccess\n(×10⁻¹⁰s)", "twrite\n(×10⁻¹⁰s)",
              "BL Margin\n(mV)", "Threshold"]
all_rows = []
row_colors = []

for thr, lbl, col in zip(THRESHOLDS, THRESH_LABELS, THRESH_COLORS):
    passing = df[df["bl_voltage_margin"] >= thr][
        ["vb_r", "vneg_r", "ta_r", "tw_r", "bl_voltage_margin"]
    ].copy()
    passing = passing.sort_values("bl_voltage_margin", ascending=False)
    # keep top 8 per threshold to avoid overflow
    for _, r in passing.head(8).iterrows():
        all_rows.append([
            f"{r.vb_r:.3f}",
            f"{r.vneg_r:.1f}",
            f"{r.ta_r:.2f}",
            f"{r.tw_r:.2f}",
            f"{r.bl_voltage_margin*1e3:.1f}",
            lbl
        ])
        row_colors.append([col] * len(col_labels))

if all_rows:
    tbl = ax_tb.table(
        cellText=all_rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.35)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#2c2f38")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            bg = row_colors[r - 1][c]
            cell.set_facecolor(mcolors.to_rgba(bg, alpha=0.18))
            cell.set_text_props(color="lightgray")
else:
    ax_tb.text(
        0.5, 0.5, "No data exceeds selected thresholds",
        ha="center", va="center", color="lightgray", fontsize=10,
        transform=ax_tb.transAxes
    )

# ─────────────────────────────────────────────────────────────────
# 6. COLUMN / ROW LABELS LEGEND STRIP
# ─────────────────────────────────────────────────────────────────
fig.text(
    0.50, 0.008,
    "Heatmap cells show mean BL voltage margin (mV) per taccess × twrite grid "
    "for each vb / vneg combination.  Dashed lines = margin thresholds.",
    ha="center", va="bottom", fontsize=7.5, color="#888"
)

# ─────────────────────────────────────────────────────────────────
# 7. SAVE + SHOW
# ─────────────────────────────────────────────────────────────────
out_file = "N2_BL_margin_analysis.png"
fig.savefig(out_file, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"✓  Saved → {out_file}")
plt.show()