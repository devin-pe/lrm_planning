#!/usr/bin/env python3
"""Bar plot: optimally-solved counts, baseline LoRA-LM vs + state-encoding loss."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

N = 500
disks = [3, 4, 5]
base     = [100*v/N for v in [260, 84, 29]]    # base Qwen3.6-27B, no fine-tuning (runs/qwen36_base)
baseline = [100*v/N for v in [260, 138, 33]]   # LoRA LM fine-tune
se       = [100*v/N for v in [261, 163, 37]]   # LoRA LM + state-encoding loss
labels = ["3 disks\n(OOD)", "4 disks\n(in-distribution)", "5 disks\n(OOD)"]

# palette: slate (base) + ocean blue (LoRA) + warm coral (gain)
GREY      = "#9AA7B0"   # base model, no fine-tuning
GREY_DARK = "#54606E"   # base value labels
BLUE      = "#2C6E9C"   # optimally solved (shared base level)
BLUE_DARK = "#1B4965"   # baseline value labels
CORAL     = "#E8804A"   # gain from state-encoding loss
CORAL_DARK = "#BC5E28"  # state-encoding value / delta labels
GRID      = "#CAD2D8"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#54606E",
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=150)
x = np.arange(len(disks))
w = 0.27

# base-model bars (no fine-tuning)
ax.bar(x - w, base, w, color=GREY, label="Base (no fine-tuning)",
       edgecolor="white", linewidth=0.6, zorder=3)
# baseline LoRA-LM bars
ax.bar(x, baseline, w, color=BLUE, label="LoRA LM",
       edgecolor="white", linewidth=0.6, zorder=3)
# state-encoding bars: shared base (= baseline level) + gain cap
deltas = [s - b for s, b in zip(se, baseline)]
ax.bar(x + w, baseline, w, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
ax.bar(x + w, deltas, w, bottom=baseline, color=CORAL,
       label="Gain from state-encoding loss", edgecolor="white", linewidth=0.6, zorder=3)

# value labels
for xi, v in zip(x - w, base):
    ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", va="bottom",
            fontsize=10, color=GREY_DARK, fontweight="bold")
for xi, b in zip(x, baseline):
    ax.text(xi, b + 0.8, f"{b:.1f}", ha="center", va="bottom",
            fontsize=10, color=BLUE_DARK, fontweight="bold")
for xi, b, s, d in zip(x + w, baseline, se, deltas):
    ax.text(xi, s + 0.8, f"{s:.1f}", ha="center", va="bottom",
            fontsize=10, color=CORAL_DARK, fontweight="bold")
    if d > 0:  # delta label always on the side of the cap
        ax.text(xi + w/2 + 0.03, b + d/2, f"+{d:.1f}", ha="left", va="center",
                fontsize=9.5, color=CORAL_DARK, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Optimally solved (%)")
ax.set_ylim(0, 60)
ax.set_yticks(range(0, 61, 10))

ax.yaxis.grid(True, color=GRID, linestyle="--", linewidth=0.7, alpha=0.6, zorder=0)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

ax.legend(frameon=False, loc="upper right", fontsize=10)
fig.tight_layout()
for out in ["/home/dpereira/Master-AI-Thesis/media/images/finetune_optimal.pdf",
            "/home/dpereira/COLM-2026-Disk-onnected-/media/images/finetune_optimal.pdf"]:
    fig.savefig(out, bbox_inches="tight")
fig.savefig("/home/dpereira/COLM-2026-Disk-onnected-/media/images/preview_ft.png",
            bbox_inches="tight", dpi=150)
print("done")
