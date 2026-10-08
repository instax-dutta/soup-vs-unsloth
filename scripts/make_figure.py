"""Render results/figures/vram.png from the raw nvidia-smi samples.

One line per arm, aligned at t=0. Inputs are the CSVs written by
sample_gpu.sh during the 4-arm matrix run (results/gpu/).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPU = os.path.join(ROOT, "results", "gpu")
OUT = os.path.join(ROOT, "results", "figures", "vram.png")

ARMS = [
    ("unsloth_native", "Unsloth native", "#1b9e77", "-"),
    ("soup_unsloth", "Soup, backend: unsloth", "#d95f02", "-"),
    ("soup_hf_plain", "Soup, backend: transformers (unsloth installed)", "#7570b3", "-"),
    ("soup_hf_pure", "Soup, backend: transformers (unsloth blocked)", "#e7298a", "--"),
]

fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200)

for tag, label, color, ls in ARMS:
    df = pd.read_csv(os.path.join(GPU, f"{tag}_gpu.csv"))
    t = (df["ts"] - df["ts"].iloc[0]) / 60
    peak = int(df["used_mib"].max())
    ax.plot(
        t, df["used_mib"] / 1024,
        color=color, linestyle=ls, linewidth=1.6,
        label=f"{label} - peak {peak:,} MiB",
    )
    if tag == "soup_hf_pure":
        ax.annotate(
            "OOM",
            xy=(t.iloc[-1], df["used_mib"].iloc[-1] / 1024),
            xytext=(8, 6), textcoords="offset points",
            color=color, fontsize=9, fontweight="bold",
        )

ax.axhline(15360 / 1024, color="#999999", linewidth=0.9, linestyle=":",
           label="T4 total: 15,360 MiB")

ax.set_xlabel("minutes since arm start")
ax.set_ylabel("GPU memory used (GiB)")
ax.set_title("GPU memory during training - 2x Tesla T4, one arm per GPU", fontsize=11)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(axis="y", color="#e8e8e8", linewidth=0.7)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="lower right", fontsize=8, frameon=False)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print("wrote", OUT)
