#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QC_DIR = ROOT / "results" / "qc"
FIG_DIR = ROOT / "results" / "figures"

THRESH_PATH = QC_DIR / "ft_shock_thresholds.tsv"
ANCHOR_PATH = FIG_DIR / "ft_fig2_shock_thresholds_anchor.tsv"
OUT_FIG = FIG_DIR / "ft_fig2_shock_thresholds.png"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(THRESH_PATH, sep="\t")
    if df.empty:
        raise SystemExit(f"Empty thresholds table: {THRESH_PATH}")

    # Keep only the thresholds we actually use/defend.
    keep_q = [0.90, 0.95, 0.975, 0.99]
    out = df[df["quantile"].isin(keep_q)].copy()
    out.to_csv(ANCHOR_PATH, sep="\t", index=False)

    # Plot: threshold_total_annual_oop_raw by interval, one line per quantile
    out = out.copy()
    out["interval"] = out["interval"].astype(str)
    interval_order = ["2011-2013", "2013-2015", "2015-2018"]
    out["interval"] = pd.Categorical(out["interval"], categories=interval_order, ordered=True)
    out = out.sort_values(["quantile", "interval"])

    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    colors = {0.90: "#4C78A8", 0.95: "#E15759", 0.975: "#59A14F", 0.99: "#B07AA1"}
    markers = {0.90: "o", 0.95: "s", 0.975: "D", 0.99: "^"}

    for q in keep_q:
        sub = out[out["quantile"] == q]
        ax.plot(
            sub["interval"].astype(str),
            sub["threshold_total_annual_oop_raw"],
            marker=markers[q],
            color=colors[q],
            linewidth=2,
            label=f"q={q:g}",
        )
        for x, y in zip(sub["interval"].astype(str), sub["threshold_total_annual_oop_raw"]):
            ax.text(x, y, f"{y:,.0f}", fontsize=9, ha="center", va="bottom")

    ax.set_title("Figure 2. Interval-specific OOP shock thresholds (raw annual OOP at t−1)")
    ax.set_xlabel("Lag interval")
    ax.set_ylabel("Threshold (2011 RMB, CPI-adjusted)")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(title="Quantile")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")

    print(f"Wrote {ANCHOR_PATH}")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
