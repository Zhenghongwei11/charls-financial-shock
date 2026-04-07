#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANCHOR_PATH = ROOT / "results" / "figures" / "ft_evidence_stack_anchor.tsv"
OUT_FIG = ROOT / "results" / "figures" / "ft_evidence_stack_forest.png"


def main() -> None:
    df = pd.read_csv(ANCHOR_PATH, sep="\t")
    if df.empty:
        raise SystemExit(f"Empty anchor table: {ANCHOR_PATH}")

    df = df.copy()
    df["label"] = df["evidence_id"] + "  " + df["line"]
    df = df.sort_values("evidence_id", ascending=False).reset_index(drop=True)

    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    style = {
        "core": {"color": "#1f77b4", "marker": "o", "label": "Core evidence"},
        "limited": {"color": "#E15759", "marker": "s", "label": "Limited (skip-pattern/coverage)"},
        "supportive": {"color": "#59A14F", "marker": "D", "label": "Supportive (non-closed loop)"},
    }

    for strength, sub in df.groupby("strength", sort=False):
        st = style.get(strength, {"color": "#1f77b4", "marker": "o", "label": str(strength)})
        ax.errorbar(
            sub["estimate"],
            sub.index.values,
            xerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
            fmt=st["marker"],
            color=st["color"],
            ecolor=st["color"],
            elinewidth=2,
            capsize=3,
            label=st["label"],
        )

    ax.axvline(0, color="black", linewidth=1, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Estimate (with 95% CI)")
    ax.set_title("Converging evidence stack (associations; not formal mediation)")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(loc="lower right", frameon=False, fontsize=9)

    # Add p-values and sample sizes as right-side text
    x_max = float(np.nanmax(df["ci_high"].values))
    x_text = x_max + (abs(x_max) * 0.25 if x_max != 0 else 0.25)
    for i, r in df.iterrows():
        p = r["p_value"]
        p_str = "p<1e-4" if (pd.notna(p) and p < 1e-4) else (f"p={p:.3f}" if pd.notna(p) else "p=NA")
        ax.text(x_text, i, f"{p_str}, n={int(r['n_obs'])}", va="center", fontsize=9)

    ax.set_xlim(
        float(np.nanmin(df["ci_low"].values)) - 0.1 * abs(float(np.nanmin(df["ci_low"].values))),
        x_text + 0.2 * abs(x_text),
    )

    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=200)
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
