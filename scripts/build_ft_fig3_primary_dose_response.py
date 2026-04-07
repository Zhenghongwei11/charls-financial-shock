#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EFFECT_DIR = ROOT / "results" / "effect_sizes"
FIG_DIR = ROOT / "results" / "figures"

PRIMARY_PATH = EFFECT_DIR / "ft_primary_models.tsv"
ANCHOR_PATH = FIG_DIR / "ft_fig3_primary_dose_response_anchor.tsv"
OUT_FIG = FIG_DIR / "ft_fig3_primary_dose_response.png"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(PRIMARY_PATH, sep="\t")
    if df.empty:
        raise SystemExit(f"Empty primary models table: {PRIMARY_PATH}")

    # Keep the terms needed for a compact figure.
    keep = df[
        (df["model_id"].isin(["dose_continuous_only", "dose_continuous_plus_shock", "dose_shock_only"]))
        & (df["term"].isin(["log_oop_raw_lag", "oop_shock_p95_lag"]))
    ].copy()
    keep = keep.sort_values(["term", "model_id"]).reset_index(drop=True)
    keep.to_csv(ANCHOR_PATH, sep="\t", index=False)

    # Plot: two panels (shock term vs continuous term) across models
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.2), sharey=False)

    def plot_term(ax, term: str, title: str) -> None:
        sub = keep[keep["term"] == term].copy()
        # order: continuous-only, continuous+shock, shock-only (where applicable)
        order = ["dose_continuous_only", "dose_continuous_plus_shock", "dose_shock_only"]
        sub["model_id"] = pd.Categorical(sub["model_id"], categories=order, ordered=True)
        sub = sub.sort_values("model_id")

        y = np.arange(len(sub))
        ax.errorbar(
            sub["estimate"],
            y,
            xerr=[sub["estimate"] - sub["ci_low"], sub["ci_high"] - sub["estimate"]],
            fmt="o",
            color="#4C78A8",
            ecolor="#4C78A8",
            elinewidth=2,
            capsize=3,
        )
        ax.axvline(0, color="black", linewidth=1, alpha=0.6)
        ax.set_yticks(y)
        label_map = {
            "dose_continuous_only": "Continuous only",
            "dose_continuous_plus_shock": "Continuous + shock",
            "dose_shock_only": "Shock only",
        }
        ax.set_yticklabels([label_map.get(m, str(m)) for m in sub["model_id"].astype(str)])
        ax.set_title(title)
        ax.set_xlabel("Estimate (95% CI)")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        for i, r in sub.reset_index(drop=True).iterrows():
            p = r["p_value"]
            p_str = "p<1e-4" if (pd.notna(p) and p < 1e-4) else (f"p={p:.3f}" if pd.notna(p) else "p=NA")
            ax.text(float(r["ci_high"]) + 0.01, i, p_str, va="center", fontsize=9)

    plot_term(axes[0], "oop_shock_p95_lag", "A) Shock term (P95 OOP at t−1)")
    plot_term(axes[1], "log_oop_raw_lag", "B) Continuous spending term (log1p OOP at t−1)")

    fig.suptitle("Figure 3. Primary association and dose-response characterization", y=1.02, fontsize=12)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, wspace=0.35)
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")

    print(f"Wrote {ANCHOR_PATH}")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
