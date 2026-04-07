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

FLOW_PATH = QC_DIR / "ft_cohort_flow.tsv"
SELECTION_PATH = QC_DIR / "ft_balanced_vs_main_selection.tsv"

ANCHOR_PATH = FIG_DIR / "ft_fig1_cohort_selection_anchor.tsv"
OUT_FIG = FIG_DIR / "ft_fig1_cohort_selection.png"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()
    flow = pd.read_csv(FLOW_PATH, sep="\t")
    sel = pd.read_csv(SELECTION_PATH, sep="\t")

    # Anchor table (keep a defensible, minimal subset)
    sel_vars = [
        "baseline_age",
        "female_ragender_eq_2",
        "rural_h1rural_eq_1",
        "insured_r1higov_eq_1",
        "education_high_proxy_raeduc_c_ge_4",
        "adl_w1",
    ]
    sel_anchor = sel[sel["variable"].isin(sel_vars)].copy()
    sel_anchor["direction"] = np.where(sel_anchor["smd"] >= 0, "balanced_higher", "excluded_higher")

    flow_anchor = flow.copy()
    flow_anchor["panel"] = "flow"
    sel_anchor["panel"] = "selection"
    out_anchor = pd.concat(
        [
            flow_anchor.rename(columns={"cohort": "label", "n_ids": "value"})[["panel", "label", "value"]],
            sel_anchor.rename(columns={"variable": "label", "smd": "value"})[["panel", "label", "value"]],
        ],
        ignore_index=True,
    )
    out_anchor.to_csv(ANCHOR_PATH, sep="\t", index=False)

    # Plot: panel A flow bars; panel B SMD dotplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1.0, 1.2]})

    # Panel A: cohort flow
    flow_order = [
        "main_cohort_wide_age60_ge_3wave",
        "ft_balanced_by_adl_4wave",
        "excluded_from_ft_balanced",
    ]
    flow_map = {
        "main_cohort_wide_age60_ge_3wave": "Main cohort\n(baseline)",
        "ft_balanced_by_adl_4wave": "Balanced FT cohort\n(ADL 4-wave)",
        "excluded_from_ft_balanced": "Excluded\nfrom balanced",
    }
    flow = flow.set_index("cohort").reindex(flow_order).reset_index()
    ax1.bar(
        [flow_map.get(c, c) for c in flow["cohort"]],
        flow["n_ids"],
        color=["#4C78A8", "#59A14F", "#E15759"],
        alpha=0.9,
    )
    for i, v in enumerate(flow["n_ids"].tolist()):
        ax1.text(i, v + max(flow["n_ids"]) * 0.02, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    ax1.set_title("A) Cohort flow")
    ax1.set_ylabel("Number of participants")
    ax1.tick_params(axis="x", rotation=20)

    # Panel B: selection SMD
    sel_anchor = sel_anchor.sort_values("smd")
    y = np.arange(len(sel_anchor))
    ax2.axvline(0, color="black", linewidth=1, alpha=0.6)
    ax2.scatter(sel_anchor["smd"], y, color="#4C78A8")
    ax2.set_yticks(y)
    ax2.set_yticklabels(
        [
            "Baseline age",
            "Female",
            "Rural (W1)",
            "Public insurance (W1)",
            "Higher education (proxy)",
            "ADL5 at baseline",
        ]
    )
    ax2.set_xlabel("Standardized mean difference (balanced − excluded)")
    ax2.set_title("B) Selection differences (SMD)")
    ax2.grid(axis="x", linestyle="--", alpha=0.25)

    fig.suptitle("Figure 1. Cohort flow and selection context", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")

    print(f"Wrote {ANCHOR_PATH}")
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()

