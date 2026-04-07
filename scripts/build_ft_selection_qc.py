#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WIDE_PATH = ROOT / "data" / "derived" / "charls_main_cohort_wide.tsv.gz"
FT_PATH = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
QC_DIR = ROOT / "results" / "qc"

FLOW_PATH = QC_DIR / "ft_cohort_flow.tsv"
SELECTION_PATH = QC_DIR / "ft_balanced_vs_main_selection.tsv"
WAVE_COMPLETENESS_PATH = QC_DIR / "ft_adl_wave_completeness.tsv"


BASELINE_CONTINUOUS = [
    ("baseline_age", "baseline_age"),
    ("wealth_hh1atotb", "hh1atotb"),
    ("adl_w1", "r1adlfive"),
]

BASELINE_BINARY = [
    ("female_ragender_eq_2", "ragender", 2),
    ("rural_h1rural_eq_1", "h1rural", 1),
    ("insured_r1higov_eq_1", "r1higov", 1),
]


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)


def smd_continuous(x1: pd.Series, x0: pd.Series) -> float:
    x1 = x1.dropna()
    x0 = x0.dropna()
    if x1.empty or x0.empty:
        return float("nan")
    m1 = float(x1.mean())
    m0 = float(x0.mean())
    s1 = float(x1.std(ddof=1))
    s0 = float(x0.std(ddof=1))
    pooled = np.sqrt((s1**2 + s0**2) / 2)
    return (m1 - m0) / pooled if pooled > 0 else float("nan")


def smd_binary(p1: float, p0: float) -> float:
    pooled = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    return (p1 - p0) / pooled if pooled > 0 else float("nan")


def main() -> None:
    ensure_dirs()
    wide = pd.read_csv(WIDE_PATH, sep="\t")
    ft = pd.read_csv(FT_PATH, sep="\t", usecols=["ID"])

    balanced_ids = set(ft["ID"].unique())
    wide["is_ft_balanced"] = wide["ID"].isin(balanced_ids).astype(int)

    flow = pd.DataFrame(
        [
            {"cohort": "main_cohort_wide_age60_ge_3wave", "n_ids": int(wide["ID"].nunique())},
            {"cohort": "ft_balanced_by_adl_4wave", "n_ids": int(wide.loc[wide["is_ft_balanced"] == 1, "ID"].nunique())},
            {"cohort": "excluded_from_ft_balanced", "n_ids": int(wide.loc[wide["is_ft_balanced"] == 0, "ID"].nunique())},
        ]
    )
    flow.to_csv(FLOW_PATH, sep="\t", index=False)

    balanced = wide[wide["is_ft_balanced"] == 1].copy()
    excluded = wide[wide["is_ft_balanced"] == 0].copy()

    rows: list[dict[str, object]] = []
    for label, col in BASELINE_CONTINUOUS:
        x1 = balanced[col]
        x0 = excluded[col]
        rows.append(
            {
                "variable": label,
                "type": "continuous",
                "balanced_n": int(x1.notna().sum()),
                "balanced_mean": float(x1.mean()) if x1.notna().any() else np.nan,
                "balanced_sd": float(x1.std(ddof=1)) if x1.notna().any() else np.nan,
                "excluded_n": int(x0.notna().sum()),
                "excluded_mean": float(x0.mean()) if x0.notna().any() else np.nan,
                "excluded_sd": float(x0.std(ddof=1)) if x0.notna().any() else np.nan,
                "smd": float(smd_continuous(x1, x0)),
            }
        )

    # Education as ordinal/categorical: report missingness + distribution and a simple binary high-edu flag sensitivity
    if "raeduc_c" in wide.columns:
        edu1 = balanced["raeduc_c"]
        edu0 = excluded["raeduc_c"]
        rows.append(
            {
                "variable": "education_raeduc_c_missing",
                "type": "missingness",
                "balanced_n": int(edu1.shape[0]),
                "balanced_mean": float(edu1.isna().mean()),
                "balanced_sd": np.nan,
                "excluded_n": int(edu0.shape[0]),
                "excluded_mean": float(edu0.isna().mean()),
                "excluded_sd": np.nan,
                "smd": np.nan,
            }
        )
        # high education proxy: raeduc_c >= 4 (heuristic, documented as proxy)
        hi1 = (edu1 >= 4).astype(float)
        hi0 = (edu0 >= 4).astype(float)
        p1 = float(hi1.mean())
        p0 = float(hi0.mean())
        rows.append(
            {
                "variable": "education_high_proxy_raeduc_c_ge_4",
                "type": "binary_proxy",
                "balanced_n": int(hi1.notna().sum()),
                "balanced_mean": p1,
                "balanced_sd": np.nan,
                "excluded_n": int(hi0.notna().sum()),
                "excluded_mean": p0,
                "excluded_sd": np.nan,
                "smd": float(smd_binary(p1, p0)),
            }
        )

    for label, col, val in BASELINE_BINARY:
        b1 = (balanced[col] == val).astype(float)
        b0 = (excluded[col] == val).astype(float)
        p1 = float(b1.mean())
        p0 = float(b0.mean())
        rows.append(
            {
                "variable": label,
                "type": "binary",
                "balanced_n": int(b1.notna().sum()),
                "balanced_mean": p1,
                "balanced_sd": np.nan,
                "excluded_n": int(b0.notna().sum()),
                "excluded_mean": p0,
                "excluded_sd": np.nan,
                "smd": float(smd_binary(p1, p0)),
            }
        )

    pd.DataFrame(rows).to_csv(SELECTION_PATH, sep="\t", index=False)

    # ADL wave completeness inside the main cohort (to explain the balanced restriction)
    adl_cols = ["r1adlfive", "r2adlfive", "r3adlfive", "r4adlfive"]
    completeness = []
    for k in range(1, 5):
        completeness.append(
            {
                "adl_nonmissing_waves": k,
                "n_ids": int((wide[adl_cols].notna().sum(axis=1) == k).sum()),
                "pct_of_main": float((wide[adl_cols].notna().sum(axis=1) == k).mean()),
            }
        )
    pd.DataFrame(completeness).to_csv(WAVE_COMPLETENESS_PATH, sep="\t", index=False)

    print(f"Wrote {FLOW_PATH}")
    print(f"Wrote {SELECTION_PATH}")
    print(f"Wrote {WAVE_COMPLETENESS_PATH}")


if __name__ == "__main__":
    main()

