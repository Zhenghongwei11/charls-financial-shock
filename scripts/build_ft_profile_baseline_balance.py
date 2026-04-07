#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FT_LONG_PATH = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
PROFILE_PATH = ROOT / "results" / "trajectory" / "ft_trajectory_profiles.tsv"
QC_DIR = ROOT / "results" / "qc"

OUT_PATH = QC_DIR / "ft_profile_baseline_balance.tsv"


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
    long_df = pd.read_csv(FT_LONG_PATH, sep="\t")
    profiles = pd.read_csv(PROFILE_PATH, sep="\t", usecols=["ID", "ft_profile"])

    # Baseline (wave 1) snapshot for comparability
    base = long_df[long_df["wave"] == 1].merge(profiles, on="ID", how="left")
    ref = base[base["ft_profile"] == "stable_low"].copy()

    rows: list[dict[str, object]] = []

    def add_cont(name: str, series: pd.Series, ref_series: pd.Series):
        rows.append(
            {
                "variable": name,
                "type": "continuous",
                "overall_n": int(series.notna().sum()),
                "overall_mean": float(series.mean()) if series.notna().any() else np.nan,
                "overall_sd": float(series.std(ddof=1)) if series.notna().any() else np.nan,
                "ref_n": int(ref_series.notna().sum()),
                "ref_mean": float(ref_series.mean()) if ref_series.notna().any() else np.nan,
                "ref_sd": float(ref_series.std(ddof=1)) if ref_series.notna().any() else np.nan,
                "smd_overall_vs_ref": float(smd_continuous(series, ref_series)),
            }
        )
        for grp, subset in base.groupby("ft_profile", sort=True):
            s = subset[name]
            rs = ref_series
            rows.append(
                {
                    "variable": name,
                    "type": "continuous_by_group",
                    "group": grp,
                    "n": int(s.notna().sum()),
                    "mean": float(s.mean()) if s.notna().any() else np.nan,
                    "sd": float(s.std(ddof=1)) if s.notna().any() else np.nan,
                    "smd_vs_ref": float(smd_continuous(s, rs)),
                }
            )

    def add_bin(name: str, series: pd.Series, ref_series: pd.Series):
        p = float(series.mean()) if series.notna().any() else np.nan
        p0 = float(ref_series.mean()) if ref_series.notna().any() else np.nan
        rows.append(
            {
                "variable": name,
                "type": "binary",
                "overall_n": int(series.notna().sum()),
                "overall_prop": p,
                "ref_n": int(ref_series.notna().sum()),
                "ref_prop": p0,
                "smd_overall_vs_ref": float(smd_binary(p, p0)) if np.isfinite(p) and np.isfinite(p0) else np.nan,
            }
        )
        for grp, subset in base.groupby("ft_profile", sort=True):
            s = subset[name]
            pg = float(s.mean()) if s.notna().any() else np.nan
            rows.append(
                {
                    "variable": name,
                    "type": "binary_by_group",
                    "group": grp,
                    "n": int(s.notna().sum()),
                    "prop": pg,
                    "smd_vs_ref": float(smd_binary(pg, p0)) if np.isfinite(pg) and np.isfinite(p0) else np.nan,
                }
            )

    # Continuous baseline variables
    for col in ["baseline_age", "ses_wealth_w1", "core_disease_count", "adl5"]:
        if col in base.columns:
            add_cont(col, base[col], ref[col])

    # Binary baseline variables
    if "sex" in base.columns:
        base["female_flag"] = np.where(base["sex"].notna(), (base["sex"] == 2).astype(float), np.nan)
        ref["female_flag"] = np.where(ref["sex"].notna(), (ref["sex"] == 2).astype(float), np.nan)
        add_bin("female_flag", base["female_flag"], ref["female_flag"])

    for col in ["residence_rural_w1", "insurance_public_w1", "doctor_visit_any_wave", "hospital_stay_any_wave"]:
        if col in base.columns:
            add_bin(col, base[col].astype(float), ref[col].astype(float))

    # Education as ordinal: report missingness + high-education proxy
    if "education_c" in base.columns:
        base["education_missing"] = base["education_c"].isna().astype(float)
        ref["education_missing"] = ref["education_c"].isna().astype(float)
        add_bin("education_missing", base["education_missing"], ref["education_missing"])

        base["education_high_proxy_ge4"] = np.where(base["education_c"].notna(), (base["education_c"] >= 4).astype(float), np.nan)
        ref["education_high_proxy_ge4"] = np.where(ref["education_c"].notna(), (ref["education_c"] >= 4).astype(float), np.nan)
        add_bin("education_high_proxy_ge4", base["education_high_proxy_ge4"], ref["education_high_proxy_ge4"])

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

