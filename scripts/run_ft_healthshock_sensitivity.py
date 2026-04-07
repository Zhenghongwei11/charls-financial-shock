#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
THRESH_PATH = ROOT / "results" / "qc" / "ft_shock_thresholds.tsv"

QC_DIR = ROOT / "results" / "qc"
EFFECT_DIR = ROOT / "results" / "effect_sizes"

OUT_PATH = EFFECT_DIR / "ft_healthshock_sensitivity.tsv"
SAMPLE_PATH = QC_DIR / "ft_healthshock_sensitivity_sample.tsv"

WAVE_TO_YEARS = {1: 0, 2: 2, 3: 4, 4: 7}


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    EFFECT_DIR.mkdir(parents=True, exist_ok=True)


def build_intervals(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["t_years"] = df["wave"].map(WAVE_TO_YEARS).astype(float)
    df = df.sort_values(["ID", "t_years"]).reset_index(drop=True)

    g = df.groupby("ID", sort=False)
    df["wave_lag"] = g["wave"].shift(1)
    df["t_years_lag"] = g["t_years"].shift(1)
    df["adl5_lag"] = g["adl5"].shift(1)
    df["core_disease_count_lag"] = g["core_disease_count"].shift(1)
    df["hospital_stay_any_lag"] = g["hospital_stay_any_wave"].shift(1)

    df["total_annual_oop_raw_lag"] = g["total_annual_oop_raw"].shift(1)

    df["dt_years"] = df["t_years"] - df["t_years_lag"]
    df["delta_adl"] = df["adl5"] - df["adl5_lag"]
    df["delta_adl_per_year"] = df["delta_adl"] / df["dt_years"]

    df["incident_disease_count"] = (df["core_disease_count"] - df["core_disease_count_lag"]).clip(lower=0)

    df["interval"] = df["wave_lag"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})
    df = df[df["interval"].notna()].copy()
    return df


def add_shock_p95(interval_df: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    thr = thr[thr["quantile"] == 0.95].copy()
    threshold_map = {int(r.wave_lag): float(r.threshold_total_annual_oop_raw) for r in thr.itertuples(index=False)}
    out = interval_df.copy()
    out["oop_shock_p95_lag"] = np.where(
        out["wave_lag"].isin(threshold_map.keys()),
        out["total_annual_oop_raw_lag"] >= out["wave_lag"].map(threshold_map),
        np.nan,
    ).astype("float")
    return out


def tidy(fit, model_id: str, outcome: str, n_obs: int, n_ids: int) -> list[dict[str, object]]:
    conf = fit.conf_int()
    rows: list[dict[str, object]] = []
    for term in fit.params.index:
        if term not in conf.index:
            continue
        rows.append(
            {
                "model_id": model_id,
                "outcome": outcome,
                "term": term,
                "estimate": float(fit.params[term]),
                "std_error": float(fit.bse[term]) if term in fit.bse.index else None,
                "p_value": float(fit.pvalues[term]) if term in fit.pvalues.index else None,
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
                "n_obs": int(n_obs),
                "n_ids": int(n_ids),
            }
        )
    return rows


def main() -> None:
    ensure_dirs()
    long_df = pd.read_csv(INPUT_PATH, sep="\t")
    thr = pd.read_csv(THRESH_PATH, sep="\t")

    interval_df = build_intervals(long_df)
    interval_df = add_shock_p95(interval_df, thr)

    required_base = [
        "delta_adl_per_year",
        "oop_shock_p95_lag",
        "adl5_lag",
        "core_disease_count_lag",
        "baseline_age",
        "sex",
        "education_c",
        "residence_rural_w1",
        "interval",
    ]
    base = interval_df.dropna(subset=required_base).copy()

    sample = pd.DataFrame(
        [
            {"model_id": "base", "n_obs": int(len(base)), "n_ids": int(base["ID"].nunique())},
            {
                "model_id": "plus_hosp_lag",
                "n_obs": int(base.dropna(subset=["hospital_stay_any_lag"]).shape[0]),
                "n_ids": int(base.dropna(subset=["hospital_stay_any_lag"])["ID"].nunique()),
            },
            {
                "model_id": "plus_incident_disease",
                "n_obs": int(base.dropna(subset=["incident_disease_count"]).shape[0]),
                "n_ids": int(base.dropna(subset=["incident_disease_count"])["ID"].nunique()),
            },
            {
                "model_id": "plus_hosp_lag_plus_incident_disease",
                "n_obs": int(base.dropna(subset=["hospital_stay_any_lag", "incident_disease_count"]).shape[0]),
                "n_ids": int(base.dropna(subset=["hospital_stay_any_lag", "incident_disease_count"])["ID"].nunique()),
            },
        ]
    )
    sample.to_csv(SAMPLE_PATH, sep="\t", index=False)

    rows: list[dict[str, object]] = []

    m0 = smf.ols(
        "delta_adl_per_year ~ oop_shock_p95_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=base,
    ).fit(cov_type="cluster", cov_kwds={"groups": base["ID"]})
    rows.extend(tidy(m0, "base", "delta_adl_per_year", int(m0.nobs), int(base["ID"].nunique())))

    m1_df = base.dropna(subset=["hospital_stay_any_lag"]).copy()
    m1 = smf.ols(
        "delta_adl_per_year ~ oop_shock_p95_lag + hospital_stay_any_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=m1_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": m1_df["ID"]})
    rows.extend(tidy(m1, "plus_hosp_lag", "delta_adl_per_year", int(m1.nobs), int(m1_df["ID"].nunique())))

    m2_df = base.dropna(subset=["incident_disease_count"]).copy()
    m2 = smf.ols(
        "delta_adl_per_year ~ oop_shock_p95_lag + incident_disease_count + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=m2_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": m2_df["ID"]})
    rows.extend(tidy(m2, "plus_incident_disease", "delta_adl_per_year", int(m2.nobs), int(m2_df["ID"].nunique())))

    m3_df = base.dropna(subset=["hospital_stay_any_lag", "incident_disease_count"]).copy()
    m3 = smf.ols(
        "delta_adl_per_year ~ oop_shock_p95_lag + hospital_stay_any_lag + incident_disease_count + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=m3_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": m3_df["ID"]})
    rows.extend(tidy(m3, "plus_hosp_lag_plus_incident_disease", "delta_adl_per_year", int(m3.nobs), int(m3_df["ID"].nunique())))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, sep="\t", index=False)

    print(f"Wrote {SAMPLE_PATH}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

