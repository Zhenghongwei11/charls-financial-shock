#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
FT_LONG_PATH = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
THRESH_PATH = ROOT / "results" / "qc" / "ft_shock_thresholds.tsv"

EFFECT_DIR = ROOT / "results" / "effect_sizes"
OUT_PATH = EFFECT_DIR / "ft_mechanism_utilization.tsv"

WAVE_TO_YEARS = {1: 0, 2: 2, 3: 4, 4: 7}


def ensure_dirs() -> None:
    EFFECT_DIR.mkdir(parents=True, exist_ok=True)


def build_intervals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t_years"] = out["wave"].map(WAVE_TO_YEARS).astype(float)
    out = out.sort_values(["ID", "t_years"]).reset_index(drop=True)

    g = out.groupby("ID", sort=False)
    out["wave_lag"] = g["wave"].shift(1)
    out["interval"] = out["wave_lag"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})

    out["doctor_visit_any_lag"] = g["doctor_visit_any_wave"].shift(1)
    out["doctor_visit_count_lag"] = g["doctor_visit_count_wave"].shift(1)
    out["hospital_stay_any_lag"] = g["hospital_stay_any_wave"].shift(1)
    out["adl5_lag"] = g["adl5"].shift(1)
    out["core_disease_count_lag"] = g["core_disease_count"].shift(1)
    out["oop_raw_lag"] = g["total_annual_oop_raw"].shift(1)
    out = out[out["interval"].notna()].copy()
    out["log_oop_raw_lag"] = np.log1p(out["oop_raw_lag"])
    return out


def add_shock_flags(interval_df: pd.DataFrame, shock_df: pd.DataFrame) -> pd.DataFrame:
    # Use p95 shock as default mechanism exposure
    thresh = shock_df[shock_df["quantile"] == 0.95].copy()
    threshold_map = {int(r.wave_lag): float(r.threshold_total_annual_oop_raw) for r in thresh.itertuples(index=False)}
    out = interval_df.copy()
    out["oop_shock_p95_lag"] = np.where(
        out["wave_lag"].isin(threshold_map.keys()),
        out["oop_raw_lag"] >= out["wave_lag"].map(threshold_map),
        np.nan,
    ).astype("float")
    return out


def fit_cluster_glm(df: pd.DataFrame, formula: str, outcome: str, model_id: str) -> pd.DataFrame:
    fit = smf.glm(formula, data=df, family=sm.families.Binomial()).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["ID"]},
        disp=False,
        maxiter=200,
    )
    conf = fit.conf_int()
    rows = []
    for term in fit.params.index:
        if term not in conf.index:
            continue
        rows.append(
            {
                "model_id": model_id,
                "outcome": outcome,
                "term": term,
                "estimate_logit": float(fit.params[term]),
                "std_error": float(fit.bse[term]) if term in fit.bse.index else None,
                "p_value": float(fit.pvalues[term]) if term in fit.pvalues.index else None,
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
                "n_obs": int(fit.nobs),
                "n_ids": int(df["ID"].nunique()),
            }
        )
    return pd.DataFrame(rows)

def fit_gee(df: pd.DataFrame, formula: str, outcome: str, model_id: str, family) -> pd.DataFrame:
    fit = smf.gee(
        formula,
        groups="ID",
        data=df,
        family=family,
    ).fit()
    conf = fit.conf_int()
    rows = []
    for term in fit.params.index:
        if term not in conf.index:
            continue
        rows.append(
            {
                "model_id": model_id,
                "outcome": outcome,
                "term": term,
                "estimate_link": float(fit.params[term]),
                "std_error": float(fit.bse[term]) if term in fit.bse.index else None,
                "p_value": float(fit.pvalues[term]) if term in fit.pvalues.index else None,
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
                "n_obs": int(fit.nobs),
                "n_ids": int(df["ID"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    long_df = pd.read_csv(FT_LONG_PATH, sep="\t")
    shock_df = pd.read_csv(THRESH_PATH, sep="\t")

    # Structural zero for counts when no visit and count is missing
    long_df.loc[
        (long_df["doctor_visit_any_wave"] == 0) & long_df["doctor_visit_count_wave"].isna(),
        "doctor_visit_count_wave",
    ] = 0.0

    interval_df = add_shock_flags(build_intervals(long_df), shock_df)

    required_common = [
        "oop_shock_p95_lag",
        "log_oop_raw_lag",
        "adl5_lag",
        "core_disease_count_lag",
        "baseline_age",
        "sex",
        "education_c",
        "residence_rural_w1",
        "interval",
    ]

    # Mechanism proxy 1: doctor visit at time t
    doc_df = interval_df.dropna(subset=["doctor_visit_any_wave", "doctor_visit_any_lag", *required_common]).copy()
    doc_out = fit_cluster_glm(
        doc_df,
        "doctor_visit_any_wave ~ oop_shock_p95_lag + log_oop_raw_lag + doctor_visit_any_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        "doctor_visit_any_wave",
        "util_mech_doctor_visit",
    )

    # Mechanism proxy 1b: doctor visit count at time t (Poisson GEE)
    doc_count_df = interval_df.dropna(subset=["doctor_visit_count_wave", "doctor_visit_count_lag", *required_common]).copy()
    # counts should be non-negative
    doc_count_df = doc_count_df[doc_count_df["doctor_visit_count_wave"] >= 0].copy()
    doc_count_out = fit_gee(
        doc_count_df,
        "doctor_visit_count_wave ~ oop_shock_p95_lag + log_oop_raw_lag + doctor_visit_count_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        "doctor_visit_count_wave",
        "util_mech_doctor_visit_count_pois_gee",
        sm.families.Poisson(),
    )

    # Mechanism proxy 2: hospital stay at time t
    hosp_df = interval_df.dropna(subset=["hospital_stay_any_wave", "hospital_stay_any_lag", *required_common]).copy()
    hosp_out = fit_cluster_glm(
        hosp_df,
        "hospital_stay_any_wave ~ oop_shock_p95_lag + log_oop_raw_lag + hospital_stay_any_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        "hospital_stay_any_wave",
        "util_mech_hospital_stay",
    )

    out = pd.concat([doc_out, doc_count_out, hosp_out], ignore_index=True)
    out.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
