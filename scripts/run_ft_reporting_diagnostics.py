#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[1]

FT_LONG_PATH = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
SHOCK_THRESH_PATH = ROOT / "results" / "qc" / "ft_shock_thresholds.tsv"

QC_DIR = ROOT / "results" / "qc"
OUT_PATH = QC_DIR / "ft_reporting_diagnostics.tsv"

WAVE_TO_YEARS = {1: 0, 2: 2, 3: 4, 4: 7}


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)


def build_intervals_for_change(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t_years"] = out["wave"].map(WAVE_TO_YEARS).astype(float)
    out = out.sort_values(["ID", "t_years"]).reset_index(drop=True)

    g = out.groupby("ID", sort=False)
    out["wave_lag"] = g["wave"].shift(1)
    out["t_years_lag"] = g["t_years"].shift(1)
    out["adl5_lag"] = g["adl5"].shift(1)
    out["core_disease_count_lag"] = g["core_disease_count"].shift(1)
    out["oop_raw_lag"] = g["total_annual_oop_raw"].shift(1)

    out["dt_years"] = out["t_years"] - out["t_years_lag"]
    out["delta_adl"] = out["adl5"] - out["adl5_lag"]
    out["delta_adl_per_year"] = out["delta_adl"] / out["dt_years"]

    out["interval"] = out["wave_lag"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})
    out = out[out["interval"].notna()].copy()
    return out


def add_shock_q95(interval_df: pd.DataFrame, shock_df: pd.DataFrame) -> pd.DataFrame:
    thresh = shock_df[shock_df["quantile"] == 0.95].copy()
    threshold_map = {int(r.wave_lag): float(r.threshold_total_annual_oop_raw) for r in thresh.itertuples(index=False)}
    out = interval_df.copy()
    out["oop_shock_q0_95_lag"] = np.where(
        out["wave_lag"].isin(threshold_map.keys()),
        out["oop_raw_lag"] >= out["wave_lag"].map(threshold_map),
        np.nan,
    ).astype("float")
    return out


def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].copy()
    x = x.dropna().astype(float)
    # Add constant for VIF matrix stability; VIF on constant is not interpreted.
    x = sm.add_constant(x, has_constant="add")
    rows = []
    for i, name in enumerate(x.columns):
        if name == "const":
            continue
        rows.append({"term": name, "vif": float(variance_inflation_factor(x.values, i))})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def change_model_diagnostics(long_df: pd.DataFrame, shock_df: pd.DataFrame) -> list[dict[str, object]]:
    interval_df = add_shock_q95(build_intervals_for_change(long_df), shock_df)
    required = [
        "delta_adl_per_year",
        "oop_shock_q0_95_lag",
        "adl5_lag",
        "core_disease_count_lag",
        "baseline_age",
        "sex",
        "education_c",
        "residence_rural_w1",
        "interval",
    ]
    d = interval_df.dropna(subset=required).copy()

    formula = (
        "delta_adl_per_year ~ oop_shock_q0_95_lag + adl5_lag + core_disease_count_lag + "
        "baseline_age + sex + education_c + residence_rural_w1 + C(interval)"
    )

    fit_cluster = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["ID"]})
    fit_plain = smf.ols(formula, data=d).fit()

    vif_cols = [
        "oop_shock_q0_95_lag",
        "adl5_lag",
        "core_disease_count_lag",
        "baseline_age",
        "sex",
        "education_c",
        "residence_rural_w1",
    ]
    vif = compute_vif(d, vif_cols)
    max_vif = float(vif["vif"].max()) if not vif.empty else np.nan
    top_terms = "; ".join(f"{r.term}={r.vif:.2f}" for r in vif.head(3).itertuples(index=False))

    rows = [
        {
            "module": "primary_change_model",
            "metric": "n_obs",
            "value": int(fit_cluster.nobs),
            "note": "person-interval observations",
        },
        {"module": "primary_change_model", "metric": "n_ids", "value": int(d["ID"].nunique()), "note": "participants"},
        {"module": "primary_change_model", "metric": "r2", "value": float(getattr(fit_cluster, "rsquared", np.nan)), "note": ""},
        {
            "module": "primary_change_model",
            "metric": "fvalue_plain",
            "value": float(getattr(fit_plain, "fvalue", np.nan)),
            "note": "conventional OLS F-statistic (non-robust)",
        },
        {
            "module": "primary_change_model",
            "metric": "f_pvalue_plain",
            "value": float(getattr(fit_plain, "f_pvalue", np.nan)),
            "note": "conventional OLS F-test p-value (non-robust)",
        },
        {"module": "primary_change_model", "metric": "max_vif", "value": max_vif, "note": top_terms},
    ]
    return rows


def poisson_gee_overdispersion(long_df: pd.DataFrame, shock_df: pd.DataFrame) -> list[dict[str, object]]:
    # Mirror the visit-count GEE used in scripts/run_ft_mechanism_utilization.py.
    df = long_df.copy()

    df.loc[
        (df["doctor_visit_any_wave"] == 0) & df["doctor_visit_count_wave"].isna(),
        "doctor_visit_count_wave",
    ] = 0.0

    # Build intervals with lagged utilization
    out = df.copy()
    out["t_years"] = out["wave"].map(WAVE_TO_YEARS).astype(float)
    out = out.sort_values(["ID", "t_years"]).reset_index(drop=True)
    g = out.groupby("ID", sort=False)
    out["wave_lag"] = g["wave"].shift(1)
    out["interval"] = out["wave_lag"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})
    out["doctor_visit_count_lag"] = g["doctor_visit_count_wave"].shift(1)
    out["adl5_lag"] = g["adl5"].shift(1)
    out["core_disease_count_lag"] = g["core_disease_count"].shift(1)
    out["oop_raw_lag"] = g["total_annual_oop_raw"].shift(1)
    out = out[out["interval"].notna()].copy()
    out["log_oop_raw_lag"] = np.log1p(out["oop_raw_lag"])

    thresh = shock_df[shock_df["quantile"] == 0.95].copy()
    threshold_map = {int(r.wave_lag): float(r.threshold_total_annual_oop_raw) for r in thresh.itertuples(index=False)}
    out["oop_shock_p95_lag"] = np.where(
        out["wave_lag"].isin(threshold_map.keys()),
        out["oop_raw_lag"] >= out["wave_lag"].map(threshold_map),
        np.nan,
    ).astype("float")

    required = [
        "doctor_visit_count_wave",
        "doctor_visit_count_lag",
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
    d = out.dropna(subset=required).copy()
    d = d[d["doctor_visit_count_wave"] >= 0].copy()

    fit = smf.gee(
        "doctor_visit_count_wave ~ oop_shock_p95_lag + log_oop_raw_lag + doctor_visit_count_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        groups="ID",
        data=d,
        family=sm.families.Poisson(),
    ).fit()

    resid = getattr(fit, "resid_pearson", None)
    if resid is None:
        dispersion = np.nan
    else:
        df_resid = float(getattr(fit, "df_resid", np.nan))
        dispersion = float(np.sum(np.asarray(resid) ** 2) / df_resid) if df_resid and df_resid > 0 else np.nan

    return [
        {
            "module": "poisson_gee_doctor_visit_count",
            "metric": "dispersion_pearson_chi2_over_df",
            "value": dispersion,
            "note": "Pearson χ²/df (>1 suggests overdispersion)",
        }
    ]


def main() -> None:
    ensure_dirs()
    long_df = pd.read_csv(FT_LONG_PATH, sep="\t")
    shock_df = pd.read_csv(SHOCK_THRESH_PATH, sep="\t")

    rows: list[dict[str, object]] = []
    rows.extend(change_model_diagnostics(long_df, shock_df))
    rows.extend(poisson_gee_overdispersion(long_df, shock_df))

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
