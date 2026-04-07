#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "derived" / "charls_main_cohort_long_extended.tsv.gz"
QC_DIR = ROOT / "results" / "qc"
EFFECT_DIR = ROOT / "results" / "effect_sizes"

ELIGIBILITY_PATH = QC_DIR / "ft_lagged_eligibility_maincohort.tsv"
SHOCK_THRESHOLDS_PATH = QC_DIR / "ft_shock_thresholds_maincohort.tsv"
IPW_SUMMARY_PATH = QC_DIR / "ft_ipw_weights_summary_maincohort.tsv"
MODEL_PATH = EFFECT_DIR / "ft_lagged_models_maincohort.tsv"

WAVE_TO_YEARS = {1: 0, 2: 2, 3: 4, 4: 7}

# CPI indices for 2011-2018 relative to 2011 (Source: NBS China)
# 2011 = 100.0
# 2013 = 105.3
# 2015 = 108.6
# 2018 = 116.1
CPI_ADJUSTMENT = {1: 1.0, 2: 1.053, 3: 1.086, 4: 1.161}

SPEND_COLS = ["outpatient_total_wave", "outpatient_oop_wave", "hospital_total_wave", "hospital_oop_wave"]


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    EFFECT_DIR.mkdir(parents=True, exist_ok=True)


def winsorize_by_wave(df: pd.DataFrame, column: str, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    out = df[column].copy()
    for wave, _factor in CPI_ADJUSTMENT.items():
        mask = df["wave"] == wave
        values = df.loc[mask, column].dropna()
        if values.empty:
            continue
        low = float(values.quantile(lower_q))
        high = float(values.quantile(upper_q))
        out.loc[mask] = df.loc[mask, column].clip(lower=low, upper=high)
    return out


def preprocess_spending(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # CPI adjustment
    for wave, factor in CPI_ADJUSTMENT.items():
        mask = out["wave"] == wave
        for col in SPEND_COLS:
            out.loc[mask, col] = out.loc[mask, col] / factor

    # Structural zeros (no NA→0 imputation)
    out.loc[(out["doctor_visit_any_wave"] == 0) & out["outpatient_total_wave"].isna(), "outpatient_total_wave"] = 0.0
    out.loc[(out["doctor_visit_any_wave"] == 0) & out["outpatient_oop_wave"].isna(), "outpatient_oop_wave"] = 0.0
    out.loc[(out["hospital_stay_any_wave"] == 0) & out["hospital_total_wave"].isna(), "hospital_total_wave"] = 0.0
    out.loc[(out["hospital_stay_any_wave"] == 0) & out["hospital_oop_wave"].isna(), "hospital_oop_wave"] = 0.0

    # Raw tracking + per-wave winsorization
    for col in SPEND_COLS:
        out[f"{col}_raw"] = out[col]
        out[col] = winsorize_by_wave(out, col)

    out["total_annual_oop"] = (out["outpatient_oop_wave"] * 12) + out["hospital_oop_wave"]
    out["total_annual_oop_raw"] = (out["outpatient_oop_wave_raw"] * 12) + out["hospital_oop_wave_raw"]

    wealth = out["ses_wealth_w1"]
    out["ft_burden_ratio_raw"] = np.where(wealth > 0, out["total_annual_oop_raw"] / wealth, np.nan)
    return out


def build_intervals(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["t_years"] = df["wave"].map(WAVE_TO_YEARS).astype(float)
    df = df.sort_values(["ID", "t_years"]).reset_index(drop=True)

    group = df.groupby("ID", sort=False)
    df["wave_lag"] = group["wave"].shift(1)
    df["t_years_lag"] = group["t_years"].shift(1)
    df["adl5_lag"] = group["adl5"].shift(1)
    df["core_disease_count_lag"] = group["core_disease_count"].shift(1)
    df["total_annual_oop_raw_lag"] = group["total_annual_oop_raw"].shift(1)
    df["ft_burden_ratio_raw_lag"] = group["ft_burden_ratio_raw"].shift(1)

    df["dt_years"] = df["t_years"] - df["t_years_lag"]
    df["delta_adl"] = df["adl5"] - df["adl5_lag"]
    df["delta_adl_per_year"] = df["delta_adl"] / df["dt_years"]

    df["interval"] = df["wave_lag"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})
    df = df[df["interval"].notna()].copy()
    return df


def build_eligibility(interval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for interval, subset in interval_df.groupby("interval", sort=False):
        rows.append(
            {
                "interval": interval,
                "n_rows": int(len(subset)),
                "n_ids": int(subset["ID"].nunique()),
                "adl5_missing_pct": float(subset["adl5"].isna().mean()),
                "adl5_lag_missing_pct": float(subset["adl5_lag"].isna().mean()),
                "oop_raw_lag_missing_pct": float(subset["total_annual_oop_raw_lag"].isna().mean()),
            }
        )
    return pd.DataFrame(rows)


def shock_thresholds(interval_df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    for q in quantiles:
        for wave_lag in [1, 2, 3]:
            subset = interval_df[interval_df["wave_lag"] == wave_lag]
            values_oop = subset["total_annual_oop_raw_lag"].dropna()
            if values_oop.empty:
                continue
            values_burden = subset["ft_burden_ratio_raw_lag"].dropna()
            rows.append(
                {
                    "wave_lag": int(wave_lag),
                    "interval": subset["interval"].iloc[0],
                    "quantile": float(q),
                    "threshold_total_annual_oop_raw": float(values_oop.quantile(q)),
                    "n_nonmissing": int(values_oop.shape[0]),
                    "threshold_ft_burden_ratio_raw": float(values_burden.quantile(q)) if not values_burden.empty else np.nan,
                    "n_nonmissing_ft_burden_ratio_raw": int(values_burden.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def build_ipw_weights(interval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = interval_df.copy()
    df["oop_raw_lag_observed"] = df["total_annual_oop_raw_lag"].notna().astype(int)

    ipw_df = df.dropna(
        subset=[
            "adl5_lag",
            "core_disease_count_lag",
            "baseline_age",
            "sex",
            "education_c",
            "residence_rural_w1",
            "oop_raw_lag_observed",
        ]
    ).copy()
    if ipw_df.empty:
        df["ipw"] = np.nan
        return df, pd.DataFrame()

    fit = smf.glm(
        "oop_raw_lag_observed ~ adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=ipw_df,
        family=sm.families.Binomial(),
    ).fit()
    p = fit.predict(ipw_df).clip(lower=1e-3, upper=1 - 1e-3)
    marginal = float(ipw_df["oop_raw_lag_observed"].mean())
    ipw_df["ipw_raw"] = (marginal / p).where(ipw_df["oop_raw_lag_observed"] == 1, np.nan)
    ipw_df["ipw"] = ipw_df["ipw_raw"].clip(upper=float(ipw_df["ipw_raw"].quantile(0.99)))

    df = df.merge(ipw_df[["ID", "wave", "interval", "ipw"]], on=["ID", "wave", "interval"], how="left")
    summary = pd.DataFrame(
        [
            {
                "n_rows_model": int(len(ipw_df)),
                "observed_fraction": marginal,
                "ipw_nonmissing_n": int(ipw_df["ipw"].notna().sum()),
                "ipw_mean": float(ipw_df["ipw"].mean()),
                "ipw_p95": float(ipw_df["ipw"].quantile(0.95)),
                "ipw_p99": float(ipw_df["ipw"].quantile(0.99)),
                "ipw_max": float(ipw_df["ipw"].max()),
                "model_converged": bool(getattr(fit, "converged", True)),
            }
        ]
    )
    return df, summary


def main() -> None:
    ensure_dirs()
    long_df = pd.read_csv(INPUT_PATH, sep="\t")
    long_df = preprocess_spending(long_df)

    interval_df = build_intervals(long_df)
    eligibility = build_eligibility(interval_df)
    eligibility.to_csv(ELIGIBILITY_PATH, sep="\t", index=False)

    shock_df = shock_thresholds(interval_df, quantiles=[0.90, 0.95, 0.975, 0.99])
    shock_df.to_csv(SHOCK_THRESHOLDS_PATH, sep="\t", index=False)

    interval_df["log_oop_raw_lag"] = np.log1p(interval_df["total_annual_oop_raw_lag"])
    interval_df, ipw_summary = build_ipw_weights(interval_df)
    ipw_summary.to_csv(IPW_SUMMARY_PATH, sep="\t", index=False)

    out_rows: list[dict[str, object]] = []

    def tidy(fit, model_id: str, outcome: str, n: int, n_ids: int, shock_quantile: float | None, weighted: bool):
        conf = fit.conf_int()
        for term in fit.params.index:
            if term not in conf.index:
                continue
            out_rows.append(
                {
                    "model_id": model_id,
                    "outcome": outcome,
                    "term": term,
                    "estimate": float(fit.params[term]),
                    "std_error": float(fit.bse[term]) if term in fit.bse.index else None,
                    "p_value": float(fit.pvalues[term]) if term in fit.pvalues.index else None,
                    "ci_low": float(conf.loc[term, 0]),
                    "ci_high": float(conf.loc[term, 1]),
                    "n_obs": int(n),
                    "n_ids": int(n_ids),
                    "r2": float(getattr(fit, "rsquared", np.nan)),
                    "shock_quantile": shock_quantile,
                    "weighted_ipw": int(weighted),
                }
            )

    for q in sorted(shock_df["quantile"].unique()):
        threshold_map = {
            int(r.wave_lag): float(r.threshold_total_annual_oop_raw)
            for r in shock_df[shock_df["quantile"] == q].itertuples(index=False)
        }
        shock_col = f"oop_shock_q{str(q).replace('.', '_')}_lag"
        interval_df[shock_col] = np.where(
            interval_df["wave_lag"].isin(threshold_map.keys()),
            interval_df["total_annual_oop_raw_lag"] >= interval_df["wave_lag"].map(threshold_map),
            np.nan,
        ).astype("float")

        base_required = [
            "log_oop_raw_lag",
            shock_col,
            "adl5_lag",
            "core_disease_count_lag",
            "baseline_age",
            "sex",
            "education_c",
            "residence_rural_w1",
            "interval",
        ]

        change_df = interval_df.dropna(subset=["delta_adl_per_year", *base_required]).copy()
        change_model = smf.ols(
            f"delta_adl_per_year ~ log_oop_raw_lag + {shock_col} + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
            data=change_df,
        ).fit(cov_type="cluster", cov_kwds={"groups": change_df["ID"]})
        tidy(change_model, f"main_change_cluster_ols_q{q}", "delta_adl_per_year", len(change_df), change_df["ID"].nunique(), float(q), False)

        change_w = change_df.dropna(subset=["ipw"]).copy()
        if not change_w.empty:
            change_ipw = smf.wls(
                f"delta_adl_per_year ~ log_oop_raw_lag + {shock_col} + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
                data=change_w,
                weights=change_w["ipw"],
            ).fit(cov_type="cluster", cov_kwds={"groups": change_w["ID"]})
            tidy(change_ipw, f"main_change_ipw_wls_q{q}", "delta_adl_per_year", len(change_w), change_w["ID"].nunique(), float(q), True)

    pd.DataFrame(out_rows).to_csv(MODEL_PATH, sep="\t", index=False)

    print(f"Wrote {ELIGIBILITY_PATH}")
    print(f"Wrote {SHOCK_THRESHOLDS_PATH}")
    print(f"Wrote {IPW_SUMMARY_PATH}")
    print(f"Wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
