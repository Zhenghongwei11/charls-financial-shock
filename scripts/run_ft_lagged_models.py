#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
QC_DIR = ROOT / "results" / "qc"
EFFECT_DIR = ROOT / "results" / "effect_sizes"

ELIGIBILITY_PATH = QC_DIR / "ft_lagged_eligibility.tsv"
SHOCK_THRESHOLDS_PATH = QC_DIR / "ft_shock_thresholds.tsv"
IPW_SUMMARY_PATH = QC_DIR / "ft_ipw_weights_summary.tsv"
MODEL_PATH = EFFECT_DIR / "ft_lagged_models.tsv"

WAVE_TO_YEARS = {1: 0, 2: 2, 3: 4, 4: 7}


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    EFFECT_DIR.mkdir(parents=True, exist_ok=True)


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
    df["total_annual_oop_lag"] = group["total_annual_oop"].shift(1)
    df["ft_burden_ratio_raw_lag"] = group["ft_burden_ratio_raw"].shift(1)

    df["dt_years"] = df["t_years"] - df["t_years_lag"]
    df["delta_adl"] = df["adl5"] - df["adl5_lag"]
    df["delta_adl_per_year"] = df["delta_adl"] / df["dt_years"]

    df["interval"] = df["wave_lag"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})
    df = df[df["interval"].notna()].copy()
    return df


def shock_thresholds(interval_df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    for q in quantiles:
        for wave_lag in [1, 2, 3]:
            subset = interval_df[interval_df["wave_lag"] == wave_lag]
            values_oop = subset["total_annual_oop_raw_lag"].dropna()
            if values_oop.empty:
                continue
            threshold_oop = float(values_oop.quantile(q))

            values_burden = subset["ft_burden_ratio_raw_lag"].dropna()
            threshold_burden = float(values_burden.quantile(q)) if not values_burden.empty else np.nan
            rows.append(
                {
                    "wave_lag": int(wave_lag),
                    "interval": subset["interval"].iloc[0],
                    "quantile": float(q),
                    "threshold_total_annual_oop_raw": threshold_oop,
                    "n_nonmissing": int(values_oop.shape[0]),
                    "threshold_ft_burden_ratio_raw": threshold_burden,
                    "n_nonmissing_ft_burden_ratio_raw": int(values_burden.shape[0]),
                }
            )
    return pd.DataFrame(rows)


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
                "dt_years_unique": ";".join(map(str, sorted(subset["dt_years"].dropna().unique()))),
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

    # Predict observation of exposure (missingness model) with interval fixed effects.
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
                "ipw_p50": float(ipw_df["ipw"].median()),
                "ipw_p95": float(ipw_df["ipw"].quantile(0.95)),
                "ipw_p99": float(ipw_df["ipw"].quantile(0.99)),
                "ipw_max": float(ipw_df["ipw"].max()),
                "model_converged": bool(getattr(fit, "converged", True)),
            }
        ]
    )
    return df, summary


def fit_models(interval_df: pd.DataFrame, shock_df: pd.DataFrame) -> pd.DataFrame:
    df = interval_df.copy()
    df["log_oop_raw_lag"] = np.log1p(df["total_annual_oop_raw_lag"])

    df, ipw_summary = build_ipw_weights(df)
    ipw_summary.to_csv(IPW_SUMMARY_PATH, sep="\t", index=False)

    def tidy(fit, model_id: str, outcome: str, n: int, n_ids: int, shock_quantile: float | None, weighted: bool) -> list[dict[str, object]]:
        rows = []
        conf = fit.conf_int()
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
                    "n_obs": int(n),
                    "n_ids": int(n_ids),
                    "r2": float(getattr(fit, "rsquared", np.nan)),
                    "shock_quantile": shock_quantile,
                    "weighted_ipw": int(weighted),
                }
            )
        return rows

    out_rows: list[dict[str, object]] = []

    # Fit per shock threshold definition (quantiles vary by wave_lag)
    for q in sorted(shock_df["quantile"].unique()):
        threshold_map = {
            int(r.wave_lag): float(r.threshold_total_annual_oop_raw)
            for r in shock_df[shock_df["quantile"] == q].itertuples(index=False)
        }
        shock_col = f"oop_shock_q{str(q).replace('.', '_')}_lag"
        df[shock_col] = np.where(
            df["wave_lag"].isin(threshold_map.keys()),
            df["total_annual_oop_raw_lag"] >= df["wave_lag"].map(threshold_map),
            np.nan,
        ).astype("float")

        common_required = [
            shock_col,
            "adl5_lag",
            "core_disease_count_lag",
            "baseline_age",
            "sex",
            "education_c",
            "residence_rural_w1",
            "interval",
        ]

        # Primary: change model (per-year ADL worsening)
        # (a) log+shock
        change_df = df.dropna(subset=["delta_adl_per_year", "log_oop_raw_lag", *common_required]).copy()
        change_model = smf.ols(
            f"delta_adl_per_year ~ log_oop_raw_lag + {shock_col} + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
            data=change_df,
        ).fit(cov_type="cluster", cov_kwds={"groups": change_df["ID"]})
        out_rows.extend(
            tidy(
                change_model,
                f"change_cluster_ols_q{q}",
                "delta_adl_per_year",
                len(change_df),
                change_df["ID"].nunique(),
                float(q),
                weighted=False,
            )
        )

        # (b) shock-only (threshold robustness without conditioning on continuous level)
        change_so_df = df.dropna(subset=["delta_adl_per_year", *common_required]).copy()
        change_so = smf.ols(
            f"delta_adl_per_year ~ {shock_col} + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
            data=change_so_df,
        ).fit(cov_type="cluster", cov_kwds={"groups": change_so_df["ID"]})
        out_rows.extend(
            tidy(
                change_so,
                f"change_shock_only_cluster_ols_q{q}",
                "delta_adl_per_year",
                len(change_so_df),
                change_so_df["ID"].nunique(),
                float(q),
                weighted=False,
            )
        )

        # IPW sensitivity for change model (requires ipw)
        change_w = change_df.dropna(subset=["ipw"]).copy()
        if not change_w.empty:
            change_ipw = smf.wls(
                f"delta_adl_per_year ~ log_oop_raw_lag + {shock_col} + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
                data=change_w,
                weights=change_w["ipw"],
            ).fit(cov_type="cluster", cov_kwds={"groups": change_w["ID"]})
            out_rows.extend(
                tidy(
                    change_ipw,
                    f"change_ipw_wls_q{q}",
                    "delta_adl_per_year",
                    len(change_w),
                    change_w["ID"].nunique(),
                    float(q),
                    weighted=True,
                )
            )

        # IPW sensitivity for shock-only model
        change_so_w = change_so_df.dropna(subset=["ipw"]).copy()
        if not change_so_w.empty:
            change_so_ipw = smf.wls(
                f"delta_adl_per_year ~ {shock_col} + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
                data=change_so_w,
                weights=change_so_w["ipw"],
            ).fit(cov_type="cluster", cov_kwds={"groups": change_so_w["ID"]})
            out_rows.extend(
                tidy(
                    change_so_ipw,
                    f"change_shock_only_ipw_wls_q{q}",
                    "delta_adl_per_year",
                    len(change_so_w),
                    change_so_w["ID"].nunique(),
                    float(q),
                    weighted=True,
                )
            )

        # Secondary: lagged level model
        level_df = df.dropna(subset=["adl5", "log_oop_raw_lag", *common_required]).copy()
        level_model = smf.ols(
            f"adl5 ~ log_oop_raw_lag + {shock_col} + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
            data=level_df,
        ).fit(cov_type="cluster", cov_kwds={"groups": level_df["ID"]})
        out_rows.extend(
            tidy(
                level_model,
                f"lagged_level_cluster_ols_q{q}",
                "adl5",
                len(level_df),
                level_df["ID"].nunique(),
                float(q),
                weighted=False,
            )
        )

        level_w = level_df.dropna(subset=["ipw"]).copy()
        if not level_w.empty:
            level_ipw = smf.wls(
                f"adl5 ~ log_oop_raw_lag + {shock_col} + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
                data=level_w,
                weights=level_w["ipw"],
            ).fit(cov_type="cluster", cov_kwds={"groups": level_w["ID"]})
            out_rows.extend(
                tidy(
                    level_ipw,
                    f"lagged_level_ipw_wls_q{q}",
                    "adl5",
                    len(level_w),
                    level_w["ID"].nunique(),
                    float(q),
                    weighted=True,
                )
            )

    return pd.DataFrame(out_rows)


def main() -> None:
    ensure_dirs()
    long_df = pd.read_csv(INPUT_PATH, sep="\t")
    interval_df = build_intervals(long_df)

    eligibility = build_eligibility(interval_df)
    eligibility.to_csv(ELIGIBILITY_PATH, sep="\t", index=False)

    shock_df = shock_thresholds(interval_df, quantiles=[0.90, 0.95, 0.975, 0.99])
    shock_df.to_csv(SHOCK_THRESHOLDS_PATH, sep="\t", index=False)

    model_df = fit_models(interval_df, shock_df)
    model_df.to_csv(MODEL_PATH, sep="\t", index=False)

    print(f"Wrote {ELIGIBILITY_PATH}")
    print(f"Wrote {SHOCK_THRESHOLDS_PATH}")
    print(f"Wrote {IPW_SUMMARY_PATH}")
    print(f"Wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
