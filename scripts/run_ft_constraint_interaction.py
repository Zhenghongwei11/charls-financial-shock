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

QC_DIR = ROOT / "results" / "qc"
EFFECT_DIR = ROOT / "results" / "effect_sizes"

SAMPLE_PATH = QC_DIR / "ft_constraint_interaction_sample.tsv"
OUT_PATH = EFFECT_DIR / "ft_constraint_interaction.tsv"

WAVE_TO_YEARS = {1: 0, 2: 2, 3: 4, 4: 7}


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    EFFECT_DIR.mkdir(parents=True, exist_ok=True)


def add_low_wealth(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    wealth = out["ses_wealth_w1"]
    out["wealth_positive"] = np.where(wealth.notna(), (wealth > 0).astype(int), np.nan)
    wealth_pos = wealth[wealth > 0]
    median = float(wealth_pos.median()) if not wealth_pos.empty else np.nan
    out["low_wealth"] = np.where(wealth > 0, (wealth <= median).astype(int), np.nan)

    # 3-level group: low/high among positive, plus missing_or_nonpositive
    out["wealth_group"] = "missing_or_nonpositive"
    out.loc[out["low_wealth"] == 1, "wealth_group"] = "low"
    out.loc[(out["low_wealth"] == 0) & (out["wealth_positive"] == 1), "wealth_group"] = "high"
    out["wealth_group"] = out["wealth_group"].astype("category")
    return out


def build_intervals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t_years"] = out["wave"].map(WAVE_TO_YEARS).astype(float)
    out = out.sort_values(["ID", "t_years"]).reset_index(drop=True)

    # Conditional inpatient cost-forgoing: only defined among those who needed inpatient care but did not get it.
    out["inpatient_forgone_cost_cond_raw"] = np.where(
        out["inpatient_need_but_no_raw"] == 1,
        out["inpatient_forgone_cost_raw"],
        np.nan,
    )

    g = out.groupby("ID", sort=False)
    out["wave_lag"] = g["wave"].shift(1)
    out["interval"] = out["wave_lag"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})

    out["doctor_visit_any_lag"] = g["doctor_visit_any_wave"].shift(1)
    out["doctor_visit_count_lag"] = g["doctor_visit_count_wave"].shift(1)
    out["hospital_stay_any_lag"] = g["hospital_stay_any_wave"].shift(1)
    out["outpatient_forgone_cost_lag"] = g["outpatient_forgone_cost_raw"].shift(1)
    out["inpatient_need_but_no_lag"] = g["inpatient_need_but_no_raw"].shift(1)
    out["inpatient_forgone_cost_lag"] = g["inpatient_forgone_cost_raw"].shift(1)
    out["inpatient_forgone_cost_cond_lag"] = g["inpatient_forgone_cost_cond_raw"].shift(1)
    out["adl5_lag"] = g["adl5"].shift(1)
    out["core_disease_count_lag"] = g["core_disease_count"].shift(1)
    out["oop_raw_lag"] = g["total_annual_oop_raw"].shift(1)
    out["burden_ratio_raw_lag"] = g["ft_burden_ratio_raw"].shift(1)
    out = out[out["interval"].notna()].copy()
    out["log_oop_raw_lag"] = np.log1p(out["oop_raw_lag"])
    out["log_burden_ratio_raw_lag"] = np.log1p(out["burden_ratio_raw_lag"])
    return out


def add_shock_p95(interval_df: pd.DataFrame, shock_df: pd.DataFrame) -> pd.DataFrame:
    thr = shock_df[shock_df["quantile"] == 0.95].copy()
    threshold_map = {int(r.wave_lag): float(r.threshold_total_annual_oop_raw) for r in thr.itertuples(index=False)}
    out = interval_df.copy()
    out["oop_shock_p95_lag"] = np.where(
        out["wave_lag"].isin(threshold_map.keys()),
        out["oop_raw_lag"] >= out["wave_lag"].map(threshold_map),
        np.nan,
    ).astype("float")
    return out


def add_burden_shock_p95(interval_df: pd.DataFrame, shock_df: pd.DataFrame) -> pd.DataFrame:
    thr = shock_df[shock_df["quantile"] == 0.95].copy()
    threshold_map = {int(r.wave_lag): float(r.threshold_ft_burden_ratio_raw) for r in thr.itertuples(index=False) if pd.notna(r.threshold_ft_burden_ratio_raw)}
    out = interval_df.copy()
    eligible = out["wave_lag"].isin(threshold_map.keys()) & out["burden_ratio_raw_lag"].notna()
    out["burden_shock_p95_lag"] = np.where(
        eligible,
        out["burden_ratio_raw_lag"] >= out["wave_lag"].map(threshold_map),
        np.nan,
    ).astype("float")
    return out


def tidy_fit(fit, model_id: str, outcome: str, dataset: str, n_obs: int, n_ids: int) -> pd.DataFrame:
    conf = fit.conf_int()
    rows = []
    for term in fit.params.index:
        if term not in conf.index:
            continue
        rows.append(
            {
                "dataset": dataset,
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

    df = add_low_wealth(long_df)
    interval_df = add_burden_shock_p95(add_shock_p95(build_intervals(df), shock_df), shock_df)

    # Sample QC
    sample = (
        interval_df.groupby(["interval", "wealth_group"], as_index=False)
        .agg(
            n_rows=("ID", "size"),
            n_ids=("ID", "nunique"),
            shock_prev=("oop_shock_p95_lag", "mean"),
            oop_missing_pct=("oop_raw_lag", lambda s: float(s.isna().mean())),
            outpatient_forgone_cost_nonmissing_pct=("outpatient_forgone_cost_raw", lambda s: float(s.notna().mean())),
            inpatient_need_but_no_nonmissing_pct=("inpatient_need_but_no_raw", lambda s: float(s.notna().mean())),
            inpatient_forgone_cost_nonmissing_pct=("inpatient_forgone_cost_raw", lambda s: float(s.notna().mean())),
            inpatient_forgone_cost_cond_nonmissing_pct=("inpatient_forgone_cost_cond_raw", lambda s: float(s.notna().mean())),
            outpatient_forgone_cost_mean=("outpatient_forgone_cost_raw", "mean"),
            inpatient_need_but_no_mean=("inpatient_need_but_no_raw", "mean"),
            inpatient_forgone_cost_mean=("inpatient_forgone_cost_raw", "mean"),
            inpatient_forgone_cost_cond_mean=("inpatient_forgone_cost_cond_raw", "mean"),
        )
    )
    sample.to_csv(SAMPLE_PATH, sep="\t", index=False)

    rows = []
    required = [
        "oop_shock_p95_lag",
        "log_oop_raw_lag",
        "adl5_lag",
        "core_disease_count_lag",
        "baseline_age",
        "sex",
        "education_c",
        "residence_rural_w1",
        "interval",
        "low_wealth",
    ]

    # Constraint interaction on any-visit (logit, cluster-robust)
    doc_any = interval_df.dropna(subset=["doctor_visit_any_wave", "doctor_visit_any_lag", *required]).copy()
    fit = smf.glm(
        "doctor_visit_any_wave ~ oop_shock_p95_lag * low_wealth + log_oop_raw_lag + doctor_visit_any_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=doc_any,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": doc_any["ID"]}, disp=False, maxiter=200)
    rows.append(tidy_fit(fit, "doc_any_logit_cluster", "doctor_visit_any_wave", "balanced_ft", int(fit.nobs), int(doc_any["ID"].nunique())))

    # Constraint interaction on visit count (Poisson GEE; interpret as intensity)
    doc_cnt = interval_df.dropna(subset=["doctor_visit_count_wave", "doctor_visit_count_lag", *required]).copy()
    doc_cnt = doc_cnt[doc_cnt["doctor_visit_count_wave"] >= 0].copy()
    gee = smf.gee(
        "doctor_visit_count_wave ~ oop_shock_p95_lag * low_wealth + log_oop_raw_lag + doctor_visit_count_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        groups="ID",
        data=doc_cnt,
        family=sm.families.Poisson(),
    ).fit()
    rows.append(tidy_fit(gee, "doc_count_pois_gee", "doctor_visit_count_wave", "balanced_ft", int(gee.nobs), int(doc_cnt["ID"].nunique())))

    # Constraint interaction on cost-related outpatient forgoing (asked subset only; logit, cluster-robust)
    # NOTE: do not condition on lagged forgoing, because outpatient-forgoing is only observed in a small skip-pattern subset.
    out_forg = interval_df.dropna(subset=["outpatient_forgone_cost_raw", *required]).copy()
    fit = smf.glm(
        "outpatient_forgone_cost_raw ~ oop_shock_p95_lag * low_wealth + log_oop_raw_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=out_forg,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": out_forg["ID"]}, disp=False, maxiter=200)
    rows.append(tidy_fit(fit, "out_forgone_cost_logit_cluster", "outpatient_forgone_cost_raw", "balanced_ft", int(fit.nobs), int(out_forg["ID"].nunique())))

    # Burden-based shock (relative to baseline wealth) as sensitivity for cost-forgoing outcomes
    out_forg_b = interval_df.dropna(subset=["outpatient_forgone_cost_raw", "burden_shock_p95_lag", "log_oop_raw_lag", "adl5_lag", "core_disease_count_lag", "baseline_age", "sex", "education_c", "residence_rural_w1", "interval", "low_wealth"]).copy()
    fit = smf.glm(
        "outpatient_forgone_cost_raw ~ burden_shock_p95_lag + low_wealth + log_oop_raw_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=out_forg_b,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": out_forg_b["ID"]}, disp=False, maxiter=200)
    rows.append(tidy_fit(fit, "out_forgone_cost_logit_cluster_burden", "outpatient_forgone_cost_raw", "balanced_ft", int(fit.nobs), int(out_forg_b["ID"].nunique())))

    # Constraint interaction on inpatient need-but-no-care (logit, cluster-robust)
    in_need = interval_df.dropna(subset=["inpatient_need_but_no_raw", "inpatient_need_but_no_lag", *required]).copy()
    fit = smf.glm(
        "inpatient_need_but_no_raw ~ oop_shock_p95_lag * low_wealth + log_oop_raw_lag + inpatient_need_but_no_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=in_need,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": in_need["ID"]}, disp=False, maxiter=200)
    rows.append(tidy_fit(fit, "in_need_but_no_logit_cluster", "inpatient_need_but_no_raw", "balanced_ft", int(fit.nobs), int(in_need["ID"].nunique())))

    # Constraint interaction on cost-related inpatient forgoing (population-level indicator)
    in_cost = interval_df.dropna(subset=["inpatient_forgone_cost_raw", "inpatient_forgone_cost_lag", *required]).copy()
    fit = smf.glm(
        "inpatient_forgone_cost_raw ~ oop_shock_p95_lag * low_wealth + log_oop_raw_lag + inpatient_forgone_cost_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=in_cost,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": in_cost["ID"]}, disp=False, maxiter=200)
    rows.append(tidy_fit(fit, "in_forgone_cost_logit_cluster", "inpatient_forgone_cost_raw", "balanced_ft", int(fit.nobs), int(in_cost["ID"].nunique())))

    # Conditional inpatient cost-forgoing among those with unmet inpatient need
    in_cost_cond = interval_df.dropna(subset=["inpatient_forgone_cost_cond_raw", "inpatient_forgone_cost_cond_lag", *required]).copy()
    fit = smf.glm(
        "inpatient_forgone_cost_cond_raw ~ oop_shock_p95_lag * low_wealth + log_oop_raw_lag + inpatient_forgone_cost_cond_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=in_cost_cond,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": in_cost_cond["ID"]}, disp=False, maxiter=200)
    rows.append(tidy_fit(fit, "in_forgone_cost_cond_logit_cluster", "inpatient_forgone_cost_cond_raw", "balanced_ft", int(fit.nobs), int(in_cost_cond["ID"].nunique())))

    in_need_b = interval_df.dropna(subset=["inpatient_need_but_no_raw", "burden_shock_p95_lag", "log_oop_raw_lag", "adl5_lag", "core_disease_count_lag", "baseline_age", "sex", "education_c", "residence_rural_w1", "interval", "low_wealth"]).copy()
    fit = smf.glm(
        "inpatient_need_but_no_raw ~ burden_shock_p95_lag + low_wealth + log_oop_raw_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=in_need_b,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": in_need_b["ID"]}, disp=False, maxiter=200)
    rows.append(tidy_fit(fit, "in_need_but_no_logit_cluster_burden", "inpatient_need_but_no_raw", "balanced_ft", int(fit.nobs), int(in_need_b["ID"].nunique())))

    in_cost_b = interval_df.dropna(subset=["inpatient_forgone_cost_raw", "burden_shock_p95_lag", "log_oop_raw_lag", "adl5_lag", "core_disease_count_lag", "baseline_age", "sex", "education_c", "residence_rural_w1", "interval", "low_wealth"]).copy()
    fit = smf.glm(
        "inpatient_forgone_cost_raw ~ burden_shock_p95_lag + low_wealth + log_oop_raw_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=in_cost_b,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": in_cost_b["ID"]}, disp=False, maxiter=200)
    rows.append(tidy_fit(fit, "in_forgone_cost_logit_cluster_burden", "inpatient_forgone_cost_raw", "balanced_ft", int(fit.nobs), int(in_cost_b["ID"].nunique())))

    in_cost_cond_b = interval_df.dropna(subset=["inpatient_forgone_cost_cond_raw", "burden_shock_p95_lag", "log_oop_raw_lag", "adl5_lag", "core_disease_count_lag", "baseline_age", "sex", "education_c", "residence_rural_w1", "interval", "low_wealth"]).copy()
    fit = smf.glm(
        "inpatient_forgone_cost_cond_raw ~ burden_shock_p95_lag + low_wealth + log_oop_raw_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)",
        data=in_cost_cond_b,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": in_cost_cond_b["ID"]}, disp=False, maxiter=200)
    rows.append(tidy_fit(fit, "in_forgone_cost_cond_logit_cluster_burden", "inpatient_forgone_cost_cond_raw", "balanced_ft", int(fit.nobs), int(in_cost_cond_b["ID"].nunique())))

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Wrote {SAMPLE_PATH}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
