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

SAMPLE_PATH = QC_DIR / "ft_mediation_chain_sample.tsv"
ATTENUATION_PATH = QC_DIR / "ft_mediation_chain_attenuation.tsv"
ATTENUATION_BURDEN_PATH = QC_DIR / "ft_mediation_chain_attenuation_burden.tsv"
OUT_PATH = EFFECT_DIR / "ft_mediation_chain.tsv"

WAVE_TO_YEARS = {1: 0, 2: 2, 3: 4, 4: 7}


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    EFFECT_DIR.mkdir(parents=True, exist_ok=True)


def build_lag_lead(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["t_years"] = out["wave"].map(WAVE_TO_YEARS).astype(float)
    out = out.sort_values(["ID", "t_years"]).reset_index(drop=True)

    # Conditional inpatient cost-forgoing: only defined among those with unmet inpatient need.
    out["inpatient_forgone_cost_cond_raw"] = np.where(
        out["inpatient_need_but_no_raw"] == 1,
        out["inpatient_forgone_cost_raw"],
        np.nan,
    )

    g = out.groupby("ID", sort=False)
    # lag (t-1)
    out["wave_lag"] = g["wave"].shift(1)
    out["interval_lag"] = out["wave_lag"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})
    out["adl5_lag"] = g["adl5"].shift(1)
    out["core_disease_count_lag"] = g["core_disease_count"].shift(1)
    out["oop_raw_lag"] = g["total_annual_oop_raw"].shift(1)
    out["burden_ratio_raw_lag"] = g["ft_burden_ratio_raw"].shift(1)
    out["doctor_visit_any_lag"] = g["doctor_visit_any_wave"].shift(1)
    out["doctor_visit_count_lag"] = g["doctor_visit_count_wave"].shift(1)
    out["hospital_stay_any_lag"] = g["hospital_stay_any_wave"].shift(1)
    out["outpatient_forgone_cost_lag"] = g["outpatient_forgone_cost_raw"].shift(1)
    out["inpatient_need_but_no_lag"] = g["inpatient_need_but_no_raw"].shift(1)
    out["inpatient_forgone_cost_lag"] = g["inpatient_forgone_cost_raw"].shift(1)
    out["inpatient_forgone_cost_cond_lag"] = g["inpatient_forgone_cost_cond_raw"].shift(1)
    out["any_contact_wave"] = out[["doctor_visit_any_wave", "hospital_stay_any_wave"]].max(axis=1, skipna=True)
    out.loc[out[["doctor_visit_any_wave", "hospital_stay_any_wave"]].isna().all(axis=1), "any_contact_wave"] = np.nan
    out["any_contact_lag"] = g["any_contact_wave"].shift(1)

    # lead (t+1)
    out["wave_lead"] = g["wave"].shift(-1)
    out["interval_lead"] = out["wave"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})
    out["adl5_lead"] = g["adl5"].shift(-1)
    out["t_years_lead"] = g["t_years"].shift(-1)
    out["dt_years_lead"] = out["t_years_lead"] - out["t_years"]
    out["delta_adl_lead"] = out["adl5_lead"] - out["adl5"]
    out["delta_adl_per_year_lead"] = out["delta_adl_lead"] / out["dt_years_lead"]

    out["log_oop_raw_lag"] = np.log1p(out["oop_raw_lag"])
    out["log_burden_ratio_raw_lag"] = np.log1p(out["burden_ratio_raw_lag"])
    return out


def add_shock_p95(panel: pd.DataFrame, shock_df: pd.DataFrame) -> pd.DataFrame:
    thr = shock_df[shock_df["quantile"] == 0.95].copy()
    threshold_map = {int(r.wave_lag): float(r.threshold_total_annual_oop_raw) for r in thr.itertuples(index=False)}
    out = panel.copy()
    out["oop_shock_p95_lag"] = np.where(
        out["wave_lag"].isin(threshold_map.keys()),
        out["oop_raw_lag"] >= out["wave_lag"].map(threshold_map),
        np.nan,
    ).astype("float")
    return out


def add_burden_shock_p95(panel: pd.DataFrame, shock_df: pd.DataFrame) -> pd.DataFrame:
    thr = shock_df[shock_df["quantile"] == 0.95].copy()
    threshold_map = {int(r.wave_lag): float(r.threshold_ft_burden_ratio_raw) for r in thr.itertuples(index=False) if pd.notna(r.threshold_ft_burden_ratio_raw)}
    out = panel.copy()
    eligible = out["wave_lag"].isin(threshold_map.keys()) & out["burden_ratio_raw_lag"].notna()
    out["burden_shock_p95_lag"] = np.where(
        eligible,
        out["burden_ratio_raw_lag"] >= out["wave_lag"].map(threshold_map),
        np.nan,
    ).astype("float")
    return out


def tidy_fit(fit, module: str, model_id: str, outcome: str, n_obs: int, n_ids: int) -> pd.DataFrame:
    conf = fit.conf_int()
    rows = []
    for term in fit.params.index:
        if term not in conf.index:
            continue
        rows.append(
            {
                "module": module,
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

def coef_row(df: pd.DataFrame, module: str, model_id: str, term: str) -> pd.Series | None:
    sub = df[(df["module"] == module) & (df["model_id"] == model_id) & (df["term"] == term)]
    if sub.empty:
        return None
    return sub.iloc[0]


def main() -> None:
    ensure_dirs()
    long_df = pd.read_csv(FT_LONG_PATH, sep="\t")
    shock_df = pd.read_csv(THRESH_PATH, sep="\t")

    # Structural zero for counts when no visit and count is missing
    long_df.loc[
        (long_df["doctor_visit_any_wave"] == 0) & long_df["doctor_visit_count_wave"].isna(),
        "doctor_visit_count_wave",
    ] = 0.0

    panel = add_burden_shock_p95(add_shock_p95(build_lag_lead(long_df), shock_df), shock_df)

    # -------------------------
    # Leg 1: shock_{t-1} -> utilization_t (two-part: any + intensity)
    # -------------------------
    leg1_common = [
        "oop_shock_p95_lag",
        "log_oop_raw_lag",
        "adl5_lag",
        "core_disease_count_lag",
        "baseline_age",
        "sex",
        "education_c",
        "residence_rural_w1",
        "interval_lag",
    ]

    leg1_doc_any = panel.dropna(subset=["doctor_visit_any_wave", "doctor_visit_any_lag", *leg1_common]).copy()
    leg1_doc_any = leg1_doc_any[leg1_doc_any["interval_lag"].notna()].copy()

    leg1_doc_cnt = panel.dropna(subset=["doctor_visit_count_wave", "doctor_visit_count_lag", *leg1_common]).copy()
    leg1_doc_cnt = leg1_doc_cnt[leg1_doc_cnt["interval_lag"].notna()].copy()
    leg1_doc_cnt = leg1_doc_cnt[leg1_doc_cnt["doctor_visit_count_wave"] >= 0].copy()
    # Intensity among users only (conditional count)
    leg1_doc_cnt_users = leg1_doc_cnt[leg1_doc_cnt["doctor_visit_any_wave"] == 1].copy()

    leg1_hosp_any = panel.dropna(subset=["hospital_stay_any_wave", "hospital_stay_any_lag", *leg1_common]).copy()
    leg1_hosp_any = leg1_hosp_any[leg1_hosp_any["interval_lag"].notna()].copy()

    leg1_any_contact = panel.dropna(subset=["any_contact_wave", "any_contact_lag", *leg1_common]).copy()
    leg1_any_contact = leg1_any_contact[leg1_any_contact["interval_lag"].notna()].copy()

    # Cost-related forgoing mediators (waves 1-3 only; wave 4 is NA by construction)
    leg1_out_forg_cost = panel.dropna(subset=["outpatient_forgone_cost_raw", *leg1_common]).copy()
    leg1_out_forg_cost = leg1_out_forg_cost[leg1_out_forg_cost["interval_lag"].notna()].copy()

    leg1_in_need_no = panel.dropna(subset=["inpatient_need_but_no_raw", "inpatient_need_but_no_lag", *leg1_common]).copy()
    leg1_in_need_no = leg1_in_need_no[leg1_in_need_no["interval_lag"].notna()].copy()

    leg1_in_forg_cost = panel.dropna(subset=["inpatient_forgone_cost_raw", "inpatient_forgone_cost_lag", *leg1_common]).copy()
    leg1_in_forg_cost = leg1_in_forg_cost[leg1_in_forg_cost["interval_lag"].notna()].copy()

    leg1_in_forg_cost_cond = panel.dropna(subset=["inpatient_forgone_cost_cond_raw", "inpatient_forgone_cost_cond_lag", *leg1_common]).copy()
    leg1_in_forg_cost_cond = leg1_in_forg_cost_cond[leg1_in_forg_cost_cond["interval_lag"].notna()].copy()

    # Burden (continuous) leg1 samples (wealth>0 required; tends to be smaller)
    leg1_common_burden = [
        "log_burden_ratio_raw_lag",
        "adl5_lag",
        "core_disease_count_lag",
        "baseline_age",
        "sex",
        "education_c",
        "residence_rural_w1",
        "interval_lag",
    ]
    leg1_out_forg_cost_b = panel.dropna(subset=["outpatient_forgone_cost_raw", *leg1_common_burden]).copy()
    leg1_out_forg_cost_b = leg1_out_forg_cost_b[leg1_out_forg_cost_b["interval_lag"].notna()].copy()

    leg1_in_need_no_b = panel.dropna(subset=["inpatient_need_but_no_raw", "inpatient_need_but_no_lag", *leg1_common_burden]).copy()
    leg1_in_need_no_b = leg1_in_need_no_b[leg1_in_need_no_b["interval_lag"].notna()].copy()

    leg1_in_forg_cost_b = panel.dropna(subset=["inpatient_forgone_cost_raw", "inpatient_forgone_cost_lag", *leg1_common_burden]).copy()
    leg1_in_forg_cost_b = leg1_in_forg_cost_b[leg1_in_forg_cost_b["interval_lag"].notna()].copy()

    leg1_in_forg_cost_cond_b = panel.dropna(subset=["inpatient_forgone_cost_cond_raw", "inpatient_forgone_cost_cond_lag", *leg1_common_burden]).copy()
    leg1_in_forg_cost_cond_b = leg1_in_forg_cost_cond_b[leg1_in_forg_cost_cond_b["interval_lag"].notna()].copy()

    # -------------------------
    # Leg 2: utilization_t -> ADL_{t+1} level, with shock_{t-1} included
    # Only waves 2,3 can form (t-1, t, t+1) triples in this 4-wave panel.
    # -------------------------
    leg2_common = [
        "oop_shock_p95_lag",
        "log_oop_raw_lag",
        "adl5",
        "adl5_lag",
        "core_disease_count_lag",
        "adl5_lead",
        "baseline_age",
        "sex",
        "education_c",
        "residence_rural_w1",
        "interval_lag",
        "interval_lead",
    ]
    leg2 = panel.dropna(subset=leg2_common).copy()
    leg2 = leg2[leg2["wave"].isin([2, 3])].copy()

    leg2_common_burden = [
        "log_burden_ratio_raw_lag",
        "adl5",
        "adl5_lag",
        "core_disease_count_lag",
        "adl5_lead",
        "baseline_age",
        "sex",
        "education_c",
        "residence_rural_w1",
        "interval_lag",
        "interval_lead",
    ]
    leg2_b = panel.dropna(subset=leg2_common_burden).copy()
    leg2_b = leg2_b[leg2_b["wave"].isin([2, 3])].copy()

    # QC sample table (module-specific)
    sample = pd.DataFrame(
        [
            {"module": "leg1_doc_any", "n_obs": int(len(leg1_doc_any)), "n_ids": int(leg1_doc_any["ID"].nunique())},
            {"module": "leg1_doc_cnt", "n_obs": int(len(leg1_doc_cnt)), "n_ids": int(leg1_doc_cnt["ID"].nunique())},
            {"module": "leg1_doc_cnt_users", "n_obs": int(len(leg1_doc_cnt_users)), "n_ids": int(leg1_doc_cnt_users["ID"].nunique())},
            {"module": "leg1_hosp_any", "n_obs": int(len(leg1_hosp_any)), "n_ids": int(leg1_hosp_any["ID"].nunique())},
            {"module": "leg1_any_contact", "n_obs": int(len(leg1_any_contact)), "n_ids": int(leg1_any_contact["ID"].nunique())},
            {"module": "leg1_out_forg_cost", "n_obs": int(len(leg1_out_forg_cost)), "n_ids": int(leg1_out_forg_cost["ID"].nunique())},
            {"module": "leg1_in_need_no", "n_obs": int(len(leg1_in_need_no)), "n_ids": int(leg1_in_need_no["ID"].nunique())},
            {"module": "leg1_in_forg_cost", "n_obs": int(len(leg1_in_forg_cost)), "n_ids": int(leg1_in_forg_cost["ID"].nunique())},
            {"module": "leg1_in_forg_cost_cond", "n_obs": int(len(leg1_in_forg_cost_cond)), "n_ids": int(leg1_in_forg_cost_cond["ID"].nunique())},
            {"module": "leg1_out_forg_cost_burden", "n_obs": int(len(leg1_out_forg_cost_b)), "n_ids": int(leg1_out_forg_cost_b["ID"].nunique())},
            {"module": "leg1_in_need_no_burden", "n_obs": int(len(leg1_in_need_no_b)), "n_ids": int(leg1_in_need_no_b["ID"].nunique())},
            {"module": "leg1_in_forg_cost_burden", "n_obs": int(len(leg1_in_forg_cost_b)), "n_ids": int(leg1_in_forg_cost_b["ID"].nunique())},
            {"module": "leg1_in_forg_cost_cond_burden", "n_obs": int(len(leg1_in_forg_cost_cond_b)), "n_ids": int(leg1_in_forg_cost_cond_b["ID"].nunique())},
            {"module": "leg2_adl_lead_level", "n_obs": int(len(leg2)), "n_ids": int(leg2["ID"].nunique())},
            {"module": "leg2_adl_lead_level_burden", "n_obs": int(len(leg2_b)), "n_ids": int(leg2_b["ID"].nunique())},
        ]
    )
    sample.to_csv(SAMPLE_PATH, sep="\t", index=False)

    outs: list[pd.DataFrame] = []

    # Leg 1 models
    glm_doc_any = smf.glm(
        "doctor_visit_any_wave ~ oop_shock_p95_lag + log_oop_raw_lag + doctor_visit_any_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_doc_any,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_doc_any["ID"]}, disp=False, maxiter=200)
    outs.append(tidy_fit(glm_doc_any, "leg1", "doc_any_logit_cluster", "doctor_visit_any_wave", int(glm_doc_any.nobs), int(leg1_doc_any["ID"].nunique())))

    gee_doc_cnt = smf.gee(
        "doctor_visit_count_wave ~ oop_shock_p95_lag + log_oop_raw_lag + doctor_visit_count_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        groups="ID",
        data=leg1_doc_cnt,
        family=sm.families.Poisson(),
    ).fit()
    outs.append(tidy_fit(gee_doc_cnt, "leg1", "doc_count_pois_gee", "doctor_visit_count_wave", int(gee_doc_cnt.nobs), int(leg1_doc_cnt["ID"].nunique())))

    gee_doc_cnt_users = smf.gee(
        "doctor_visit_count_wave ~ oop_shock_p95_lag + log_oop_raw_lag + doctor_visit_count_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        groups="ID",
        data=leg1_doc_cnt_users,
        family=sm.families.Poisson(),
    ).fit()
    outs.append(
        tidy_fit(
            gee_doc_cnt_users,
            "leg1",
            "doc_count_users_pois_gee",
            "doctor_visit_count_wave",
            int(gee_doc_cnt_users.nobs),
            int(leg1_doc_cnt_users["ID"].nunique()),
        )
    )

    glm_hosp_any = smf.glm(
        "hospital_stay_any_wave ~ oop_shock_p95_lag + log_oop_raw_lag + hospital_stay_any_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_hosp_any,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_hosp_any["ID"]}, disp=False, maxiter=200)
    outs.append(tidy_fit(glm_hosp_any, "leg1", "hosp_any_logit_cluster", "hospital_stay_any_wave", int(glm_hosp_any.nobs), int(leg1_hosp_any["ID"].nunique())))

    glm_any_contact = smf.glm(
        "any_contact_wave ~ oop_shock_p95_lag + log_oop_raw_lag + any_contact_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_any_contact,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_any_contact["ID"]}, disp=False, maxiter=200)
    outs.append(tidy_fit(glm_any_contact, "leg1", "any_contact_logit_cluster", "any_contact_wave", int(glm_any_contact.nobs), int(leg1_any_contact["ID"].nunique())))

    glm_out_forg_cost = smf.glm(
        "outpatient_forgone_cost_raw ~ oop_shock_p95_lag + log_oop_raw_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_out_forg_cost,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_out_forg_cost["ID"]}, disp=False, maxiter=200)
    outs.append(
        tidy_fit(
            glm_out_forg_cost,
            "leg1",
            "out_forg_cost_logit_cluster",
            "outpatient_forgone_cost_raw",
            int(glm_out_forg_cost.nobs),
            int(leg1_out_forg_cost["ID"].nunique()),
        )
    )

    glm_in_need_no = smf.glm(
        "inpatient_need_but_no_raw ~ oop_shock_p95_lag + log_oop_raw_lag + inpatient_need_but_no_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_in_need_no,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_in_need_no["ID"]}, disp=False, maxiter=200)
    outs.append(
        tidy_fit(
            glm_in_need_no,
            "leg1",
            "in_need_no_logit_cluster",
            "inpatient_need_but_no_raw",
            int(glm_in_need_no.nobs),
            int(leg1_in_need_no["ID"].nunique()),
        )
    )

    glm_in_forg_cost = smf.glm(
        "inpatient_forgone_cost_raw ~ oop_shock_p95_lag + log_oop_raw_lag + inpatient_forgone_cost_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_in_forg_cost,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_in_forg_cost["ID"]}, disp=False, maxiter=200)
    outs.append(
        tidy_fit(
            glm_in_forg_cost,
            "leg1",
            "in_forg_cost_logit_cluster",
            "inpatient_forgone_cost_raw",
            int(glm_in_forg_cost.nobs),
            int(leg1_in_forg_cost["ID"].nunique()),
        )
    )

    glm_in_forg_cost_cond = smf.glm(
        "inpatient_forgone_cost_cond_raw ~ oop_shock_p95_lag + log_oop_raw_lag + inpatient_forgone_cost_cond_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_in_forg_cost_cond,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_in_forg_cost_cond["ID"]}, disp=False, maxiter=200)
    outs.append(
        tidy_fit(
            glm_in_forg_cost_cond,
            "leg1",
            "in_forg_cost_cond_logit_cluster",
            "inpatient_forgone_cost_cond_raw",
            int(glm_in_forg_cost_cond.nobs),
            int(leg1_in_forg_cost_cond["ID"].nunique()),
        )
    )

    # Burden (continuous) leg1 models (relative to baseline wealth)
    glm_out_forg_cost_b = smf.glm(
        "outpatient_forgone_cost_raw ~ log_burden_ratio_raw_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_out_forg_cost_b,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_out_forg_cost_b["ID"]}, disp=False, maxiter=200)
    outs.append(
        tidy_fit(
            glm_out_forg_cost_b,
            "leg1_burden",
            "out_forg_cost_logit_cluster_burden",
            "outpatient_forgone_cost_raw",
            int(glm_out_forg_cost_b.nobs),
            int(leg1_out_forg_cost_b["ID"].nunique()),
        )
    )

    glm_in_need_no_b = smf.glm(
        "inpatient_need_but_no_raw ~ log_burden_ratio_raw_lag + inpatient_need_but_no_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_in_need_no_b,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_in_need_no_b["ID"]}, disp=False, maxiter=200)
    outs.append(
        tidy_fit(
            glm_in_need_no_b,
            "leg1_burden",
            "in_need_no_logit_cluster_burden",
            "inpatient_need_but_no_raw",
            int(glm_in_need_no_b.nobs),
            int(leg1_in_need_no_b["ID"].nunique()),
        )
    )

    glm_in_forg_cost_b = smf.glm(
        "inpatient_forgone_cost_raw ~ log_burden_ratio_raw_lag + inpatient_forgone_cost_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_in_forg_cost_b,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_in_forg_cost_b["ID"]}, disp=False, maxiter=200)
    outs.append(
        tidy_fit(
            glm_in_forg_cost_b,
            "leg1_burden",
            "in_forg_cost_logit_cluster_burden",
            "inpatient_forgone_cost_raw",
            int(glm_in_forg_cost_b.nobs),
            int(leg1_in_forg_cost_b["ID"].nunique()),
        )
    )

    glm_in_forg_cost_cond_b = smf.glm(
        "inpatient_forgone_cost_cond_raw ~ log_burden_ratio_raw_lag + inpatient_forgone_cost_cond_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_in_forg_cost_cond_b,
        family=sm.families.Binomial(),
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_in_forg_cost_cond_b["ID"]}, disp=False, maxiter=200)
    outs.append(
        tidy_fit(
            glm_in_forg_cost_cond_b,
            "leg1_burden",
            "in_forg_cost_cond_logit_cluster_burden",
            "inpatient_forgone_cost_cond_raw",
            int(glm_in_forg_cost_cond_b.nobs),
            int(leg1_in_forg_cost_cond_b["ID"].nunique()),
        )
    )

    # Leg 2 attenuation checks (each mediator tested separately)
    # 2A baseline (no mediator)
    ols_base = smf.ols(
        "adl5_lead ~ oop_shock_p95_lag + log_oop_raw_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2["ID"]})
    outs.append(tidy_fit(ols_base, "leg2", "adl_lead_no_mediator_cluster_ols", "adl5_lead", int(ols_base.nobs), int(leg2["ID"].nunique())))

    # Burden baseline for attenuation checks (separate, smaller cohort due to wealth eligibility)
    ols_base_b = smf.ols(
        "adl5_lead ~ log_burden_ratio_raw_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2_b,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2_b["ID"]})
    outs.append(tidy_fit(ols_base_b, "leg2_burden", "adl_lead_no_mediator_cluster_ols_burden", "adl5_lead", int(ols_base_b.nobs), int(leg2_b["ID"].nunique())))

    # GEE binomial sensitivity for ADL lead (adl5/5)
    leg2 = leg2.copy()
    leg2["adl5_lead_prop"] = leg2["adl5_lead"] / 5.0
    gee_base = smf.gee(
        "adl5_lead_prop ~ oop_shock_p95_lag + log_oop_raw_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        groups="ID",
        data=leg2,
        family=sm.families.Binomial(),
        weights=np.full(len(leg2), 5.0),
    ).fit()
    outs.append(tidy_fit(gee_base, "leg2", "adl_lead_no_mediator_gee_binomial", "adl5_lead_prop", int(gee_base.nobs), int(leg2["ID"].nunique())))

    attenuation_rows: list[dict[str, object]] = []

    def add_att_matched_abs(mediator: str, subset_df: pd.DataFrame, fit_with) -> None:
        base_fit = smf.ols(
            "adl5_lead ~ oop_shock_p95_lag + log_oop_raw_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
            data=subset_df,
        ).fit(cov_type="cluster", cov_kwds={"groups": subset_df["ID"]})
        attenuation_rows.append(
            {
                "base_scope": "matched",
                "mediator": mediator,
                "shock_est_no_mediator": float(base_fit.params.get("oop_shock_p95_lag", np.nan)),
                "shock_est_with_mediator": float(fit_with.params.get("oop_shock_p95_lag", np.nan)),
                "delta": float(fit_with.params.get("oop_shock_p95_lag", np.nan) - base_fit.params.get("oop_shock_p95_lag", np.nan)),
                "p_no_mediator": float(base_fit.pvalues.get("oop_shock_p95_lag", np.nan)),
                "p_with_mediator": float(fit_with.pvalues.get("oop_shock_p95_lag", np.nan)),
                "n_obs": int(fit_with.nobs),
                "n_ids": int(subset_df["ID"].nunique()),
            }
        )

    att_burden_rows: list[dict[str, object]] = []

    def add_att_matched_burden(mediator: str, subset_df: pd.DataFrame, fit_with) -> None:
        base_fit = smf.ols(
            "adl5_lead ~ log_burden_ratio_raw_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
            data=subset_df,
        ).fit(cov_type="cluster", cov_kwds={"groups": subset_df["ID"]})
        att_burden_rows.append(
            {
                "base_scope": "matched",
                "mediator": mediator,
                "shock_est_no_mediator": float(base_fit.params.get("log_burden_ratio_raw_lag", np.nan)),
                "shock_est_with_mediator": float(fit_with.params.get("log_burden_ratio_raw_lag", np.nan)),
                "delta": float(fit_with.params.get("log_burden_ratio_raw_lag", np.nan) - base_fit.params.get("log_burden_ratio_raw_lag", np.nan)),
                "p_no_mediator": float(base_fit.pvalues.get("log_burden_ratio_raw_lag", np.nan)),
                "p_with_mediator": float(fit_with.pvalues.get("log_burden_ratio_raw_lag", np.nan)),
                "n_obs": int(fit_with.nobs),
                "n_ids": int(subset_df["ID"].nunique()),
            }
        )

    # Mediator: doctor_visit_any_wave (plus lag)
    leg2_doc_any = leg2.dropna(subset=["doctor_visit_any_wave", "doctor_visit_any_lag"]).copy()
    ols_doc_any = smf.ols(
        "adl5_lead ~ oop_shock_p95_lag + log_oop_raw_lag + doctor_visit_any_wave + doctor_visit_any_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2_doc_any,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2_doc_any["ID"]})
    outs.append(tidy_fit(ols_doc_any, "leg2", "adl_lead_with_doc_any_cluster_ols", "adl5_lead", int(ols_doc_any.nobs), int(leg2_doc_any["ID"].nunique())))
    add_att_matched_abs("doc_any", leg2_doc_any, ols_doc_any)
    leg2_doc_any["adl5_lead_prop"] = leg2_doc_any["adl5_lead"] / 5.0
    gee_doc_any = smf.gee(
        "adl5_lead_prop ~ oop_shock_p95_lag + log_oop_raw_lag + doctor_visit_any_wave + doctor_visit_any_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        groups="ID",
        data=leg2_doc_any,
        family=sm.families.Binomial(),
        weights=np.full(len(leg2_doc_any), 5.0),
    ).fit()
    outs.append(tidy_fit(gee_doc_any, "leg2", "adl_lead_with_doc_any_gee_binomial", "adl5_lead_prop", int(gee_doc_any.nobs), int(leg2_doc_any["ID"].nunique())))

    # Mediator: doctor_visit_count_wave (plus lag)
    leg2_doc_cnt = leg2.dropna(subset=["doctor_visit_count_wave", "doctor_visit_count_lag"]).copy()
    leg2_doc_cnt = leg2_doc_cnt[leg2_doc_cnt["doctor_visit_count_wave"] >= 0].copy()
    ols_doc_cnt = smf.ols(
        "adl5_lead ~ oop_shock_p95_lag + log_oop_raw_lag + doctor_visit_count_wave + doctor_visit_count_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2_doc_cnt,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2_doc_cnt["ID"]})
    outs.append(tidy_fit(ols_doc_cnt, "leg2", "adl_lead_with_doc_count_cluster_ols", "adl5_lead", int(ols_doc_cnt.nobs), int(leg2_doc_cnt["ID"].nunique())))
    add_att_matched_abs("doc_count", leg2_doc_cnt, ols_doc_cnt)
    leg2_doc_cnt["adl5_lead_prop"] = leg2_doc_cnt["adl5_lead"] / 5.0
    gee_doc_cnt = smf.gee(
        "adl5_lead_prop ~ oop_shock_p95_lag + log_oop_raw_lag + doctor_visit_count_wave + doctor_visit_count_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        groups="ID",
        data=leg2_doc_cnt,
        family=sm.families.Binomial(),
        weights=np.full(len(leg2_doc_cnt), 5.0),
    ).fit()
    outs.append(tidy_fit(gee_doc_cnt, "leg2", "adl_lead_with_doc_count_gee_binomial", "adl5_lead_prop", int(gee_doc_cnt.nobs), int(leg2_doc_cnt["ID"].nunique())))

    # Mediator: hospital_stay_any_wave (plus lag)
    leg2_hosp_any = leg2.dropna(subset=["hospital_stay_any_wave", "hospital_stay_any_lag"]).copy()
    ols_hosp_any = smf.ols(
        "adl5_lead ~ oop_shock_p95_lag + log_oop_raw_lag + hospital_stay_any_wave + hospital_stay_any_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2_hosp_any,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2_hosp_any["ID"]})
    outs.append(tidy_fit(ols_hosp_any, "leg2", "adl_lead_with_hosp_any_cluster_ols", "adl5_lead", int(ols_hosp_any.nobs), int(leg2_hosp_any["ID"].nunique())))
    add_att_matched_abs("hosp_any", leg2_hosp_any, ols_hosp_any)
    leg2_hosp_any["adl5_lead_prop"] = leg2_hosp_any["adl5_lead"] / 5.0
    gee_hosp_any = smf.gee(
        "adl5_lead_prop ~ oop_shock_p95_lag + log_oop_raw_lag + hospital_stay_any_wave + hospital_stay_any_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        groups="ID",
        data=leg2_hosp_any,
        family=sm.families.Binomial(),
        weights=np.full(len(leg2_hosp_any), 5.0),
    ).fit()
    outs.append(tidy_fit(gee_hosp_any, "leg2", "adl_lead_with_hosp_any_gee_binomial", "adl5_lead_prop", int(gee_hosp_any.nobs), int(leg2_hosp_any["ID"].nunique())))

    # Mediator: any_contact_wave (plus lag)
    leg2_any = leg2.dropna(subset=["any_contact_wave", "any_contact_lag"]).copy()
    ols_any = smf.ols(
        "adl5_lead ~ oop_shock_p95_lag + log_oop_raw_lag + any_contact_wave + any_contact_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2_any,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2_any["ID"]})
    outs.append(tidy_fit(ols_any, "leg2", "adl_lead_with_any_contact_cluster_ols", "adl5_lead", int(ols_any.nobs), int(leg2_any["ID"].nunique())))
    add_att_matched_abs("any_contact", leg2_any, ols_any)
    leg2_any["adl5_lead_prop"] = leg2_any["adl5_lead"] / 5.0
    gee_any = smf.gee(
        "adl5_lead_prop ~ oop_shock_p95_lag + log_oop_raw_lag + any_contact_wave + any_contact_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        groups="ID",
        data=leg2_any,
        family=sm.families.Binomial(),
        weights=np.full(len(leg2_any), 5.0),
    ).fit()
    outs.append(tidy_fit(gee_any, "leg2", "adl_lead_with_any_contact_gee_binomial", "adl5_lead_prop", int(gee_any.nobs), int(leg2_any["ID"].nunique())))

    # Mediator: outpatient_forgone_cost_raw (asked subset only; plus lag)
    leg2_out_forg = leg2.dropna(subset=["outpatient_forgone_cost_raw"]).copy()
    ols_out_forg = smf.ols(
        "adl5_lead ~ oop_shock_p95_lag + log_oop_raw_lag + outpatient_forgone_cost_raw + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2_out_forg,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2_out_forg["ID"]})
    outs.append(tidy_fit(ols_out_forg, "leg2", "adl_lead_with_out_forg_cost_cluster_ols", "adl5_lead", int(ols_out_forg.nobs), int(leg2_out_forg["ID"].nunique())))
    add_att_matched_abs("out_forg_cost", leg2_out_forg, ols_out_forg)

    # Mediator: inpatient_need_but_no_raw (plus lag)
    leg2_in_need = leg2.dropna(subset=["inpatient_need_but_no_raw", "inpatient_need_but_no_lag"]).copy()
    ols_in_need = smf.ols(
        "adl5_lead ~ oop_shock_p95_lag + log_oop_raw_lag + inpatient_need_but_no_raw + inpatient_need_but_no_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2_in_need,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2_in_need["ID"]})
    outs.append(tidy_fit(ols_in_need, "leg2", "adl_lead_with_in_need_no_cluster_ols", "adl5_lead", int(ols_in_need.nobs), int(leg2_in_need["ID"].nunique())))
    add_att_matched_abs("in_need_no", leg2_in_need, ols_in_need)

    # Mediator: inpatient_forgone_cost_raw (population-level indicator; plus lag)
    leg2_in_cost = leg2.dropna(subset=["inpatient_forgone_cost_raw", "inpatient_forgone_cost_lag"]).copy()
    ols_in_cost = smf.ols(
        "adl5_lead ~ oop_shock_p95_lag + log_oop_raw_lag + inpatient_forgone_cost_raw + inpatient_forgone_cost_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2_in_cost,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2_in_cost["ID"]})
    outs.append(tidy_fit(ols_in_cost, "leg2", "adl_lead_with_in_forg_cost_cluster_ols", "adl5_lead", int(ols_in_cost.nobs), int(leg2_in_cost["ID"].nunique())))
    add_att_matched_abs("in_forg_cost", leg2_in_cost, ols_in_cost)

    # Burden-shock mediator models for attenuation (use leg2_b)
    leg2_out_forg_b = leg2_b.dropna(subset=["outpatient_forgone_cost_raw"]).copy()
    ols_out_forg_b = smf.ols(
        "adl5_lead ~ log_burden_ratio_raw_lag + outpatient_forgone_cost_raw + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2_out_forg_b,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2_out_forg_b["ID"]})
    outs.append(
        tidy_fit(
            ols_out_forg_b,
            "leg2_burden",
            "adl_lead_with_out_forg_cost_cluster_ols_burden",
            "adl5_lead",
            int(ols_out_forg_b.nobs),
            int(leg2_out_forg_b["ID"].nunique()),
        )
    )
    add_att_matched_burden("out_forg_cost", leg2_out_forg_b, ols_out_forg_b)

    leg2_in_need_b = leg2_b.dropna(subset=["inpatient_need_but_no_raw", "inpatient_need_but_no_lag"]).copy()
    ols_in_need_b = smf.ols(
        "adl5_lead ~ log_burden_ratio_raw_lag + inpatient_need_but_no_raw + inpatient_need_but_no_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2_in_need_b,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2_in_need_b["ID"]})
    outs.append(
        tidy_fit(
            ols_in_need_b,
            "leg2_burden",
            "adl_lead_with_in_need_no_cluster_ols_burden",
            "adl5_lead",
            int(ols_in_need_b.nobs),
            int(leg2_in_need_b["ID"].nunique()),
        )
    )
    add_att_matched_burden("in_need_no", leg2_in_need_b, ols_in_need_b)

    leg2_in_cost_b = leg2_b.dropna(subset=["inpatient_forgone_cost_raw", "inpatient_forgone_cost_lag"]).copy()
    ols_in_cost_b = smf.ols(
        "adl5_lead ~ log_burden_ratio_raw_lag + inpatient_forgone_cost_raw + inpatient_forgone_cost_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2_in_cost_b,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2_in_cost_b["ID"]})
    outs.append(
        tidy_fit(
            ols_in_cost_b,
            "leg2_burden",
            "adl_lead_with_in_forg_cost_cluster_ols_burden",
            "adl5_lead",
            int(ols_in_cost_b.nobs),
            int(leg2_in_cost_b["ID"].nunique()),
        )
    )
    add_att_matched_burden("in_forg_cost", leg2_in_cost_b, ols_in_cost_b)

    outs_df = pd.concat(outs, ignore_index=True)

    pd.DataFrame(attenuation_rows).to_csv(ATTENUATION_PATH, sep="\t", index=False)
    pd.DataFrame(att_burden_rows).to_csv(ATTENUATION_BURDEN_PATH, sep="\t", index=False)
    outs_df.to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Wrote {SAMPLE_PATH}")
    print(f"Wrote {ATTENUATION_PATH}")
    print(f"Wrote {ATTENUATION_BURDEN_PATH}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
