#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
FT_LONG_PATH = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
PROFILE_PATH = ROOT / "results" / "trajectory" / "ft_trajectory_profiles.tsv"

QC_DIR = ROOT / "results" / "qc"
EFFECT_DIR = ROOT / "results" / "effect_sizes"

SAMPLE_PATH = QC_DIR / "ft_mixedlm_sample.tsv"
MODEL_PATH = EFFECT_DIR / "ft_mixedlm_adl.tsv"

WAVE_TO_YEARS = {1: 0, 2: 2, 3: 4, 4: 7}


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    EFFECT_DIR.mkdir(parents=True, exist_ok=True)


def fit_mixedlm(df: pd.DataFrame):
    # Random intercept + random slope on time (may fail; we fall back later)
    model = smf.mixedlm(
        "adl5 ~ t_years + C(ft_profile) + t_years:C(ft_profile) + baseline_age + sex + education_c + residence_rural_w1 + core_disease_count",
        data=df,
        groups=df["ID"],
        re_formula="~t_years",
    )
    return model.fit(reml=False, method="lbfgs", maxiter=400, disp=False)


def fit_fallback_cluster_ols(df: pd.DataFrame):
    fit = smf.ols(
        "adl5 ~ t_years + C(ft_profile) + t_years:C(ft_profile) + baseline_age + sex + education_c + residence_rural_w1 + core_disease_count",
        data=df,
    ).fit(cov_type="cluster", cov_kwds={"groups": df["ID"]})
    return fit


def tidy_fit(fit, model_id: str, n_obs: int, n_ids: int) -> pd.DataFrame:
    conf = fit.conf_int()
    rows = []
    for term in fit.params.index:
        if term not in conf.index:
            continue
        rows.append(
            {
                "model_id": model_id,
                "term": term,
                "estimate": float(fit.params[term]),
                "std_error": float(fit.bse[term]) if term in fit.bse.index else None,
                "p_value": float(fit.pvalues[term]) if term in fit.pvalues.index else None,
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
                "n_obs": int(n_obs),
                "n_ids": int(n_ids),
                "log_likelihood": float(getattr(fit, "llf", np.nan)),
                "aic": float(getattr(fit, "aic", np.nan)),
                "bic": float(getattr(fit, "bic", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    long_df = pd.read_csv(FT_LONG_PATH, sep="\t")
    profiles = pd.read_csv(PROFILE_PATH, sep="\t", usecols=["ID", "ft_profile"])

    df = long_df.merge(profiles, on="ID", how="left")
    df["t_years"] = df["wave"].map(WAVE_TO_YEARS).astype(float)

    # Keep complete covariates for the first-pass impact model
    required = [
        "adl5",
        "t_years",
        "ft_profile",
        "baseline_age",
        "sex",
        "education_c",
        "residence_rural_w1",
        "core_disease_count",
    ]
    analysis = df.dropna(subset=required).copy()

    sample = (
        analysis.groupby(["ft_profile", "wave"], as_index=False)
        .agg(n_rows=("ID", "size"), n_ids=("ID", "nunique"), adl5_mean=("adl5", "mean"))
    )
    sample.to_csv(SAMPLE_PATH, sep="\t", index=False)

    n_obs = int(len(analysis))
    n_ids = int(analysis["ID"].nunique())

    try:
        fit = fit_mixedlm(analysis)
        out = tidy_fit(fit, "mixedlm_re_int_slope", n_obs, n_ids)
    except Exception:
        fit = fit_fallback_cluster_ols(analysis)
        out = tidy_fit(fit, "cluster_ols_fallback", n_obs, n_ids)

    out.to_csv(MODEL_PATH, sep="\t", index=False)
    print(f"Wrote {SAMPLE_PATH}")
    print(f"Wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()

