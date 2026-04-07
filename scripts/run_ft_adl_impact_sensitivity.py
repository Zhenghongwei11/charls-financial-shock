#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
QC_DIR = ROOT / "results" / "qc"
EFFECT_DIR = ROOT / "results" / "effect_sizes"

BALANCED_LONG = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
BALANCED_PROFILES = ROOT / "results" / "trajectory" / "ft_trajectory_profiles.tsv"
MAIN_LONG = ROOT / "data" / "derived" / "charls_main_cohort_long_extended.tsv.gz"
MAIN_PROFILES = ROOT / "results" / "trajectory" / "ft_trajectory_profiles_maincohort.tsv"

OUT_PATH = EFFECT_DIR / "ft_adl_impact_sensitivity.tsv"
SAMPLE_PATH = QC_DIR / "ft_adl_impact_sensitivity_sample.tsv"

WAVE_TO_YEARS = {1: 0, 2: 2, 3: 4, 4: 7}


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    long_path: Path
    profile_path: Path
    profile_col: str
    restrict_adl_complete_4wave: bool


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    EFFECT_DIR.mkdir(parents=True, exist_ok=True)


def add_baseline_covariates(df: pd.DataFrame) -> pd.DataFrame:
    # baseline disease count + baseline ADL from wave 1
    base = df[df["wave"] == 1][["ID", "core_disease_count", "adl5"]].rename(
        columns={"core_disease_count": "core_disease_count_w1", "adl5": "adl5_w1"}
    )
    out = df.merge(base, on="ID", how="left")
    return out


def restrict_to_adl_complete(df: pd.DataFrame) -> pd.DataFrame:
    ids = df.dropna(subset=["adl5"]).groupby("ID")["wave"].nunique()
    keep = set(ids[ids == 4].index)
    return df[df["ID"].isin(keep)].copy()


def tidy_fit(fit, dataset_id: str, model_id: str, profile_col: str, covar_set: str, n_obs: int, n_ids: int) -> list[dict[str, object]]:
    conf = fit.conf_int()
    rows = []
    for term in fit.params.index:
        if term not in conf.index:
            continue
        rows.append(
            {
                "dataset_id": dataset_id,
                "model_id": model_id,
                "profile_col": profile_col,
                "covar_set": covar_set,
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
    return rows


def fit_models(df: pd.DataFrame, dataset_id: str, profile_col: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    df = df.copy()
    df["t_years"] = df["wave"].map(WAVE_TO_YEARS).astype(float)

    # Ensure profile is a category with stable reference
    df[profile_col] = df[profile_col].astype("category")

    base_required = ["adl5", "t_years", profile_col, "baseline_age", "sex", "education_c", "residence_rural_w1"]
    df = df.dropna(subset=base_required).copy()

    # covariate sets
    covar_sets = {
        "baseline_disease": ["core_disease_count_w1"],
        "baseline_disease_adl": ["core_disease_count_w1", "adl5_w1"],
    }

    for covar_set, extra in covar_sets.items():
        work = df.dropna(subset=extra).copy()
        n_obs = int(len(work))
        n_ids = int(work["ID"].nunique())
        if n_obs == 0:
            continue

        rhs = "t_years + C(%s) + t_years:C(%s) + baseline_age + sex + education_c + residence_rural_w1" % (
            profile_col,
            profile_col,
        )
        if extra:
            rhs += " + " + " + ".join(extra)

        # MixedLM random intercept only (more stable); if fails, cluster-OLS fallback
        try:
            model = smf.mixedlm(f"adl5 ~ {rhs}", data=work, groups=work["ID"])
            fit = model.fit(reml=False, method="lbfgs", maxiter=400, disp=False)
            rows.extend(tidy_fit(fit, dataset_id, "mixedlm_re_int", profile_col, covar_set, n_obs, n_ids))
        except Exception:
            fit = smf.ols(f"adl5 ~ {rhs}", data=work).fit(cov_type="cluster", cov_kwds={"groups": work["ID"]})
            rows.extend(tidy_fit(fit, dataset_id, "cluster_ols_fallback", profile_col, covar_set, n_obs, n_ids))

        # Binomial GLM sensitivity (adl5 as successes out of 5)
        # Binomial GEE sensitivity (adl5 as proportion with n_trials=5)
        work = work.copy()
        work["adl5_prop"] = work["adl5"] / 5.0
        gee = smf.gee(
            f"adl5_prop ~ {rhs}",
            groups="ID",
            data=work,
            family=sm.families.Binomial(),
            weights=np.full(len(work), 5.0),
        ).fit()
        rows.extend(tidy_fit(gee, dataset_id, "gee_binomial", profile_col, covar_set, n_obs, n_ids))

    return rows


def main() -> None:
    ensure_dirs()

    specs = [
        DatasetSpec(
            dataset_id="balanced_ft",
            long_path=BALANCED_LONG,
            profile_path=BALANCED_PROFILES,
            profile_col="ft_profile",
            restrict_adl_complete_4wave=True,
        ),
        DatasetSpec(
            dataset_id="main_cohort",
            long_path=MAIN_LONG,
            profile_path=MAIN_PROFILES,
            profile_col="ft_profile",
            restrict_adl_complete_4wave=False,
        ),
        DatasetSpec(
            dataset_id="main_cohort",
            long_path=MAIN_LONG,
            profile_path=MAIN_PROFILES,
            profile_col="ft_profile_alt_p90_only",
            restrict_adl_complete_4wave=False,
        ),
        DatasetSpec(
            dataset_id="main_cohort",
            long_path=MAIN_LONG,
            profile_path=MAIN_PROFILES,
            profile_col="ft_profile_alt_p95_any",
            restrict_adl_complete_4wave=False,
        ),
    ]

    all_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for spec in specs:
        long_df = pd.read_csv(spec.long_path, sep="\t")
        prof = pd.read_csv(spec.profile_path, sep="\t", usecols=["ID", spec.profile_col])

        df = long_df.merge(prof, on="ID", how="left")
        df = add_baseline_covariates(df)
        if spec.restrict_adl_complete_4wave:
            df = restrict_to_adl_complete(df)

        # sample summary
        tmp = df.dropna(subset=["adl5", spec.profile_col]).copy()
        tmp["t_years"] = tmp["wave"].map(WAVE_TO_YEARS).astype(float)
        sample = (
            tmp.groupby([spec.profile_col, "wave"], as_index=False)
            .agg(n_rows=("ID", "size"), n_ids=("ID", "nunique"), adl5_mean=("adl5", "mean"))
            .rename(columns={spec.profile_col: "profile_level"})
        )
        sample["dataset_id"] = spec.dataset_id
        sample["profile_col"] = spec.profile_col
        sample_rows.append(sample)

        all_rows.extend(fit_models(df, spec.dataset_id, spec.profile_col))

    pd.concat(sample_rows, ignore_index=True).to_csv(SAMPLE_PATH, sep="\t", index=False)
    pd.DataFrame(all_rows).to_csv(OUT_PATH, sep="\t", index=False)
    print(f"Wrote {SAMPLE_PATH}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
