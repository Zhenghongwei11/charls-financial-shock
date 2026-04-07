#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
FT_LONG_PATH = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
THRESH_PATH = ROOT / "results" / "qc" / "ft_shock_thresholds.tsv"

QC_DIR = ROOT / "results" / "qc"
EFFECT_DIR = ROOT / "results" / "effect_sizes"

SAMPLE_PATH = QC_DIR / "ft_cesd_chain_sample.tsv"
ATTENUATION_PATH = QC_DIR / "ft_cesd_attenuation.tsv"
OUT_PATH = EFFECT_DIR / "ft_cesd_chain.tsv"

WAVE_TO_YEARS = {1: 0, 2: 2, 3: 4, 4: 7}


def ensure_dirs() -> None:
    QC_DIR.mkdir(parents=True, exist_ok=True)
    EFFECT_DIR.mkdir(parents=True, exist_ok=True)


def build_panel(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["t_years"] = df["wave"].map(WAVE_TO_YEARS).astype(float)
    df = df.sort_values(["ID", "t_years"]).reset_index(drop=True)

    g = df.groupby("ID", sort=False)
    df["wave_lag"] = g["wave"].shift(1)
    df["interval_lag"] = df["wave_lag"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})
    df["adl5_lag"] = g["adl5"].shift(1)
    df["core_disease_count_lag"] = g["core_disease_count"].shift(1)
    df["oop_raw_lag"] = g["total_annual_oop_raw"].shift(1)
    df["cesd10_lag"] = g["cesd10"].shift(1)

    df["wave_lead"] = g["wave"].shift(-1)
    df["interval_lead"] = df["wave"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})
    df["adl5_lead"] = g["adl5"].shift(-1)
    df["t_years_lead"] = g["t_years"].shift(-1)
    df["dt_years_lead"] = df["t_years_lead"] - df["t_years"]
    df["delta_adl_lead"] = df["adl5_lead"] - df["adl5"]
    df["delta_adl_per_year_lead"] = df["delta_adl_lead"] / df["dt_years_lead"]

    df["log_oop_raw_lag"] = np.log1p(df["oop_raw_lag"])
    return df


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


def tidy(fit, module: str, model_id: str, outcome: str, n_obs: int, n_ids: int) -> pd.DataFrame:
    conf = fit.conf_int()
    rows: list[dict[str, object]] = []
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
                "std_error": float(fit.bse.get(term, np.nan)),
                "p_value": float(fit.pvalues.get(term, np.nan)),
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

    panel = add_shock_p95(build_panel(long_df), shock_df)

    outs: list[pd.DataFrame] = []
    sample_rows: list[dict[str, object]] = []
    att_rows: list[dict[str, object]] = []

    # -------------------------
    # Leg 1: shock_{t-1} -> CESD(t)
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
        "cesd10",
    ]

    leg1 = panel.dropna(subset=leg1_common).copy()
    leg1 = leg1[leg1["interval_lag"].notna()].copy()
    sample_rows.append({"module": "leg1_cesd_level", "n_obs": int(len(leg1)), "n_ids": int(leg1["ID"].nunique())})

    # Version A: without CESD lag (matches common “shock -> CESD(t)” phrasing)
    fit_leg1_a = smf.ols(
        "cesd10 ~ oop_shock_p95_lag + log_oop_raw_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1["ID"]})
    outs.append(tidy(fit_leg1_a, "leg1", "cesd_leg1_no_cesdlag_cluster_ols", "cesd10", int(fit_leg1_a.nobs), int(leg1["ID"].nunique())))

    # Version B: with CESD lag (cross-lag style; more conservative)
    leg1_b = panel.dropna(subset=[*leg1_common, "cesd10_lag"]).copy()
    leg1_b = leg1_b[leg1_b["interval_lag"].notna()].copy()
    sample_rows.append({"module": "leg1_cesd_level_with_lag", "n_obs": int(len(leg1_b)), "n_ids": int(leg1_b["ID"].nunique())})
    fit_leg1_b = smf.ols(
        "cesd10 ~ oop_shock_p95_lag + log_oop_raw_lag + cesd10_lag + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag)",
        data=leg1_b,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg1_b["ID"]})
    outs.append(tidy(fit_leg1_b, "leg1", "cesd_leg1_with_cesdlag_cluster_ols", "cesd10", int(fit_leg1_b.nobs), int(leg1_b["ID"].nunique())))

    # -------------------------
    # Leg 2: CESD(t) -> ΔADL(t -> t+1) (change per year), with shock_{t-1} included
    # Only waves 2,3 can form full triples (t-1, t, t+1).
    # -------------------------
    leg2_common = [
        "delta_adl_per_year_lead",
        "cesd10",
        "oop_shock_p95_lag",
        "log_oop_raw_lag",
        "adl5",
        "adl5_lag",
        "core_disease_count_lag",
        "baseline_age",
        "sex",
        "education_c",
        "residence_rural_w1",
        "interval_lag",
        "interval_lead",
    ]
    leg2 = panel.dropna(subset=leg2_common).copy()
    leg2 = leg2[leg2["wave"].isin([2, 3])].copy()
    sample_rows.append({"module": "leg2_delta_adl_per_year_lead", "n_obs": int(len(leg2)), "n_ids": int(leg2["ID"].nunique())})

    fit_leg2 = smf.ols(
        "delta_adl_per_year_lead ~ cesd10 + oop_shock_p95_lag + log_oop_raw_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2["ID"]})
    outs.append(tidy(fit_leg2, "leg2", "cesd_leg2_delta_per_year_lead_cluster_ols", "delta_adl_per_year_lead", int(fit_leg2.nobs), int(leg2["ID"].nunique())))

    # Attenuation check (matched on CESD availability)
    fit_base = smf.ols(
        "delta_adl_per_year_lead ~ oop_shock_p95_lag + log_oop_raw_lag + adl5 + adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval_lag) + C(interval_lead)",
        data=leg2,
    ).fit(cov_type="cluster", cov_kwds={"groups": leg2["ID"]})

    att_rows.append(
        {
            "base_scope": "matched",
            "mediator": "cesd10",
            "shock_est_no_mediator": float(fit_base.params.get("oop_shock_p95_lag", np.nan)),
            "shock_est_with_mediator": float(fit_leg2.params.get("oop_shock_p95_lag", np.nan)),
            "delta": float(fit_leg2.params.get("oop_shock_p95_lag", np.nan) - fit_base.params.get("oop_shock_p95_lag", np.nan)),
            "p_no_mediator": float(fit_base.pvalues.get("oop_shock_p95_lag", np.nan)),
            "p_with_mediator": float(fit_leg2.pvalues.get("oop_shock_p95_lag", np.nan)),
            "n_obs": int(fit_leg2.nobs),
            "n_ids": int(leg2["ID"].nunique()),
        }
    )

    # Write outputs
    pd.DataFrame(sample_rows).to_csv(SAMPLE_PATH, sep="\t", index=False)
    pd.DataFrame(att_rows).to_csv(ATTENUATION_PATH, sep="\t", index=False)
    pd.concat(outs, ignore_index=True).to_csv(OUT_PATH, sep="\t", index=False)

    print(f"Wrote {OUT_PATH}")
    print(f"Wrote {SAMPLE_PATH}")
    print(f"Wrote {ATTENUATION_PATH}")


if __name__ == "__main__":
    main()

