#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HARMONIZED = ROOT / "data" / "raw" / "charls" / "extracted" / "harmonized" / "H_CHARLS_D_Data.dta"
WIDE_PATH = ROOT / "data" / "derived" / "charls_main_cohort_wide.tsv.gz"
OUTPUT_PATH = ROOT / "data" / "derived" / "charls_main_cohort_long.tsv.gz"

WAVE_TO_YEAR = {1: 2011, 2: 2013, 3: 2015, 4: 2018}


def ensure_dirs() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def core_disease_count(frame: pd.DataFrame) -> pd.Series:
    disease_cols = ["hibpe", "diabe", "dyslipe", "hearte", "stroke", "lunge", "arthre"]
    disease = frame[disease_cols]
    out = disease.eq(1).sum(axis=1).astype(float)
    out.loc[disease.isna().any(axis=1)] = np.nan
    return out


def main() -> None:
    ensure_dirs()

    # Read cohort IDs from the wide build (enforces cohort definition).
    wide = pd.read_csv(WIDE_PATH, sep="\t", usecols=["ID", "baseline_age", "baseline_age_wave"])
    wide["ID"] = pd.to_numeric(wide["ID"], errors="coerce").astype("Int64")
    wide = wide.dropna(subset=["ID"]).copy()
    cohort_ids = wide["ID"].unique()

    cols = [
        "ID",
        # Baseline covariates (constant across waves)
        "ragender",
        "raeduc_c",
        "raeducl",
        "h1rural",
        "r1higov",
        "hh1atotb",
        "r1doctor1m",
        # Wave-specific age and outcomes
        "r1agey", "r2agey", "r3agey", "r4agey",
        "r1adlfive", "r2adlfive", "r3adlfive", "r4adlfive",
        "r1shlta", "r2shlta", "r3shlta", "r4shlta",
        # Wave-specific chronic disease indicators
        "r1hibpe", "r2hibpe", "r3hibpe", "r4hibpe",
        "r1diabe", "r2diabe", "r3diabe", "r4diabe",
        "r1dyslipe", "r2dyslipe", "r3dyslipe", "r4dyslipe",
        "r1hearte", "r2hearte", "r3hearte", "r4hearte",
        "r1stroke", "r2stroke", "r3stroke", "r4stroke",
        "r1lunge", "r2lunge", "r3lunge", "r4lunge",
        "r1arthre", "r2arthre", "r3arthre", "r4arthre",
    ]
    harm = pd.read_stata(HARMONIZED, columns=cols, convert_categoricals=False, preserve_dtypes=False)
    harm["ID"] = pd.to_numeric(harm["ID"], errors="coerce").astype("Int64")
    harm = harm[harm["ID"].isin(cohort_ids)].copy()

    # Baseline fields (match internal derived conventions)
    harm = harm.rename(
        columns={
            "ragender": "sex",
            "raeduc_c": "education_c",
            "raeducl": "education_level",
            "hh1atotb": "ses_wealth_w1",
            "r1doctor1m": "doctor_visit_last_month_w1",
        }
    )
    harm["baseline_age"] = harm["r1agey"]
    harm["baseline_age_wave"] = 1
    harm["residence_rural_w1"] = harm["h1rural"]
    harm["insurance_public_w1"] = harm["r1higov"]

    base_cols = [
        "ID",
        "baseline_age",
        "baseline_age_wave",
        "sex",
        "education_c",
        "education_level",
        "residence_rural_w1",
        "insurance_public_w1",
        "ses_wealth_w1",
        "doctor_visit_last_month_w1",
    ]

    frames: list[pd.DataFrame] = []
    for wave in [1, 2, 3, 4]:
        frame = harm[base_cols].copy()
        frame["wave"] = wave
        frame["year"] = WAVE_TO_YEAR[wave]

        frame["age_wave"] = harm[f"r{wave}agey"]
        frame["adl5"] = harm[f"r{wave}adlfive"]
        frame["shlta"] = harm[f"r{wave}shlta"]

        for var in ["hibpe", "diabe", "dyslipe", "hearte", "stroke", "lunge", "arthre"]:
            frame[var] = harm[f"r{wave}{var}"]

        frame["core_disease_count"] = core_disease_count(frame)
        frame["core_disease_scaled"] = frame["core_disease_count"] / 7.0
        frame["adl5_scaled"] = frame["adl5"] / 5.0
        frame["shlta_scaled"] = (frame["shlta"] - 1.0) / 4.0

        # Legacy columns (kept for backward compatibility; not used by FT analyses)
        frame["tcm_visit_any"] = np.nan
        frame["tcm_visit_count"] = np.nan
        frame["primary_burden"] = np.nan
        frame["usable_primary"] = np.nan

        frames.append(frame)

    long = pd.concat(frames, ignore_index=True)
    # Column order matches the internal derived long file (subset + legacy placeholders).
    col_order = [
        "ID",
        "baseline_age",
        "baseline_age_wave",
        "sex",
        "education_c",
        "education_level",
        "residence_rural_w1",
        "insurance_public_w1",
        "ses_wealth_w1",
        "doctor_visit_last_month_w1",
        "age_wave",
        "tcm_visit_any",
        "tcm_visit_count",
        "hibpe",
        "diabe",
        "dyslipe",
        "hearte",
        "stroke",
        "lunge",
        "arthre",
        "adl5",
        "shlta",
        "core_disease_count",
        "core_disease_scaled",
        "adl5_scaled",
        "shlta_scaled",
        "primary_burden",
        "usable_primary",
        "wave",
        "year",
    ]
    long = long[col_order]

    long.to_csv(OUTPUT_PATH, sep="\t", index=False, compression="gzip")
    print(f"Wrote {OUTPUT_PATH} (N_ids={long['ID'].nunique():,}; N_rows={len(long):,})")


if __name__ == "__main__":
    main()

