#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HARMONIZED = ROOT / "data" / "raw" / "charls" / "extracted" / "harmonized" / "H_CHARLS_D_Data.dta"
OUTPUT_PATH = ROOT / "data" / "derived" / "charls_main_cohort_wide.tsv.gz"


def ensure_dirs() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()

    cols = [
        "ID",
        # Baseline covariates / SES
        "r1agey",
        "ragender",
        "raeduc_c",
        "raeducl",
        "h1rural",
        "r1higov",
        "hh1atotb",
        # ADL availability for cohort definition + selection QC
        "r1adlfive",
        "r2adlfive",
        "r3adlfive",
        "r4adlfive",
    ]
    df = pd.read_stata(HARMONIZED, columns=cols, convert_categoricals=False, preserve_dtypes=False)
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["ID"]).copy()

    # Cohort definition (implementation-aligned):
    # - baseline age >= 60 at Wave 1
    # - ADL observed in >=3 of 4 waves (allows missing at one wave)
    adl_cols = ["r1adlfive", "r2adlfive", "r3adlfive", "r4adlfive"]
    df["baseline_age"] = df["r1agey"]
    df["baseline_age_wave"] = 1
    df["adl_nonmissing_waves"] = df[adl_cols].notna().sum(axis=1)
    cohort = df[(df["baseline_age"] >= 60) & (df["adl_nonmissing_waves"] >= 3)].copy()
    cohort = cohort.drop(columns=["adl_nonmissing_waves"])

    cohort.to_csv(OUTPUT_PATH, sep="\t", index=False, compression="gzip")
    print(f"Wrote {OUTPUT_PATH} (N={cohort['ID'].nunique():,})")


if __name__ == "__main__":
    main()

