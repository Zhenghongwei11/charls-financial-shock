#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXT_LONG_PATH = ROOT / "data" / "derived" / "charls_main_cohort_long_extended.tsv.gz"
OUTPUT_PATH = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
QC_DIR = ROOT / "results" / "qc"
MISSINGNESS_PATH = QC_DIR / "ft_spending_missingness.tsv"
COMBO_AUDIT_PATH = QC_DIR / "ft_impossible_combinations.tsv"

# CPI indices for 2011-2018 relative to 2011 (Source: NBS China)
# 2011 = 100.0
# 2013 = 105.3
# 2015 = 108.6
# 2018 = 116.1
CPI_ADJUSTMENT = {
    1: 1.0,
    2: 1.053,
    3: 1.086,
    4: 1.161
}

SPEND_COLS = [
    "outpatient_total_wave",
    "outpatient_oop_wave",
    "hospital_total_wave",
    "hospital_oop_wave",
]


def ensure_dirs() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)


def winsorize_by_wave(df: pd.DataFrame, column: str, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    out = df[column].copy()
    for wave, factor in CPI_ADJUSTMENT.items():
        mask = df["wave"] == wave
        values = df.loc[mask, column].dropna()
        if values.empty:
            continue
        low = float(values.quantile(lower_q))
        high = float(values.quantile(upper_q))
        out.loc[mask] = df.loc[mask, column].clip(lower=low, upper=high)
    return out


def build_missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for wave in sorted(df["wave"].dropna().unique()):
        wave_df = df[df["wave"] == wave]
        for col in [*SPEND_COLS, "total_annual_oop", "total_annual_oop_raw", "ft_burden_ratio"]:
            if col not in wave_df.columns:
                continue
            series = wave_df[col]
            rows.append(
                {
                    "wave": int(wave),
                    "year": int(wave_df["year"].iloc[0]) if "year" in wave_df.columns else None,
                    "variable": col,
                    "n_rows": int(len(wave_df)),
                    "n_missing": int(series.isna().sum()),
                    "pct_missing": float(series.isna().mean()),
                    "n_zero": int((series == 0).sum()) if series.notna().any() else 0,
                    "pct_zero": float((series == 0).mean()) if series.notna().any() else 0.0,
                }
            )
    return pd.DataFrame(rows)


def build_combo_audit_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for wave in sorted(df["wave"].dropna().unique()):
        wave_df = df[df["wave"] == wave]

        def count(mask: pd.Series) -> int:
            return int(mask.sum())

        outpatient_all_zero = (
            (wave_df["doctor_visit_any_wave"] == 1)
            & (wave_df["outpatient_total_wave"] == 0)
            & (wave_df["outpatient_oop_wave"] == 0)
        )
        hospital_all_zero = (
            (wave_df["hospital_stay_any_wave"] == 1)
            & (wave_df["hospital_total_wave"] == 0)
            & (wave_df["hospital_oop_wave"] == 0)
        )

        rows.append(
            {
                "wave": int(wave),
                "year": int(wave_df["year"].iloc[0]) if "year" in wave_df.columns else None,
                "n_rows": int(len(wave_df)),
                "doctor_visit_any_wave_nonmissing_pct": float(wave_df["doctor_visit_any_wave"].notna().mean()),
                "hospital_stay_any_wave_nonmissing_pct": float(wave_df["hospital_stay_any_wave"].notna().mean()),
                "visit_eq_1_and_outpatient_total_eq_0_and_oop_eq_0_n": count(outpatient_all_zero),
                "stay_eq_1_and_hospital_total_eq_0_and_oop_eq_0_n": count(hospital_all_zero),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_dirs()
    print(f"Reading {EXT_LONG_PATH}...")
    df = pd.read_csv(EXT_LONG_PATH, sep="\t")
    
    # 1. Inflation Adjustment (Real 2011 Prices)
    print("Applying CPI adjustment (Real 2011 Prices)...")
    for wave, factor in CPI_ADJUSTMENT.items():
        mask = df["wave"] == wave
        for col in SPEND_COLS:
            df.loc[mask, col] = df.loc[mask, col] / factor

    # 2. Structural zeros (do NOT impute unknown spending to 0)
    # - If utilization indicator says no visit/stay, spending is structurally 0.
    # - Otherwise, preserve missing spending as NA.
    print("Applying structural-zero rules using utilization indicators...")
    outpatient_zero_mask = (df["doctor_visit_any_wave"] == 0) & df["outpatient_total_wave"].isna()
    df.loc[outpatient_zero_mask, "outpatient_total_wave"] = 0.0
    outpatient_zero_mask = (df["doctor_visit_any_wave"] == 0) & df["outpatient_oop_wave"].isna()
    df.loc[outpatient_zero_mask, "outpatient_oop_wave"] = 0.0

    hospital_zero_mask = (df["hospital_stay_any_wave"] == 0) & df["hospital_total_wave"].isna()
    df.loc[hospital_zero_mask, "hospital_total_wave"] = 0.0
    hospital_zero_mask = (df["hospital_stay_any_wave"] == 0) & df["hospital_oop_wave"].isna()
    df.loc[hospital_zero_mask, "hospital_oop_wave"] = 0.0

    # 3. Preserve raw spending for "shock" profiling and winsorize for stable mean-modeling
    print("Tracking raw spending and performing per-wave 1%/99% winsorization (no NA→0)...")
    for col in SPEND_COLS:
        df[f"{col}_raw"] = df[col]
        df[col] = winsorize_by_wave(df, col, lower_q=0.01, upper_q=0.99)
        df[f"{col}_winsor"] = df[col]
    
    # 4. Calculate Derived Toxicity Metrics
    # Total OOP = Outpatient OOP (1m) * 12 + Hospital OOP (1y)
    # Note: outpatient is 1 month recall, hospital is 1 year recall.
    # We annualize outpatient to match hospital recall.
    df["total_annual_oop"] = (df["outpatient_oop_wave"] * 12) + df["hospital_oop_wave"]
    df["total_annual_oop_raw"] = (df["outpatient_oop_wave_raw"] * 12) + df["hospital_oop_wave_raw"]
    
    # Financial Burden Ratio (secondary) = Annual OOP / Baseline Wealth, only when wealth > 0.
    wealth = df["ses_wealth_w1"]
    df["wealth_nonpositive_flag"] = np.where(wealth.notna(), (wealth <= 0).astype(float), np.nan)
    df["ft_burden_ratio"] = np.where(wealth > 0, df["total_annual_oop"] / wealth, np.nan)
    df["ft_burden_ratio_raw"] = np.where(wealth > 0, df["total_annual_oop_raw"] / wealth, np.nan)
    
    # 5. Filter for a balanced-by-ADL cohort (4 waves) and keep spending missingness explicit
    sub = df[["ID", "wave", "adl5"]].dropna(subset=["adl5"])
    ids_4wave = sub.groupby("ID")["wave"].nunique()
    balanced_ids = ids_4wave[ids_4wave == 4].index
    balanced = df[df["ID"].isin(balanced_ids)].copy()
    
    print(f"Processed Financial Toxicity data for {len(balanced_ids)} balanced individuals.")

    # 6. Write FT-specific QC tables
    missingness = build_missingness_table(balanced)
    missingness.to_csv(MISSINGNESS_PATH, sep="\t", index=False)
    combo_audit = build_combo_audit_table(balanced)
    combo_audit.to_csv(COMBO_AUDIT_PATH, sep="\t", index=False)
    
    # Save the processed data
    balanced.to_csv(OUTPUT_PATH, sep="\t", index=False, compression="gzip")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote {MISSINGNESS_PATH}")
    print(f"Wrote {COMBO_AUDIT_PATH}")


if __name__ == "__main__":
    main()
