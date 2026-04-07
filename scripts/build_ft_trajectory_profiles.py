#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
OUT_DIR = ROOT / "results" / "trajectory"

THRESHOLDS_PATH = OUT_DIR / "ft_thresholds_by_wave.tsv"
PROFILE_PATH = OUT_DIR / "ft_trajectory_profiles.tsv"
SUMMARY_PATH = OUT_DIR / "ft_trajectory_summary.tsv"

WAVES = [1, 2, 3, 4]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def wave_thresholds(long_df: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for wave in WAVES:
        subset = long_df[long_df["wave"] == wave]
        values = subset["total_annual_oop_raw"].dropna()
        for q in quantiles:
            rows.append(
                {
                    "wave": wave,
                    "year": int(subset["year"].iloc[0]) if "year" in subset.columns and not subset.empty else None,
                    "variable": "total_annual_oop_raw",
                    "quantile": float(q),
                    "threshold": float(values.quantile(q)) if not values.empty else np.nan,
                    "n_nonmissing": int(values.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def build_profiles(long_df: pd.DataFrame, thr_df: pd.DataFrame) -> pd.DataFrame:
    # Map wave->threshold for q90/q95
    q90 = thr_df[thr_df["quantile"] == 0.90].set_index("wave")["threshold"].to_dict()
    q95 = thr_df[thr_df["quantile"] == 0.95].set_index("wave")["threshold"].to_dict()

    df = long_df[["ID", "wave", "total_annual_oop_raw"]].copy()
    df["oop_raw"] = df["total_annual_oop_raw"]
    df["p90_thr"] = df["wave"].map(q90)
    df["p95_thr"] = df["wave"].map(q95)
    df["above_p90"] = np.where(df["oop_raw"].notna(), (df["oop_raw"] >= df["p90_thr"]).astype(int), np.nan)
    df["above_p95"] = np.where(df["oop_raw"].notna(), (df["oop_raw"] >= df["p95_thr"]).astype(int), np.nan)
    df["nonzero"] = np.where(df["oop_raw"].notna(), (df["oop_raw"] > 0).astype(int), np.nan)

    wide_oop = df.pivot(index="ID", columns="wave", values="oop_raw")
    wide_p90 = df.pivot(index="ID", columns="wave", values="above_p90")
    wide_p95 = df.pivot(index="ID", columns="wave", values="above_p95")
    wide_nz = df.pivot(index="ID", columns="wave", values="nonzero")

    out = pd.DataFrame(index=wide_oop.index)
    for wave in WAVES:
        out[f"oop_raw_w{wave}"] = wide_oop.get(wave)
        out[f"nonzero_w{wave}"] = wide_nz.get(wave)
        out[f"above_p90_w{wave}"] = wide_p90.get(wave)
        out[f"above_p95_w{wave}"] = wide_p95.get(wave)

    oop_cols = [f"oop_raw_w{w}" for w in WAVES]
    out["oop_nonmissing_waves"] = out[oop_cols].notna().sum(axis=1)
    out["oop_nonzero_waves"] = out[[f"nonzero_w{w}" for w in WAVES]].sum(axis=1, min_count=1)
    out["oop_above_p90_waves"] = out[[f"above_p90_w{w}" for w in WAVES]].sum(axis=1, min_count=1)
    out["oop_above_p95_waves"] = out[[f"above_p95_w{w}" for w in WAVES]].sum(axis=1, min_count=1)

    out["oop_raw_mean"] = out[oop_cols].mean(axis=1, skipna=True)
    out["oop_raw_max"] = out[oop_cols].max(axis=1, skipna=True)

    # First p95 wave (spike timing)
    def first_wave_p95(row: pd.Series) -> float:
        for w in WAVES:
            val = row.get(f"above_p95_w{w}")
            if val == 1:
                return float(w)
        return np.nan

    out["first_p95_wave"] = out.apply(first_wave_p95, axis=1)

    # Rule-based trajectory profiles (baseline, interpretable)
    # - stable_low: never above p90 and at most 1 nonzero wave
    # - shock_spike: >=1 p95 wave and <=1 p90 wave in total
    # - chronic_high: >=2 p90 waves OR >=2 nonzero waves AND high mean
    # - mixed: otherwise
    out["ft_profile"] = "mixed"

    stable_low = (out["oop_above_p90_waves"] == 0) & (out["oop_nonzero_waves"] <= 1)
    shock_spike = (out["oop_above_p95_waves"] >= 1) & (out["oop_above_p90_waves"] <= 1)

    # high-mean heuristic: above pooled median of per-person means among non-missing
    mean_median = float(out["oop_raw_mean"].median(skipna=True)) if out["oop_raw_mean"].notna().any() else 0.0
    chronic_high = (out["oop_above_p90_waves"] >= 2) | ((out["oop_nonzero_waves"] >= 3) & (out["oop_raw_mean"] >= mean_median))

    out.loc[stable_low, "ft_profile"] = "stable_low"
    out.loc[shock_spike, "ft_profile"] = "shock_spike"
    out.loc[chronic_high, "ft_profile"] = "chronic_high"

    # Resolve overlaps deterministically
    out.loc[stable_low & shock_spike, "ft_profile"] = "shock_spike"
    out.loc[shock_spike & chronic_high, "ft_profile"] = "chronic_high"
    out = out.reset_index().rename(columns={"index": "ID"})
    return out


def build_summary(long_df: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    merged = long_df.merge(profiles[["ID", "ft_profile"]], on="ID", how="left")
    rows: list[dict[str, object]] = []
    for profile, subset in merged.groupby("ft_profile", dropna=False, sort=True):
        for wave in WAVES:
            wave_df = subset[subset["wave"] == wave]
            s = wave_df["total_annual_oop_raw"]
            rows.append(
                {
                    "ft_profile": profile,
                    "wave": wave,
                    "year": int(wave_df["year"].iloc[0]) if "year" in wave_df.columns and not wave_df.empty else None,
                    "n_rows": int(len(wave_df)),
                    "n_ids": int(wave_df["ID"].nunique()),
                    "oop_raw_missing_pct": float(s.isna().mean()),
                    "oop_raw_zero_pct": float((s == 0).mean()) if s.notna().any() else np.nan,
                    "oop_raw_mean": float(s.mean()) if s.notna().any() else np.nan,
                    "oop_raw_median": float(s.median()) if s.notna().any() else np.nan,
                    "oop_raw_p90": float(s.quantile(0.90)) if s.notna().any() else np.nan,
                    "oop_raw_p95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
                }
            )
    # Profile sizes
    size = (
        profiles["ft_profile"]
        .value_counts(dropna=False)
        .rename_axis("ft_profile")
        .reset_index(name="profile_n_ids")
    )
    out = pd.DataFrame(rows).merge(size, on="ft_profile", how="left")
    total = float(size["profile_n_ids"].sum()) if not size.empty else np.nan
    out["profile_pct_ids"] = out["profile_n_ids"] / total
    return out


def main() -> None:
    ensure_dirs()
    long_df = pd.read_csv(INPUT_PATH, sep="\t")

    thr_df = wave_thresholds(long_df, quantiles=[0.75, 0.90, 0.95, 0.975])
    thr_df.to_csv(THRESHOLDS_PATH, sep="\t", index=False)

    profiles = build_profiles(long_df, thr_df)
    profiles.to_csv(PROFILE_PATH, sep="\t", index=False)

    summary = build_summary(long_df, profiles)
    summary.to_csv(SUMMARY_PATH, sep="\t", index=False)

    print(f"Wrote {THRESHOLDS_PATH}")
    print(f"Wrote {PROFILE_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
