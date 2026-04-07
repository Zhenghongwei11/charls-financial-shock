#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LONG_PATH = ROOT / "data" / "derived" / "charls_main_cohort_long.tsv.gz"
HARMONIZED = ROOT / "data" / "raw" / "charls" / "extracted" / "harmonized" / "H_CHARLS_D_Data.dta"

EXT_LONG_PATH = ROOT / "data" / "derived" / "charls_main_cohort_long_extended.tsv.gz"
SUMMARY_PATH = ROOT / "results" / "qc" / "charls_extended_long_summary.tsv"
MEMO_PATH = ROOT / "docs" / "EXTENDED_LONG_BUILD_MEMO.md"


RAW_STATUS_FILES = {
    1: ROOT / "data" / "raw" / "charls" / "extracted" / "2011" / "health_status_and_functioning.dta",
    2: ROOT / "data" / "raw" / "charls" / "extracted" / "2013" / "Health_Status_and_Functioning.dta",
    3: ROOT / "data" / "raw" / "charls" / "extracted" / "2015" / "Health_Status_and_Functioning.dta",
    4: ROOT / "data" / "raw" / "charls" / "extracted" / "2018" / "Health_Status_and_Functioning.dta",
}

RAW_CARE_FILES = {
    1: ROOT / "data" / "raw" / "charls" / "extracted" / "2011" / "health_care_and_insurance.dta",
    2: ROOT / "data" / "raw" / "charls" / "extracted" / "2013" / "Health_Care_and_Insurance.dta",
    3: ROOT / "data" / "raw" / "charls" / "extracted" / "2015" / "Health_Care_and_Insurance.dta",
    4: ROOT / "data" / "raw" / "charls" / "extracted" / "2018" / "Health_Care_and_Insurance.dta",
}


def ensure_dirs() -> None:
    EXT_LONG_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMO_PATH.parent.mkdir(parents=True, exist_ok=True)


def to_binary_12(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out.loc[series == 1] = 1.0
    out.loc[series == 2] = 0.0
    return out


def to_checkbox(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out.loc[series.notna()] = 1.0
    return out


def to_binary_diff_1234(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out.loc[series == 1] = 0.0
    out.loc[series.isin([2, 3, 4])] = 1.0
    return out


def to_binary_pain_2013(series: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=series.index, dtype="float")
    out.loc[series == 1] = 0.0
    out.loc[series.isin([2, 3, 4, 5])] = 1.0
    return out


def build_bridge() -> pd.DataFrame:
    harm = pd.read_stata(
        HARMONIZED,
        columns=["ID", "ID_w1"],
        convert_categoricals=False,
        preserve_dtypes=False,
    )
    harm["ID"] = pd.to_numeric(harm["ID"], errors="coerce").astype("Int64")
    harm["ID_str"] = harm["ID"].astype("string")
    harm["ID_w1_str"] = harm["ID_w1"].astype("string")

    frames = []
    w1 = harm[["ID", "ID_w1_str"]].copy()
    w1["wave"] = 1
    w1["raw_id_str"] = w1["ID_w1_str"]
    frames.append(w1[["ID", "wave", "raw_id_str"]])

    for wave in [2, 3, 4]:
        frame = harm[["ID", "ID_str"]].copy()
        frame["wave"] = wave
        frame["raw_id_str"] = frame["ID_str"]
        frames.append(frame[["ID", "wave", "raw_id_str"]])

    bridge = pd.concat(frames, ignore_index=True)
    bridge = bridge.dropna(subset=["ID", "raw_id_str"]).copy()
    return bridge


def build_harmonized_extensions() -> pd.DataFrame:
    cols = [
        "ID",
        "r1doctor1m", "r2doctor1m", "r3doctor1m", "r4doctor1m",
        "r1doctim1m", "r2doctim1m", "r3doctim1m", "r4doctim1m",
        "r1hosp1y", "r2hosp1y", "r3hosp1y", "r4hosp1y",
        "r1oopdoc1m", "r2oopdoc1m", "r3oopdoc1m", "r4oopdoc1m",
        "r1totdoc1m", "r2totdoc1m", "r3totdoc1m", "r4totdoc1m",
        "r1oophos1y", "r2oophos1y", "r3oophos1y", "r4oophos1y",
        "r1tothos1y", "r2tothos1y", "r3tothos1y", "r4tothos1y",
        "r2iadlza", "r3iadlza", "r4iadlza",
        "r1cesd10", "r2cesd10", "r3cesd10", "r4cesd10",
        # WM Medications
        "r1rxhibp_c", "r2rxhibp_c", "r3rxhibp_c", "r4rxhibp_c",
        "r1rxdiab_c", "r2rxdiab_c", "r3rxdiab_c", "r4rxdiab_c",
        "r1rxdyslip_c", "r2rxdyslip_c", "r3rxdyslip_c", "r4rxdyslip_c",
        "r1rxheart_c", "r2rxheart_c", "r3rxheart_c", "r4rxheart_c",
        "r1rxlung_c", "r2rxlung_c", "r3rxlung_c", "r4rxlung_c",
    ]
    harm = pd.read_stata(HARMONIZED, columns=cols, convert_categoricals=False, preserve_dtypes=False)
    harm["ID"] = pd.to_numeric(harm["ID"], errors="coerce").astype("Int64")

    frames = []
    for wave in [1, 2, 3, 4]:
        cols_wave = {
            f"r{wave}doctor1m": "doctor_visit_any_wave",
            f"r{wave}doctim1m": "doctor_visit_count_wave",
            f"r{wave}hosp1y": "hospital_stay_any_wave",
            f"r{wave}oopdoc1m": "outpatient_oop_wave",
            f"r{wave}totdoc1m": "outpatient_total_wave",
            f"r{wave}oophos1y": "hospital_oop_wave",
            f"r{wave}tothos1y": "hospital_total_wave",
            f"r{wave}cesd10": "cesd10",
            # WM Meds mapping
            f"r{wave}rxhibp_c": "rx_hibp",
            f"r{wave}rxdiab_c": "rx_diab",
            f"r{wave}rxdyslip_c": "rx_dyslip",
            f"r{wave}rxheart_c": "rx_heart",
            f"r{wave}rxlung_c": "rx_lung",
        }
        if wave >= 2:
            cols_wave[f"r{wave}iadlza"] = "iadl5_wave"
        frame = harm[["ID", *cols_wave.keys()]].copy()
        frame["wave"] = wave
        frame = frame.rename(columns=cols_wave)
        
        # Derived Med Summary (Binary: taking any core WM meds)
        med_cols = ["rx_hibp", "rx_diab", "rx_dyslip", "rx_heart", "rx_lung"]
        frame["wm_med_any"] = (frame[med_cols] == 1).any(axis=1).astype(float)
        frame.loc[frame[med_cols].isna().all(axis=1), "wm_med_any"] = np.nan
        
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def build_raw_status_extensions(bridge: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for wave, path in RAW_STATUS_FILES.items():
        if wave == 1:
            raw = pd.read_stata(
                path,
                columns=["ID", "da023", "da024", "da041", "db002", "db003"],
                convert_categoricals=False,
                preserve_dtypes=False,
            )
            raw["fall_any_raw"] = to_binary_12(raw["da023"])
            raw["fall_count_raw"] = raw["da024"]
            raw["pain_any_raw"] = to_binary_12(raw["da041"])
        elif wave == 2:
            raw = pd.read_stata(
                path,
                columns=["ID", "da023", "da024", "wb16", "db002", "db003"],
                convert_categoricals=False,
                preserve_dtypes=False,
            )
            raw["fall_any_raw"] = to_binary_12(raw["da023"])
            raw["fall_count_raw"] = raw["da024"]
            raw["pain_any_raw"] = to_binary_pain_2013(raw["wb16"])
        elif wave == 3:
            raw = pd.read_stata(
                path,
                columns=["ID", "da023", "da024", "da041", "db002", "db003"],
                convert_categoricals=False,
                preserve_dtypes=False,
            )
            raw["fall_any_raw"] = to_binary_12(raw["da023"])
            raw["fall_count_raw"] = raw["da024"]
            raw["pain_any_raw"] = to_binary_12(raw["da041"])
        else:
            raw = pd.read_stata(
                path,
                columns=["ID", "da023_w4", "da024", "da041_w4", "db002", "db003"],
                convert_categoricals=False,
                preserve_dtypes=False,
            )
            raw["fall_any_raw"] = to_binary_12(raw["da023_w4"])
            raw["fall_count_raw"] = raw["da024"]
            raw["pain_any_raw"] = to_binary_pain_2013(raw["da041_w4"])

        raw["walk_1km_diff_raw"] = to_binary_diff_1234(raw["db002"])
        raw["walk_100m_diff_raw"] = to_binary_diff_1234(raw["db003"])
        raw["mobility_limit_any_raw"] = raw[["walk_1km_diff_raw", "walk_100m_diff_raw"]].max(axis=1, skipna=True)
        raw.loc[raw[["walk_1km_diff_raw", "walk_100m_diff_raw"]].isna().all(axis=1), "mobility_limit_any_raw"] = np.nan
        raw.loc[(raw["fall_any_raw"] == 0) & (raw["fall_count_raw"].isna()), "fall_count_raw"] = 0.0

        raw["raw_id_str"] = raw["ID"].astype("string")
        raw = raw[[
            "raw_id_str",
            "fall_any_raw",
            "fall_count_raw",
            "pain_any_raw",
            "walk_1km_diff_raw",
            "walk_100m_diff_raw",
            "mobility_limit_any_raw",
        ]]
        raw["wave"] = wave

        frame = bridge[bridge["wave"] == wave].merge(raw, on=["wave", "raw_id_str"], how="left")
        frames.append(frame.drop(columns=["raw_id_str"]))

    return pd.concat(frames, ignore_index=True)


def build_raw_care_extensions(bridge: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for wave, path in RAW_CARE_FILES.items():
        if wave in [1, 2, 3]:
            raw = pd.read_stata(
                path,
                columns=["ID", "ed021s7", "ee021s8", "ef001s3", "ed003", "ee001", "ee002"],
                convert_categoricals=False,
                preserve_dtypes=False,
            )
            raw["traditional_outpatient_raw"] = to_checkbox(raw["ed021s7"])
            raw["traditional_inpatient_raw"] = to_checkbox(raw["ee021s8"])
            raw["traditional_selfmed_raw"] = to_checkbox(raw["ef001s3"])

            # Economic forgoing (raw care modules, waves 1-3 only)
            # - ed003: reason for not seeking outpatient care (cost reason code=3; label varies by wave)
            # - ee001: needed inpatient care but did not get (1 yes / 2 no)
            # - ee002: reason for not seeking inpatient care (cost reason code=1)
            raw["outpatient_forgone_any_raw"] = to_checkbox(raw["ed003"])
            raw["outpatient_forgone_cost_raw"] = np.where(
                raw["ed003"].notna(),
                np.where(raw["ed003"] == 3, 1.0, 0.0),
                np.nan,
            )

            raw["inpatient_need_but_no_raw"] = to_binary_12(raw["ee001"])
            raw["inpatient_forgone_cost_raw"] = np.where(
                raw["ee001"].notna(),
                np.where(
                    raw["ee001"] == 1,
                    np.where(raw["ee002"].notna(), np.where(raw["ee002"] == 1, 1.0, 0.0), np.nan),
                    0.0,
                ),
                np.nan,
            )
        else:
            raw = pd.read_stata(
                path,
                columns=["ID", "ed004_w4_s3"],
                convert_categoricals=False,
                preserve_dtypes=False,
            )
            raw["traditional_outpatient_raw"] = np.where(raw["ed004_w4_s3"] == 3, 1.0, np.where(raw["ed004_w4_s3"] == 0, 0.0, np.nan))
            raw["traditional_inpatient_raw"] = np.nan
            raw["traditional_selfmed_raw"] = np.nan
            raw["outpatient_forgone_any_raw"] = np.nan
            raw["outpatient_forgone_cost_raw"] = np.nan
            raw["inpatient_need_but_no_raw"] = np.nan
            raw["inpatient_forgone_cost_raw"] = np.nan

        raw["raw_id_str"] = raw["ID"].astype("string")
        raw = raw[[
            "raw_id_str",
            "traditional_outpatient_raw",
            "traditional_inpatient_raw",
            "traditional_selfmed_raw",
            "outpatient_forgone_any_raw",
            "outpatient_forgone_cost_raw",
            "inpatient_need_but_no_raw",
            "inpatient_forgone_cost_raw",
        ]]
        raw["wave"] = wave

        frame = bridge[bridge["wave"] == wave].merge(raw, on=["wave", "raw_id_str"], how="left")
        frames.append(frame.drop(columns=["raw_id_str"]))

    return pd.concat(frames, ignore_index=True)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tracked = [
        "doctor_visit_any_wave",
        "doctor_visit_count_wave",
        "hospital_stay_any_wave",
        "iadl5_wave",
        "cesd10",
        "fall_any_raw",
        "fall_count_raw",
        "pain_any_raw",
        "walk_1km_diff_raw",
        "walk_100m_diff_raw",
        "mobility_limit_any_raw",
        "traditional_outpatient_raw",
        "traditional_inpatient_raw",
        "traditional_selfmed_raw",
        "outpatient_forgone_any_raw",
        "outpatient_forgone_cost_raw",
        "inpatient_need_but_no_raw",
        "inpatient_forgone_cost_raw",
    ]
    for col in tracked:
        series = df[col]
        rows.append(
            {
                "variable": col,
                "n_nonmissing": int(series.notna().sum()),
                "nonmissing_pct": round(float(series.notna().mean()), 6),
                "mean_if_numeric": round(float(series.mean()), 6) if series.notna().any() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_PATH, sep="\t", index=False)
    return out


def write_memo(summary: pd.DataFrame) -> None:
    def row(name: str) -> pd.Series:
        return summary[summary["variable"] == name].iloc[0]

    lines = [
        "# Extended Long Build Memo",
        "",
        "## Purpose",
        "Build an enhanced long-format CHARLS main-cohort file that preserves the existing manuscript-facing long table while adding same-dataset extension variables needed for stronger lagged arthritis-functional analyses.",
        "",
        "## Added Harmonized Repeated Variables",
        "- wave-level doctor visit indicator and count",
        "- wave-level hospital stay indicator",
        "- wave-level outpatient and inpatient spending variables",
        "- wave-level IADL summary score for Waves 2 to 4",
        "",
        "## Added Raw-Module Variables",
        "- falls and fall counts",
        "- pain-any indicator",
        "- walking difficulty indicators and a mobility-limit summary",
        "- support-only traditional-treatment indicators from care modules",
        "- outpatient and inpatient care-forgoing indicators (including cost-related forgoing) for Waves 1 to 3",
        "",
        "## Coverage Snapshot",
        f"- `doctor_visit_any_wave` non-missing: `{int(row('doctor_visit_any_wave')['n_nonmissing'])}` ({row('doctor_visit_any_wave')['nonmissing_pct']*100:.2f}%).",
        f"- `hospital_stay_any_wave` non-missing: `{int(row('hospital_stay_any_wave')['n_nonmissing'])}` ({row('hospital_stay_any_wave')['nonmissing_pct']*100:.2f}%).",
        f"- `iadl5_wave` non-missing: `{int(row('iadl5_wave')['n_nonmissing'])}` ({row('iadl5_wave')['nonmissing_pct']*100:.2f}%).",
        f"- `fall_any_raw` non-missing: `{int(row('fall_any_raw')['n_nonmissing'])}` ({row('fall_any_raw')['nonmissing_pct']*100:.2f}%).",
        f"- `pain_any_raw` non-missing: `{int(row('pain_any_raw')['n_nonmissing'])}` ({row('pain_any_raw')['nonmissing_pct']*100:.2f}%).",
        f"- `mobility_limit_any_raw` non-missing: `{int(row('mobility_limit_any_raw')['n_nonmissing'])}` ({row('mobility_limit_any_raw')['nonmissing_pct']*100:.2f}%).",
        f"- `outpatient_forgone_cost_raw` non-missing: `{int(row('outpatient_forgone_cost_raw')['n_nonmissing'])}` ({row('outpatient_forgone_cost_raw')['nonmissing_pct']*100:.2f}%).",
        f"- `inpatient_forgone_cost_raw` non-missing: `{int(row('inpatient_forgone_cost_raw')['n_nonmissing'])}` ({row('inpatient_forgone_cost_raw')['nonmissing_pct']*100:.2f}%).",
        "",
        "## Interpretation",
        "The project now has a practical same-dataset route to strengthen the lagged question without switching databases. The next step should be to rerun lagged models with sharper subgroup and outcome definitions rather than continue relying on the coarse original exposure-outcome pairing.",
        "",
        "## Files",
        f"- Extended long table: `{EXT_LONG_PATH.relative_to(ROOT)}`",
        f"- Coverage summary: `{SUMMARY_PATH.relative_to(ROOT)}`",
    ]
    MEMO_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    base = pd.read_csv(LONG_PATH, sep="\t")
    base["ID"] = pd.to_numeric(base["ID"], errors="coerce").astype("Int64")

    bridge = build_bridge()
    harm_ext = build_harmonized_extensions()
    raw_status_ext = build_raw_status_extensions(bridge)
    raw_care_ext = build_raw_care_extensions(bridge)

    ext = base.merge(harm_ext, on=["ID", "wave"], how="left")
    ext = ext.merge(raw_status_ext, on=["ID", "wave"], how="left")
    ext = ext.merge(raw_care_ext, on=["ID", "wave"], how="left")

    ext.to_csv(EXT_LONG_PATH, sep="\t", index=False, compression="gzip")
    summary = build_summary(ext)
    write_memo(summary)

    print(f"Wrote {EXT_LONG_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote {MEMO_PATH}")


if __name__ == "__main__":
    main()
