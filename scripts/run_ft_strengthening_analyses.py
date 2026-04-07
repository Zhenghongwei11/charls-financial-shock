#!/usr/bin/env python3
"""
加强分析统一脚本：
  加强1：医保缓冲效应（oop_shock × insurance_public_w1 交互）
  加强1b：医保缓冲敏感性（在控制连续支出 log_oop 后的 oop_shock × insurance 交互）
  加强2a：城乡异质性（oop_shock × residence_rural_w1 交互）
  加强2b：性别异质性（oop_shock × sex 交互）
  加强3：剂量反应关系（连续性 vs 二元暴露对比）
  加强4：临床意义量化（效应量与已知风险因素比较）

输出：
  results/effect_sizes/ft_strengthening_analyses.tsv
  results/effect_sizes/ft_clinical_significance.tsv
  results/effect_sizes/ft_primary_models.tsv
  results/figures/ft_forest_strengthening.png
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "derived" / "charls_financial_toxicity_processed.tsv.gz"
EFFECT_DIR = ROOT / "results" / "effect_sizes"
FIG_DIR = ROOT / "results" / "figures"
QC_DIR = ROOT / "results" / "qc"

WAVE_TO_YEARS = {1: 0, 2: 2, 3: 4, 4: 7}
SHOCK_QUANTILE = 0.95  # 主分析用 p95


def ensure_dirs() -> None:
    EFFECT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── 数据准备 ───────────────────────────

def build_intervals(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df["t_years"] = df["wave"].map(WAVE_TO_YEARS).astype(float)
    df = df.sort_values(["ID", "t_years"]).reset_index(drop=True)

    group = df.groupby("ID", sort=False)
    df["wave_lag"] = group["wave"].shift(1)
    df["t_years_lag"] = group["t_years"].shift(1)
    df["adl5_lag"] = group["adl5"].shift(1)
    df["core_disease_count_lag"] = group["core_disease_count"].shift(1)
    df["total_annual_oop_raw_lag"] = group["total_annual_oop_raw"].shift(1)
    df["ft_burden_ratio_raw_lag"] = group["ft_burden_ratio_raw"].shift(1)
    df["log_oop_raw_lag"] = np.log1p(df["total_annual_oop_raw_lag"])
    df["log_burden_ratio_raw_lag"] = np.log1p(df["ft_burden_ratio_raw_lag"])

    df["dt_years"] = df["t_years"] - df["t_years_lag"]
    df["delta_adl"] = df["adl5"] - df["adl5_lag"]
    df["delta_adl_per_year"] = df["delta_adl"] / df["dt_years"]

    df["interval"] = df["wave_lag"].map({1: "2011-2013", 2: "2013-2015", 3: "2015-2018"})
    df = df[df["interval"].notna()].copy()
    return df


def compute_shock_thresholds(interval_df: pd.DataFrame, q: float) -> dict[int, float]:
    thresholds = {}
    for wave_lag in [1, 2, 3]:
        subset = interval_df[interval_df["wave_lag"] == wave_lag]
        vals = subset["total_annual_oop_raw_lag"].dropna()
        if not vals.empty:
            thresholds[int(wave_lag)] = float(vals.quantile(q))
    return thresholds


def apply_shock(interval_df: pd.DataFrame, thresholds: dict[int, float], q: float) -> pd.DataFrame:
    col = f"oop_shock_p{int(q * 100)}_lag"
    interval_df[col] = np.where(
        interval_df["wave_lag"].isin(thresholds.keys()),
        interval_df["total_annual_oop_raw_lag"] >= interval_df["wave_lag"].map(thresholds),
        np.nan,
    ).astype("float")
    return col


def prepare_analysis_df(long_df: pd.DataFrame, q: float = SHOCK_QUANTILE) -> tuple[pd.DataFrame, str]:
    df = build_intervals(long_df)
    thresholds = compute_shock_thresholds(df, q)
    shock_col = apply_shock(df, thresholds, q)
    # 确保协变量是正确类型（insurance 有少量缺失，保留 NaN）
    df["sex"] = df["sex"].astype(int)
    df["residence_rural_w1"] = df["residence_rural_w1"].astype(int)
    df["insurance_public_w1"] = pd.array(df["insurance_public_w1"], dtype="Int64")
    df["education_c"] = df["education_c"].astype(int)
    return df, shock_col


# ─────────────────────────── 模型拟合工具 ───────────────────────────

def tidy_row(fit, model_id: str, outcome: str, n: int, n_ids: int, term_filter: list[str] | None = None) -> pd.DataFrame:
    rows = []
    conf = fit.conf_int()
    for term in fit.params.index:
        if term_filter and term not in term_filter:
            continue
        if term not in conf.index:
            continue
        rows.append({
            "model_id": model_id,
            "outcome": outcome,
            "term": term,
            "estimate": float(fit.params[term]),
            "std_error": float(fit.bse.get(term, np.nan)),
            "p_value": float(fit.pvalues.get(term, np.nan)),
            "ci_low": float(conf.loc[term, 0]),
            "ci_high": float(conf.loc[term, 1]),
            "n_obs": int(n),
            "n_ids": int(n_ids),
            "r2": float(getattr(fit, "rsquared", np.nan)),
        })
    return pd.DataFrame(rows)


COVARIATES = "adl5_lag + core_disease_count_lag + baseline_age + sex + education_c + residence_rural_w1 + C(interval)"


def fit_change_model(df: pd.DataFrame, formula: str, model_id: str) -> pd.DataFrame:
    # 从公式中提取所有预测变量名（排除 C(interval) 这种固定效应）
    rhs = formula.split("~", 1)[1]
    # 移除 C(interval)
    rhs_clean = rhs.replace("C(interval)", "")
    # 拆分成变量名（处理交互项中的 :）
    raw_terms = [t.strip() for t in rhs_clean.split("+") if t.strip()]
    # 交互项拆成单独的列名
    needed_cols = set()
    for term in raw_terms:
        for sub in term.split(":"):
            sub = sub.strip()
            if sub and not sub.startswith("C("):
                needed_cols.add(sub)

    sub = df.dropna(subset=["delta_adl_per_year"] + list(needed_cols)).copy()
    sub = sub[sub["interval"].notna()]
    if sub.empty:
        return pd.DataFrame()
    fit = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["ID"]})
    return tidy_row(fit, model_id, "delta_adl_per_year", len(sub), sub["ID"].nunique())


# ─────────────────────────── 加强 1：医保缓冲 ───────────────────────────

def analysis_insurance_interaction(df: pd.DataFrame, shock_col: str) -> pd.DataFrame:
    """oop_shock × insurance_public_w1 交互"""
    formula = f"delta_adl_per_year ~ {shock_col} + insurance_public_w1 + {shock_col}:insurance_public_w1 + {COVARIATES}"
    rows = fit_change_model(df, formula, "insurance_interaction")
    if not rows.empty:
        rows["analysis"] = "insurance_buffer"
    return rows


def analysis_insurance_interaction_logshock(df: pd.DataFrame, shock_col: str) -> pd.DataFrame:
    """
    保险缓冲敏感性：在控制连续支出水平 (log_oop_raw_lag) 后，检验 shock × insurance。
    解释口径更“干净”：保险是否缓冲“阈值冲击的额外效应”。
    """
    formula = f"delta_adl_per_year ~ log_oop_raw_lag + {shock_col} + insurance_public_w1 + {shock_col}:insurance_public_w1 + {COVARIATES}"
    rows = fit_change_model(df, formula, "insurance_interaction_logshock")
    if not rows.empty:
        rows["analysis"] = "insurance_buffer_sensitivity_logshock"
    return rows


# ─────────────────────────── 加强 2a：城乡异质性 ───────────────────────────

def analysis_rural_interaction(df: pd.DataFrame, shock_col: str) -> pd.DataFrame:
    """oop_shock × residence_rural_w1 交互"""
    # residence_rural_w1 main effect is already included in COVARIATES
    formula = f"delta_adl_per_year ~ {shock_col} + {shock_col}:residence_rural_w1 + {COVARIATES}"
    rows = fit_change_model(df, formula, "rural_interaction")
    if not rows.empty:
        rows["analysis"] = "heterogeneity_rural"
    return rows


# ─────────────────────────── 加强 2b：性别异质性 ───────────────────────────

def analysis_sex_interaction(df: pd.DataFrame, shock_col: str) -> pd.DataFrame:
    """oop_shock × sex 交互"""
    # sex main effect is already included in COVARIATES
    formula = f"delta_adl_per_year ~ {shock_col} + {shock_col}:sex + {COVARIATES}"
    rows = fit_change_model(df, formula, "sex_interaction")
    if not rows.empty:
        rows["analysis"] = "heterogeneity_sex"
    return rows


# ─────────────────────────── 加强 3：剂量反应 ───────────────────────────

def analysis_dose_response(df: pd.DataFrame, shock_col: str) -> pd.DataFrame:
    """连续性暴露（log_oop）vs 二元冲击的对比"""
    rows_list = []

    # 模型 A：仅连续性
    formula_a = f"delta_adl_per_year ~ log_oop_raw_lag + {COVARIATES}"
    r_a = fit_change_model(df, formula_a, "dose_continuous_only")
    if not r_a.empty:
        r_a["analysis"] = "dose_response"
        rows_list.append(r_a)

    # 模型 B：连续性 + 二元冲击同时入模
    formula_b = f"delta_adl_per_year ~ log_oop_raw_lag + {shock_col} + {COVARIATES}"
    r_b = fit_change_model(df, formula_b, "dose_continuous_plus_shock")
    if not r_b.empty:
        r_b["analysis"] = "dose_response"
        rows_list.append(r_b)

    # 模型 C：仅二元冲击（基准）
    formula_c = f"delta_adl_per_year ~ {shock_col} + {COVARIATES}"
    r_c = fit_change_model(df, formula_c, "dose_shock_only")
    if not r_c.empty:
        r_c["analysis"] = "dose_response"
        rows_list.append(r_c)

    return pd.concat(rows_list, ignore_index=True) if rows_list else pd.DataFrame()


# ─────────────────────────── 加强 4：临床意义量化 ───────────────────────────

def analysis_clinical_significance(df: pd.DataFrame, shock_col: str) -> pd.DataFrame:
    """将 shock 效应量与已知风险因素（疾病计数、年龄）比较"""
    # 用标准模型（已有的主模型）提取所有协变量的效应量
    formula = f"delta_adl_per_year ~ {shock_col} + {COVARIATES}"
    rows = fit_change_model(df, formula, "clinical_comparison")
    if not rows.empty:
        rows["analysis"] = "clinical_significance"
    return rows


# ─────────────────────────── Forest Plot ───────────────────────────

def plot_forest_strengthening(all_rows: pd.DataFrame) -> None:
    """绘制加强分析的 forest plot"""
    # 只取关键交互项和主效应
    terms_of_interest = [
        "oop_shock_p95_lag",
        "oop_shock_p95_lag:insurance_public_w1",
        "oop_shock_p95_lag:residence_rural_w1",
        "oop_shock_p95_lag:sex",
        "log_oop_raw_lag",
        "core_disease_count_lag",
        "baseline_age",
    ]

    # Curate to avoid duplicate terms across overlapping models (e.g., insurance sensitivity).
    preferred_model_ids = [
        "insurance_interaction",
        "rural_interaction",
        "sex_interaction",
        "dose_continuous_only",
        "dose_continuous_plus_shock",
        "clinical_comparison",
    ]
    cand = all_rows[all_rows["term"].isin(terms_of_interest)].copy()
    cand["model_priority"] = cand["model_id"].apply(lambda m: preferred_model_ids.index(m) if m in preferred_model_ids else 999)
    cand = cand.sort_values(["term", "model_priority"]).copy()
    plot_df = cand.groupby("term", as_index=False).head(1).copy()
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("term").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_positions = range(len(plot_df))

    for i, (_, row) in enumerate(plot_df.iterrows()):
        est = row["estimate"]
        ci_low = row["ci_low"]
        ci_high = row["ci_high"]
        ax.plot([ci_low, ci_high], [i, i], "k-", linewidth=1.5, alpha=0.7)
        ax.plot(est, i, "o", markersize=6, color="#1f77b4")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels([t.replace(":", " × ") for t in plot_df["term"].values], fontsize=9)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Estimate (95% CI)", fontsize=10)
    ax.set_title("Strengthening Analyses: Financial Shock & ADL Decline", fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    out_path = FIG_DIR / "ft_forest_strengthening.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")

def export_clinical_significance_table(all_rows: pd.DataFrame) -> None:
    """
    Export a compact, manuscript-facing summary table for:
    - insurance buffering (% reduction)
    - dose-response pattern
    - clinical equivalence (shock vs chronic disease)
    """
    out_path = EFFECT_DIR / "ft_clinical_significance.tsv"

    rows: list[dict[str, object]] = []

    def add(metric_id: str, description: str, estimate, p_value=None, ci_low=None, ci_high=None, source_model_id=None, source_term=None, n_obs=None, n_ids=None) -> None:
        rows.append(
            {
                "metric_id": metric_id,
                "description": description,
                "estimate": float(estimate) if estimate is not None and pd.notna(estimate) else np.nan,
                "p_value": float(p_value) if p_value is not None and pd.notna(p_value) else np.nan,
                "ci_low": float(ci_low) if ci_low is not None and pd.notna(ci_low) else np.nan,
                "ci_high": float(ci_high) if ci_high is not None and pd.notna(ci_high) else np.nan,
                "source_model_id": source_model_id,
                "source_term": source_term,
                "n_obs": int(n_obs) if n_obs is not None and pd.notna(n_obs) else np.nan,
                "n_ids": int(n_ids) if n_ids is not None and pd.notna(n_ids) else np.nan,
            }
        )

    # Insurance buffer summary
    ins = all_rows[(all_rows["analysis"] == "insurance_buffer") & (all_rows["model_id"] == "insurance_interaction")].copy()
    if not ins.empty:
        shock = ins[ins["term"] == "oop_shock_p95_lag"]
        inter = ins[ins["term"] == "oop_shock_p95_lag:insurance_public_w1"]
        if not shock.empty and not inter.empty:
            shock_est = float(shock.iloc[0]["estimate"])
            inter_est = float(inter.iloc[0]["estimate"])
            insured_est = shock_est + inter_est
            pct_reduction = (1.0 - (insured_est / shock_est)) if shock_est != 0 else np.nan
            add(
                "insurance_interaction_beta",
                "Interaction term (shock × public insurance) on ΔADL/year",
                inter.iloc[0]["estimate"],
                inter.iloc[0]["p_value"],
                inter.iloc[0]["ci_low"],
                inter.iloc[0]["ci_high"],
                "insurance_interaction",
                "oop_shock_p95_lag:insurance_public_w1",
                inter.iloc[0]["n_obs"],
                inter.iloc[0]["n_ids"],
            )
            add(
                "shock_effect_uninsured_beta",
                "Shock effect on ΔADL/year among uninsured (reference group)",
                shock_est,
                shock.iloc[0]["p_value"],
                shock.iloc[0]["ci_low"],
                shock.iloc[0]["ci_high"],
                "insurance_interaction",
                "oop_shock_p95_lag",
                shock.iloc[0]["n_obs"],
                shock.iloc[0]["n_ids"],
            )
            add(
                "shock_effect_insured_beta",
                "Shock effect on ΔADL/year among publicly insured (β_shock + β_interaction)",
                insured_est,
                None,
                None,
                None,
                "insurance_interaction",
                "oop_shock_p95_lag + oop_shock_p95_lag:insurance_public_w1",
                shock.iloc[0]["n_obs"],
                shock.iloc[0]["n_ids"],
            )
            add(
                "insurance_buffer_pct_reduction",
                "Percent reduction of shock effect with public insurance (1 - β_insured/β_uninsured)",
                pct_reduction,
                None,
                None,
                None,
                "insurance_interaction",
                "derived",
                shock.iloc[0]["n_obs"],
                shock.iloc[0]["n_ids"],
            )

    # Insurance buffer (log+shock sensitivity)
    ins_s = all_rows[(all_rows["analysis"] == "insurance_buffer_sensitivity_logshock") & (all_rows["model_id"] == "insurance_interaction_logshock")].copy()
    if not ins_s.empty:
        shock = ins_s[ins_s["term"] == "oop_shock_p95_lag"]
        inter = ins_s[ins_s["term"] == "oop_shock_p95_lag:insurance_public_w1"]
        if not shock.empty and not inter.empty:
            shock_est = float(shock.iloc[0]["estimate"])
            inter_est = float(inter.iloc[0]["estimate"])
            insured_est = shock_est + inter_est
            pct_reduction = (1.0 - (insured_est / shock_est)) if shock_est != 0 else np.nan
            add(
                "insurance_interaction_beta_logshock",
                "Interaction term (shock × public insurance) controlling for log OOP (sensitivity)",
                inter.iloc[0]["estimate"],
                inter.iloc[0]["p_value"],
                inter.iloc[0]["ci_low"],
                inter.iloc[0]["ci_high"],
                "insurance_interaction_logshock",
                "oop_shock_p95_lag:insurance_public_w1",
                inter.iloc[0]["n_obs"],
                inter.iloc[0]["n_ids"],
            )
            add(
                "shock_effect_uninsured_beta_logshock",
                "Shock effect among uninsured controlling for log OOP (reference group; sensitivity)",
                shock_est,
                shock.iloc[0]["p_value"],
                shock.iloc[0]["ci_low"],
                shock.iloc[0]["ci_high"],
                "insurance_interaction_logshock",
                "oop_shock_p95_lag",
                shock.iloc[0]["n_obs"],
                shock.iloc[0]["n_ids"],
            )
            add(
                "shock_effect_insured_beta_logshock",
                "Shock effect among insured controlling for log OOP (β_shock + β_interaction; sensitivity)",
                insured_est,
                None,
                None,
                None,
                "insurance_interaction_logshock",
                "oop_shock_p95_lag + oop_shock_p95_lag:insurance_public_w1",
                shock.iloc[0]["n_obs"],
                shock.iloc[0]["n_ids"],
            )
            add(
                "insurance_buffer_pct_reduction_logshock",
                "Percent reduction of shock effect with public insurance controlling for log OOP (sensitivity)",
                pct_reduction,
                None,
                None,
                None,
                "insurance_interaction_logshock",
                "derived",
                shock.iloc[0]["n_obs"],
                shock.iloc[0]["n_ids"],
            )

    # Dose-response summary
    dose = all_rows[all_rows["analysis"] == "dose_response"].copy()
    if not dose.empty:
        cont_only = dose[(dose["model_id"] == "dose_continuous_only") & (dose["term"] == "log_oop_raw_lag")]
        cont_plus = dose[(dose["model_id"] == "dose_continuous_plus_shock") & (dose["term"] == "log_oop_raw_lag")]
        shock_only = dose[(dose["model_id"] == "dose_shock_only") & (dose["term"] == "oop_shock_p95_lag")]
        if not cont_only.empty:
            add("dose_continuous_only_beta", "Continuous exposure (log OOP) only", cont_only.iloc[0]["estimate"], cont_only.iloc[0]["p_value"], cont_only.iloc[0]["ci_low"], cont_only.iloc[0]["ci_high"], cont_only.iloc[0]["model_id"], cont_only.iloc[0]["term"], cont_only.iloc[0]["n_obs"], cont_only.iloc[0]["n_ids"])
        if not cont_plus.empty:
            add("dose_continuous_plus_shock_beta", "Continuous exposure (log OOP) with binary shock included", cont_plus.iloc[0]["estimate"], cont_plus.iloc[0]["p_value"], cont_plus.iloc[0]["ci_low"], cont_plus.iloc[0]["ci_high"], cont_plus.iloc[0]["model_id"], cont_plus.iloc[0]["term"], cont_plus.iloc[0]["n_obs"], cont_plus.iloc[0]["n_ids"])
        if not shock_only.empty:
            add("dose_shock_only_beta", "Binary shock only (p95)", shock_only.iloc[0]["estimate"], shock_only.iloc[0]["p_value"], shock_only.iloc[0]["ci_low"], shock_only.iloc[0]["ci_high"], shock_only.iloc[0]["model_id"], shock_only.iloc[0]["term"], shock_only.iloc[0]["n_obs"], shock_only.iloc[0]["n_ids"])

    # Clinical equivalence summary (shock vs chronic disease from the same model)
    cs = all_rows[(all_rows["analysis"] == "clinical_significance") & (all_rows["model_id"] == "clinical_comparison")].copy()
    if not cs.empty:
        shock_row = cs[cs["term"] == "oop_shock_p95_lag"]
        disease_row = cs[cs["term"] == "core_disease_count_lag"]
        age_row = cs[cs["term"] == "baseline_age"]
        if not shock_row.empty:
            add("clinical_shock_beta", "Shock effect (p95) on ΔADL/year", shock_row.iloc[0]["estimate"], shock_row.iloc[0]["p_value"], shock_row.iloc[0]["ci_low"], shock_row.iloc[0]["ci_high"], "clinical_comparison", "oop_shock_p95_lag", shock_row.iloc[0]["n_obs"], shock_row.iloc[0]["n_ids"])
        if not disease_row.empty:
            add("clinical_per_disease_beta", "Per-chronic-disease effect on ΔADL/year", disease_row.iloc[0]["estimate"], disease_row.iloc[0]["p_value"], disease_row.iloc[0]["ci_low"], disease_row.iloc[0]["ci_high"], "clinical_comparison", "core_disease_count_lag", disease_row.iloc[0]["n_obs"], disease_row.iloc[0]["n_ids"])
        if not age_row.empty:
            add("clinical_per_year_age_beta", "Per-year age effect on ΔADL/year", age_row.iloc[0]["estimate"], age_row.iloc[0]["p_value"], age_row.iloc[0]["ci_low"], age_row.iloc[0]["ci_high"], "clinical_comparison", "baseline_age", age_row.iloc[0]["n_obs"], age_row.iloc[0]["n_ids"])
        if not shock_row.empty and not disease_row.empty:
            shock_est = float(shock_row.iloc[0]["estimate"])
            disease_est = float(disease_row.iloc[0]["estimate"])
            ratio = shock_est / disease_est if disease_est != 0 else np.nan
            add("clinical_shock_vs_disease_ratio", "Ratio β_shock / β_per_disease (≈ chronic disease equivalents)", ratio, None, None, None, "clinical_comparison", "derived", shock_row.iloc[0]["n_obs"], shock_row.iloc[0]["n_ids"])

    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {out_path}")

def export_primary_models_table(all_rows: pd.DataFrame) -> None:
    """
    Export a compact table for manuscript main + supplement:
      - Main: shock-only (dose_shock_only)
      - Supplement: log+shock (dose_continuous_plus_shock)
    """
    out_path = EFFECT_DIR / "ft_primary_models.tsv"
    keep = []

    shock_only = all_rows[(all_rows["analysis"] == "dose_response") & (all_rows["model_id"] == "dose_shock_only") & (all_rows["term"] == "oop_shock_p95_lag")]
    if not shock_only.empty:
        r = shock_only.iloc[0].to_dict()
        r["role"] = "main"
        keep.append(r)

    log_shock = all_rows[(all_rows["analysis"] == "dose_response") & (all_rows["model_id"] == "dose_continuous_plus_shock") & (all_rows["term"].isin(["log_oop_raw_lag", "oop_shock_p95_lag"]))]
    for _, row in log_shock.iterrows():
        r = row.to_dict()
        r["role"] = "supplement"
        keep.append(r)

    if keep:
        out = pd.DataFrame(keep)
        out.to_csv(out_path, sep="\t", index=False)
        print(f"Wrote {out_path}")


# ─────────────────────────── 主函数 ───────────────────────────

def main() -> None:
    ensure_dirs()
    print("Loading data...")
    long_df = pd.read_csv(INPUT_PATH, sep="\t")
    df, shock_col = prepare_analysis_df(long_df, q=SHOCK_QUANTILE)
    print(f"Analysis dataset: {len(df)} rows, {df['ID'].nunique()} IDs")
    print(f"Shock column: {shock_col}, threshold by wave_lag: {compute_shock_thresholds(df, SHOCK_QUANTILE)}")

    results = []

    # 加强 1：医保
    print("\n[1/5] Insurance buffer interaction...")
    r1 = analysis_insurance_interaction(df, shock_col)
    if not r1.empty:
        results.append(r1)
        print(f"  Done: {len(r1)} terms, n_obs={r1.iloc[0]['n_obs']}")

    print("[1b/5] Insurance buffer sensitivity (log+shock)...")
    r1b = analysis_insurance_interaction_logshock(df, shock_col)
    if not r1b.empty:
        results.append(r1b)
        print(f"  Done: {len(r1b)} terms, n_obs={r1b.iloc[0]['n_obs']}")

    # 加强 2a：城乡
    print("[2a/5] Rural heterogeneity interaction...")
    r2a = analysis_rural_interaction(df, shock_col)
    if not r2a.empty:
        results.append(r2a)
        print(f"  Done: {len(r2a)} terms, n_obs={r2a.iloc[0]['n_obs']}")

    # 加强 2b：性别
    print("[2b/5] Sex heterogeneity interaction...")
    r2b = analysis_sex_interaction(df, shock_col)
    if not r2b.empty:
        results.append(r2b)
        print(f"  Done: {len(r2b)} terms, n_obs={r2b.iloc[0]['n_obs']}")

    # 加强 3：剂量反应
    print("[3/5] Dose-response (continuous vs binary)...")
    r3 = analysis_dose_response(df, shock_col)
    if not r3.empty:
        results.append(r3)
        print(f"  Done: {len(r3)} rows")

    # 加强 4：临床意义
    print("[4/5] Clinical significance comparison...")
    r4 = analysis_clinical_significance(df, shock_col)
    if not r4.empty:
        results.append(r4)
        # 打印关键比较
        key_terms = r4[r4["term"].isin(["oop_shock_p95_lag", "core_disease_count_lag", "baseline_age"])]
        if not key_terms.empty:
            print("  Key effect sizes for clinical significance:")
            for _, row in key_terms.iterrows():
                print(f"    {row['term']}: β={row['estimate']:.4f}, p={row['p_value']:.4f}")

    # 合并输出
    all_results = pd.concat(results, ignore_index=True)
    out_path = EFFECT_DIR / "ft_strengthening_analyses.tsv"
    all_results.to_csv(out_path, sep="\t", index=False)
    print(f"\nWrote {out_path} ({len(all_results)} rows)")

    # Forest plot
    print("\nGenerating forest plot...")
    plot_forest_strengthening(all_results)

    # Clinical significance table (compact summary)
    export_clinical_significance_table(all_results)
    export_primary_models_table(all_results)

    # 打印总结
    print("\n" + "=" * 60)
    print("SUMMARY OF STRENGTHENING ANALYSES")
    print("=" * 60)

    # 交互项总结
    for analysis_name in ["insurance_buffer", "heterogeneity_rural", "heterogeneity_sex"]:
        sub = all_results[all_results["analysis"] == analysis_name]
        interaction_terms = sub[sub["term"].str.contains(":", na=False)]
        if not interaction_terms.empty:
            for _, row in interaction_terms.iterrows():
                sig = "✓" if row["p_value"] < 0.05 else " "
                print(f"  [{sig}] {analysis_name}: {row['term']} = {row['estimate']:.4f} (p={row['p_value']:.4f})")

    # 临床意义总结
    cs = all_results[all_results["analysis"] == "clinical_significance"]
    shock_row = cs[cs["term"] == "oop_shock_p95_lag"]
    disease_row = cs[cs["term"] == "core_disease_count_lag"]
    if not shock_row.empty and not disease_row.empty:
        shock_est = shock_row.iloc[0]["estimate"]
        disease_est = disease_row.iloc[0]["estimate"]
        ratio = shock_est / disease_est if disease_est != 0 else float("nan")
        print(f"\n  Clinical significance:")
        print(f"    Financial shock effect  : β = {shock_est:.4f}")
        print(f"    Per-disease effect      : β = {disease_est:.4f}")
        print(f"    Ratio (shock/disease)   : {ratio:.2f}x")
        if 0.5 <= ratio <= 2.0:
            print(f"    → Comparable to adding ~{ratio:.1f} chronic disease(s)")


if __name__ == "__main__":
    main()
