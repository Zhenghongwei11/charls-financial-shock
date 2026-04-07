#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

QC_DIR = ROOT / "results" / "qc"
EFFECT_DIR = ROOT / "results" / "effect_sizes"
TABLE_DIR = ROOT / "results" / "tables"

FLOW_PATH = QC_DIR / "ft_cohort_flow.tsv"
SELECTION_PATH = QC_DIR / "ft_balanced_vs_main_selection.tsv"
PRIMARY_PATH = EFFECT_DIR / "ft_primary_models.tsv"
STRENGTH_PATH = EFFECT_DIR / "ft_strengthening_analyses.tsv"
CLIN_PATH = EFFECT_DIR / "ft_clinical_significance.tsv"
LAGGED_PATH = EFFECT_DIR / "ft_lagged_models.tsv"
HEALTHSHOCK_PATH = EFFECT_DIR / "ft_healthshock_sensitivity.tsv"
EVIDENCE_STACK_PATH = TABLE_DIR / "ft_evidence_stack.tsv"
DIAG_PATH = QC_DIR / "ft_reporting_diagnostics.tsv"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def fmt_ci(estimate: float, ci_low: float, ci_high: float) -> str:
    return f"{estimate:.3f} ({ci_low:.3f}, {ci_high:.3f})"


def write_md_table(title: str, subtitle: str, df: pd.DataFrame, out_md: Path) -> None:
    lines: list[str] = []
    lines.append(f"# {title}")
    if subtitle:
        lines.append("")
        lines.append(subtitle)
    lines.append("")
    lines.append("| " + " | ".join(df.columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()) + " |")
    out_md.write_text("\n".join(lines) + "\n")


def build_table1_selection() -> None:
    flow = pd.read_csv(FLOW_PATH, sep="\t")
    sel = pd.read_csv(SELECTION_PATH, sep="\t")

    # Keep a focused subset for Table 1.
    keep = {
        "baseline_age": "Baseline age, mean (SD)",
        "female_ragender_eq_2": "Female, % (W1)",
        "rural_h1rural_eq_1": "Rural residence, % (W1)",
        "insured_r1higov_eq_1": "Public insurance, % (W1)",
        "adl_w1": "ADL5 at baseline, mean (SD)",
    }
    sub = sel[sel["variable"].isin(list(keep.keys()))].copy()

    rows: list[dict[str, object]] = []
    for var, label in keep.items():
        r = sub[sub["variable"] == var].iloc[0]
        if r["type"] == "continuous":
            balanced = f"{float(r['balanced_mean']):.1f} ({float(r['balanced_sd']):.1f})"
            excluded = f"{float(r['excluded_mean']):.1f} ({float(r['excluded_sd']):.1f})"
        else:
            # binary: mean is a proportion
            balanced = f"{100*float(r['balanced_mean']):.1f}%"
            excluded = f"{100*float(r['excluded_mean']):.1f}%"
        rows.append(
            {
                "Characteristic": label,
                "Balanced cohort (N=4,212)": balanced,
                "Excluded (N=1,300)": excluded,
                "SMD": f"{float(r['smd']):.3f}" if pd.notna(r["smd"]) else "",
            }
        )

    df = pd.DataFrame(rows)
    out_tsv = TABLE_DIR / "ft_table1_selection.tsv"
    out_md = TABLE_DIR / "ft_table1_selection.md"
    df.to_csv(out_tsv, sep="\t", index=False)
    write_md_table("Table 1", "Cohort flow and baseline differences between the balanced cohort and excluded participants.", df, out_md)

    # Also export the cohort flow counts for reference.
    flow_out = TABLE_DIR / "ft_table1_flow_counts.tsv"
    flow.to_csv(flow_out, sep="\t", index=False)


def build_table2_primary() -> None:
    primary = pd.read_csv(PRIMARY_PATH, sep="\t")
    strengthening = pd.read_csv(STRENGTH_PATH, sep="\t")

    # Present the three dose-response models as a single table.
    keep_models = {
        "dose_continuous_only": "Continuous only",
        "dose_continuous_plus_shock": "Continuous + shock",
        "dose_shock_only": "Shock only",
    }
    keep_terms = {
        "log_oop_raw_lag": "log1p(OOP at t−1)",
        "oop_shock_p95_lag": "Shock (P95 OOP at t−1)",
    }
    sub = primary[primary["model_id"].isin(list(keep_models.keys())) & primary["term"].isin(list(keep_terms.keys()))].copy()

    rows: list[dict[str, object]] = []
    for model_id, model_label in keep_models.items():
        for term, term_label in keep_terms.items():
            r = sub[(sub["model_id"] == model_id) & (sub["term"] == term)]
            # `ft_primary_models.tsv` is a compact table and does not include the continuous-only model.
            # Pull that row from the strengthening table to keep Table 2 complete.
            if r.empty and model_id == "dose_continuous_only":
                r = strengthening[(strengthening["model_id"] == "dose_continuous_only") & (strengthening["term"] == term)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append(
                {
                    "Model": model_label,
                    "Term": term_label,
                    "Estimate (95% CI)": fmt_ci(float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])),
                    "p": f"{float(r['p_value']):.4f}",
                    "N_obs": int(r["n_obs"]),
                    "N_ids": int(r["n_ids"]),
                }
            )
    df = pd.DataFrame(rows)
    df = df.rename(columns={"N_obs": "Observations", "N_ids": "Participants"})
    out_tsv = TABLE_DIR / "ft_table2_primary_dose_response.tsv"
    out_md = TABLE_DIR / "ft_table2_primary_dose_response.md"
    df.to_csv(out_tsv, sep="\t", index=False)
    write_md_table("Table 2", "Primary association and dose-response characterization (balanced cohort).", df, out_md)


def build_tableS1_strengthening() -> None:
    strengthening = pd.read_csv(STRENGTH_PATH, sep="\t")
    clin = pd.read_csv(CLIN_PATH, sep="\t")

    rows: list[dict[str, object]] = []

    # Insurance interaction (main)
    r = clin[clin["metric_id"] == "insurance_interaction_beta"].iloc[0]
    rows.append(
        {
            "Module": "Insurance buffering",
            "Term": "shock × public insurance",
            "Estimate (95% CI)": fmt_ci(float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])),
            "p": f"{float(r['p_value']):.4f}",
            "N_obs": int(r["n_obs"]),
            "N_ids": int(r["n_ids"]),
        }
    )

    # Insurance log+shock sensitivity (interaction)
    r2 = strengthening[(strengthening["model_id"] == "insurance_interaction_logshock") & (strengthening["term"] == "oop_shock_p95_lag:insurance_public_w1")].iloc[0]
    rows.append(
        {
            "Module": "Insurance buffering (log+shock)",
            "Term": "shock × public insurance",
            "Estimate (95% CI)": fmt_ci(float(r2["estimate"]), float(r2["ci_low"]), float(r2["ci_high"])),
            "p": f"{float(r2['p_value']):.4f}",
            "N_obs": int(r2["n_obs"]),
            "N_ids": int(r2["n_ids"]),
        }
    )

    # Rural & sex heterogeneity (interactions)
    r3 = strengthening[(strengthening["model_id"] == "rural_interaction") & (strengthening["term"] == "oop_shock_p95_lag:residence_rural_w1")].iloc[0]
    rows.append(
        {
            "Module": "Heterogeneity (rural)",
            "Term": "shock × rural residence",
            "Estimate (95% CI)": fmt_ci(float(r3["estimate"]), float(r3["ci_low"]), float(r3["ci_high"])),
            "p": f"{float(r3['p_value']):.4f}",
            "N_obs": int(r3["n_obs"]),
            "N_ids": int(r3["n_ids"]),
        }
    )

    r4 = strengthening[(strengthening["model_id"] == "sex_interaction") & (strengthening["term"] == "oop_shock_p95_lag:sex")].iloc[0]
    rows.append(
        {
            "Module": "Heterogeneity (sex)",
            "Term": "shock × female sex",
            "Estimate (95% CI)": fmt_ci(float(r4["estimate"]), float(r4["ci_low"]), float(r4["ci_high"])),
            "p": f"{float(r4['p_value']):.4f}",
            "N_obs": int(r4["n_obs"]),
            "N_ids": int(r4["n_ids"]),
        }
    )

    df = pd.DataFrame(rows)
    df = df.rename(columns={"N_obs": "Observations", "N_ids": "Participants"})
    out_tsv = TABLE_DIR / "ft_tableS1_strengthening.tsv"
    out_md = TABLE_DIR / "ft_tableS1_strengthening.md"
    df.to_csv(out_tsv, sep="\t", index=False)
    write_md_table("Table S1", "Strengthening analyses (key interactions).", df, out_md)


def build_tableS2_evidence_stack() -> None:
    stack = pd.read_csv(EVIDENCE_STACK_PATH, sep="\t")
    # Order: core first, then limited/supportive.
    strength_order = {"core": 0, "limited": 1, "supportive": 2}
    stack = stack.copy()
    if "strength" in stack.columns:
        stack["_strength_order"] = stack["strength"].map(lambda s: strength_order.get(str(s), 99))
        stack = stack.sort_values(["_strength_order", "evidence_id"]).drop(columns=["_strength_order"])
    df = stack[
        [
            "evidence_id",
            "strength",
            "line",
            "estimate",
            "ci_low",
            "ci_high",
            "p_value",
            "n_obs",
            "n_ids",
            "note",
        ]
    ].copy()
    df["Estimate (95% CI)"] = df.apply(lambda r: fmt_ci(float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])), axis=1)
    df["p"] = df["p_value"].map(lambda x: f"{float(x):.4g}" if pd.notna(x) else "")
    df = df.rename(columns={"evidence_id": "ID", "line": "Association", "strength": "Strength", "n_obs": "N_obs", "n_ids": "N_ids"})[
        ["ID", "Strength", "Association", "Estimate (95% CI)", "p", "N_obs", "N_ids", "note"]
    ]
    df = df.rename(columns={"N_obs": "Observations", "N_ids": "Participants"})

    out_tsv = TABLE_DIR / "ft_tableS2_evidence_stack.tsv"
    out_md = TABLE_DIR / "ft_tableS2_evidence_stack.md"
    df.to_csv(out_tsv, sep="\t", index=False)
    write_md_table("Table S2", "Converging lines of evidence (associations; not formal mediation).", df, out_md)

def build_tableS3_shock_quantile_sensitivity() -> None:
    df = pd.read_csv(LAGGED_PATH, sep="\t")
    sub = df[(df["outcome"] == "delta_adl_per_year") & (df["model_id"].str.startswith("change_shock_only_cluster_ols_q"))].copy()
    if sub.empty:
        raise SystemExit(f"No shock-only quantile sensitivity rows found in {LAGGED_PATH}. Rerun scripts/run_ft_lagged_models.py")

    # Keep only the shock term itself
    sub = sub[sub["term"].str.startswith("oop_shock_q")].copy()
    sub["Quantile (q)"] = sub["shock_quantile"].map(lambda x: f"{float(x):g}")
    sub["Estimate (95% CI)"] = sub.apply(lambda r: fmt_ci(float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])), axis=1)
    sub["p"] = sub["p_value"].map(lambda x: f"{float(x):.4g}" if pd.notna(x) else "")
    out = sub.sort_values("shock_quantile")[["Quantile (q)", "Estimate (95% CI)", "p", "n_obs", "n_ids"]].rename(
        columns={"n_obs": "Observations", "n_ids": "Participants"}
    )

    out_tsv = TABLE_DIR / "ft_tableS3_shock_quantile_sensitivity.tsv"
    out_md = TABLE_DIR / "ft_tableS3_shock_quantile_sensitivity.md"
    out.to_csv(out_tsv, sep="\t", index=False)
    write_md_table("Table S3", "Sensitivity of the shock threshold choice (shock-only change model; balanced cohort).", out, out_md)


def build_tableS4_healthshock_sensitivity() -> None:
    df = pd.read_csv(HEALTHSHOCK_PATH, sep="\t")
    shock_rows = df[df["term"].eq("oop_shock_p95_lag")].copy()
    if shock_rows.empty:
        raise SystemExit(f"No shock term rows found in {HEALTHSHOCK_PATH}. Rerun scripts/run_ft_healthshock_sensitivity.py")

    model_labels = {
        "base": "Base model",
        "plus_hosp_lag": "+ lagged hospitalization",
        "plus_incident_disease": "+ incident chronic disease",
        "plus_hosp_lag_plus_incident_disease": "+ both proxies",
    }
    shock_rows["Model"] = shock_rows["model_id"].map(lambda x: model_labels.get(str(x), str(x)))
    shock_rows["Estimate (95% CI)"] = shock_rows.apply(
        lambda r: fmt_ci(float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])), axis=1
    )
    shock_rows["p"] = shock_rows["p_value"].map(lambda x: f"{float(x):.4g}" if pd.notna(x) else "")
    out = shock_rows[["Model", "Estimate (95% CI)", "p", "n_obs", "n_ids"]].rename(
        columns={"n_obs": "Observations", "n_ids": "Participants"}
    )

    out_tsv = TABLE_DIR / "ft_tableS4_healthshock_sensitivity.tsv"
    out_md = TABLE_DIR / "ft_tableS4_healthshock_sensitivity.md"
    out.to_csv(out_tsv, sep="\t", index=False)
    write_md_table(
        "Table S4",
        "Health-shock severity proxy sensitivity (shock coefficient across augmented models; balanced cohort).",
        out,
        out_md,
    )

def build_tableS5_ipw_primary_comparison() -> None:
    df = pd.read_csv(LAGGED_PATH, sep="\t")
    sub = df[(df["outcome"] == "delta_adl_per_year") & (df["term"] == "oop_shock_q0_95_lag")].copy()
    if sub.empty:
        raise SystemExit(f"No q0.95 shock term rows found in {LAGGED_PATH}. Rerun scripts/run_ft_lagged_models.py")

    want = {
        "change_shock_only_cluster_ols_q0.95": "Complete-case (clustered OLS)",
        "change_shock_only_ipw_wls_q0.95": "IPW-weighted (WLS; clustered SEs)",
    }
    sub = sub[sub["model_id"].isin(want.keys())].copy()
    if sub.shape[0] != len(want):
        missing = sorted(set(want.keys()) - set(sub["model_id"].unique()))
        raise SystemExit(f"Missing expected models in {LAGGED_PATH}: {missing}")

    sub["Model"] = sub["model_id"].map(lambda x: want.get(str(x), str(x)))
    sub["Estimate (95% CI)"] = sub.apply(lambda r: fmt_ci(float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"])), axis=1)
    sub["p"] = sub["p_value"].map(lambda x: f"{float(x):.4g}" if pd.notna(x) else "")
    out = sub[["Model", "Estimate (95% CI)", "p", "n_obs", "n_ids"]].rename(
        columns={"n_obs": "Observations", "n_ids": "Participants"}
    )

    out_tsv = TABLE_DIR / "ft_tableS5_ipw_vs_completecase.tsv"
    out_md = TABLE_DIR / "ft_tableS5_ipw_vs_completecase.md"
    out.to_csv(out_tsv, sep="\t", index=False)
    write_md_table(
        "Table S5",
        "Inverse-probability weighting (IPW) sensitivity for exposure observation: primary shock-only change model (balanced cohort).",
        out,
        out_md,
    )

def build_tableS6_reporting_diagnostics() -> None:
    df = pd.read_csv(DIAG_PATH, sep="\t")
    if df.empty:
        raise SystemExit(f"No rows found in {DIAG_PATH}. Rerun scripts/run_ft_reporting_diagnostics.py")

    # Pivot to a compact table for supplement reporting
    primary = df[df["module"] == "primary_change_model"].copy()
    gee = df[df["module"] == "poisson_gee_doctor_visit_count"].copy()

    def get_metric(sub: pd.DataFrame, metric: str) -> tuple[float | None, str]:
        r = sub[sub["metric"] == metric]
        if r.empty:
            return None, ""
        row = r.iloc[0]
        return (float(row["value"]) if pd.notna(row["value"]) else None), (str(row.get("note", "")) if "note" in row else "")

    r2, _ = get_metric(primary, "r2")
    fvalue, f_note = get_metric(primary, "fvalue_plain")
    f_p, _ = get_metric(primary, "f_pvalue_plain")
    max_vif, vif_note = get_metric(primary, "max_vif")
    disp, disp_note = get_metric(gee, "dispersion_pearson_chi2_over_df")

    out = pd.DataFrame(
        [
            {"Module": "Primary change model", "Diagnostic": "R²", "Value": f"{r2:.3f}" if r2 is not None else "", "Notes": ""},
            {
                "Module": "Primary change model",
                "Diagnostic": "F-statistic (conventional OLS)",
                "Value": (
                    f"{fvalue:.1f}; p<1e-16"
                    if fvalue is not None and f_p is not None and float(f_p) == 0.0
                    else (f"{fvalue:.1f}; p={f_p:.3g}" if fvalue is not None and f_p is not None else "")
                ),
                "Notes": f_note,
            },
            {
                "Module": "Primary change model",
                "Diagnostic": "Max VIF (core covariates)",
                "Value": f"{max_vif:.2f}" if max_vif is not None else "",
                "Notes": vif_note,
            },
            {
                "Module": "Doctor visit count model",
                "Diagnostic": "Overdispersion (Pearson χ²/df)",
                "Value": f"{disp:.2f}" if disp is not None else "",
                "Notes": disp_note,
            },
        ]
    )

    out_tsv = TABLE_DIR / "ft_tableS6_model_diagnostics.tsv"
    out_md = TABLE_DIR / "ft_tableS6_model_diagnostics.md"
    out.to_csv(out_tsv, sep="\t", index=False)
    write_md_table("Table S6", "Model fit statistics.", out, out_md)


def main() -> None:
    ensure_dirs()
    build_table1_selection()
    build_table2_primary()
    build_tableS1_strengthening()
    build_tableS2_evidence_stack()
    build_tableS3_shock_quantile_sensitivity()
    build_tableS4_healthshock_sensitivity()
    build_tableS5_ipw_primary_comparison()
    build_tableS6_reporting_diagnostics()
    print("Wrote FT manuscript tables under results/tables/")


if __name__ == "__main__":
    main()
