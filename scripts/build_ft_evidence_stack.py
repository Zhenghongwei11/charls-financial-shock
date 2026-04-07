#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EFFECT_DIR = ROOT / "results" / "effect_sizes"
QC_DIR = ROOT / "results" / "qc"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

OUT_TABLE = TABLE_DIR / "ft_evidence_stack.tsv"
OUT_ANCHOR = FIG_DIR / "ft_evidence_stack_anchor.tsv"


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def pick_row(df: pd.DataFrame, **conds: object) -> pd.Series:
    sub = df.copy()
    for k, v in conds.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        raise ValueError(f"No rows match {conds}")
    if len(sub) > 1:
        # Deterministic: pick the first after stable sorting.
        sub = sub.sort_values(list(conds.keys())).head(1)
    return sub.iloc[0]


def main() -> None:
    ensure_dirs()

    primary = pd.read_csv(EFFECT_DIR / "ft_primary_models.tsv", sep="\t")
    clin = pd.read_csv(EFFECT_DIR / "ft_clinical_significance.tsv", sep="\t")
    strengthening = pd.read_csv(EFFECT_DIR / "ft_strengthening_analyses.tsv", sep="\t")
    constraint = pd.read_csv(EFFECT_DIR / "ft_constraint_interaction.tsv", sep="\t")
    mediation = pd.read_csv(EFFECT_DIR / "ft_mediation_chain.tsv", sep="\t")
    cesd = pd.read_csv(EFFECT_DIR / "ft_cesd_chain.tsv", sep="\t")

    rows: list[dict[str, object]] = []

    # 1) shock(t-1) -> ADL worsening(t)
    r1 = pick_row(primary, model_id="dose_shock_only", term="oop_shock_p95_lag")
    rows.append(
        {
            "evidence_id": "E1",
            "strength": "core",
            "line": "shock(t-1) → ADL worsening(t)",
            "dataset": "balanced_ft",
            "model_id": str(r1["model_id"]),
            "outcome": str(r1["outcome"]),
            "term": str(r1["term"]),
            "estimate": float(r1["estimate"]),
            "ci_low": float(r1["ci_low"]),
            "ci_high": float(r1["ci_high"]),
            "p_value": float(r1["p_value"]),
            "n_obs": int(r1["n_obs"]),
            "n_ids": int(r1["n_ids"]),
            "note": "Primary association; longitudinal change model with clustered SEs.",
        }
    )

    # 2) shock(t-1) -> utilization intensity(t) [doctor visit count]
    r2 = mediation[(mediation["module"] == "leg1") & (mediation["model_id"] == "doc_count_pois_gee") & (mediation["term"] == "oop_shock_p95_lag")]
    if r2.empty:
        raise ValueError("Missing leg1 doc_count_pois_gee oop_shock_p95_lag in ft_mediation_chain.tsv")
    r2 = r2.iloc[0]
    rows.append(
        {
            "evidence_id": "E2",
            "strength": "core",
            "line": "shock(t-1) → utilization intensity(t)",
            "dataset": "balanced_ft",
            "model_id": str(r2["model_id"]),
            "outcome": str(r2["outcome"]),
            "term": str(r2["term"]),
            "estimate": float(r2["estimate"]),
            "ci_low": float(r2["ci_low"]),
            "ci_high": float(r2["ci_high"]),
            "p_value": float(r2["p_value"]),
            "n_obs": int(r2["n_obs"]),
            "n_ids": int(r2["n_ids"]),
            "note": "Visit-count intensity proxy; interpret as utilization association, not care-forgoing.",
        }
    )

    # 3) shock × low_wealth -> utilization intensity(t) [interaction]
    r3 = constraint[
        (constraint["dataset"] == "balanced_ft")
        & (constraint["model_id"] == "doc_count_pois_gee")
        & (constraint["term"] == "oop_shock_p95_lag:low_wealth")
    ]
    if r3.empty:
        raise ValueError("Missing doc_count_pois_gee oop_shock_p95_lag:low_wealth in ft_constraint_interaction.tsv")
    r3 = r3.iloc[0]
    rows.append(
        {
            "evidence_id": "E3",
            "strength": "core",
            "line": "shock×low-wealth → utilization intensity(t)",
            "dataset": "balanced_ft",
            "model_id": str(r3["model_id"]),
            "outcome": str(r3["outcome"]),
            "term": str(r3["term"]),
            "estimate": float(r3["estimate"]),
            "ci_low": float(r3["ci_low"]),
            "ci_high": float(r3["ci_high"]),
            "p_value": float(r3["p_value"]),
            "n_obs": int(r3["n_obs"]),
            "n_ids": int(r3["n_ids"]),
            "note": "Constraint-consistent effect modification; low-wealth defined per pipeline rules.",
        }
    )

    # 4) shock×insurance -> ADL worsening(t) [buffering interaction]
    # Prefer the compact clinical significance table when available.
    r4 = clin[clin["metric_id"] == "insurance_interaction_beta"]
    if not r4.empty:
        r4 = r4.iloc[0]
        rows.append(
                {
                    "evidence_id": "E4",
                    "strength": "core",
                    "line": "shock×insurance → attenuated ADL worsening",
                    "dataset": "balanced_ft",
                    "model_id": str(r4["source_model_id"]),
                    "outcome": "delta_adl_per_year",
                    "term": str(r4["source_term"]),
                "estimate": float(r4["estimate"]),
                "ci_low": float(r4["ci_low"]),
                "ci_high": float(r4["ci_high"]),
                "p_value": float(r4["p_value"]),
                "n_obs": int(r4["n_obs"]),
                "n_ids": int(r4["n_ids"]),
                "note": "Public insurance buffering interaction (main strengthening result).",
            }
        )
    else:
        r4b = strengthening[
            (strengthening["model_id"] == "insurance_interaction") & (strengthening["term"] == "oop_shock_p95_lag:insurance_public_w1")
        ]
        if r4b.empty:
            raise ValueError("Missing insurance interaction term in ft_strengthening_analyses.tsv")
        r4b = r4b.iloc[0]
        rows.append(
                {
                    "evidence_id": "E4",
                    "strength": "core",
                    "line": "shock×insurance → attenuated ADL worsening",
                    "dataset": "balanced_ft",
                    "model_id": str(r4b["model_id"]),
                    "outcome": str(r4b["outcome"]),
                    "term": str(r4b["term"]),
                "estimate": float(r4b["estimate"]),
                "ci_low": float(r4b["ci_low"]),
                "ci_high": float(r4b["ci_high"]),
                "p_value": float(r4b["p_value"]),
                "n_obs": int(r4b["n_obs"]),
                "n_ids": int(r4b["n_ids"]),
                "note": "Public insurance buffering interaction (fallback to full table).",
            }
        )

    # 5) cost-related care-forgoing(t) -> subsequent ADL (t+1) [association]
    r5 = mediation[
        (mediation["module"] == "leg2")
        & (mediation["model_id"] == "adl_lead_with_out_forg_cost_cluster_ols")
        & (mediation["term"] == "outpatient_forgone_cost_raw")
    ]
    if r5.empty:
        raise ValueError("Missing outpatient_forgone_cost_raw leg2 row in ft_mediation_chain.tsv")
    r5 = r5.iloc[0]
    rows.append(
        {
            "evidence_id": "E5",
            "strength": "limited",
            "line": "cost-related outpatient forgoing(t) → worse ADL(t+1)",
            "dataset": "balanced_ft",
            "model_id": str(r5["model_id"]),
            "outcome": str(r5["outcome"]),
            "term": str(r5["term"]),
            "estimate": float(r5["estimate"]),
            "ci_low": float(r5["ci_low"]),
            "ci_high": float(r5["ci_high"]),
            "p_value": float(r5["p_value"]),
            "n_obs": int(r5["n_obs"]),
            "n_ids": int(r5["n_ids"]),
            "note": "Skip-pattern/partial-coverage construct; interpret cautiously (association only).",
        }
    )

    # 6) CESD(t) -> subsequent ΔADL(t→t+1) [supportive psychosocial pathway]
    r6 = cesd[
        (cesd["module"] == "leg2")
        & (cesd["model_id"] == "cesd_leg2_delta_per_year_lead_cluster_ols")
        & (cesd["term"] == "cesd10")
    ]
    if r6.empty:
        raise ValueError("Missing cesd10 leg2 row in ft_cesd_chain.tsv")
    r6 = r6.iloc[0]
    rows.append(
        {
            "evidence_id": "E6",
            "strength": "supportive",
            "line": "CESD(t) → ADL worsening(t→t+1)",
            "dataset": "balanced_ft",
            "model_id": str(r6["model_id"]),
            "outcome": str(r6["outcome"]),
            "term": str(r6["term"]),
            "estimate": float(r6["estimate"]),
            "ci_low": float(r6["ci_low"]),
            "ci_high": float(r6["ci_high"]),
            "p_value": float(r6["p_value"]),
            "n_obs": int(r6["n_obs"]),
            "n_ids": int(r6["n_ids"]),
            "note": "Supportive association; shock→CESD was not statistically supported (see CESD chain).",
        }
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLE, sep="\t", index=False)

    # Anchor table for plotting (keep only plot-relevant columns).
    anchor = out[
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
        ]
    ].copy()
    anchor.to_csv(OUT_ANCHOR, sep="\t", index=False)

    print(f"Wrote {OUT_TABLE}")
    print(f"Wrote {OUT_ANCHOR}")


if __name__ == "__main__":
    main()
