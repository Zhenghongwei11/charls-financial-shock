#!/usr/bin/env bash
set -euo pipefail

python3 scripts/build_extended_long.py
python3 scripts/preprocess_financial_toxicity_data.py
python3 scripts/build_ft_selection_qc.py

python3 scripts/run_ft_lagged_models.py
python3 scripts/run_ft_lagged_models_maincohort.py
python3 scripts/run_ft_strengthening_analyses.py
python3 scripts/run_ft_healthshock_sensitivity.py
python3 scripts/run_ft_reporting_diagnostics.py

python3 scripts/run_ft_mechanism_utilization.py
python3 scripts/run_ft_constraint_interaction.py
python3 scripts/run_ft_mediation_proxy_chain.py
python3 scripts/run_ft_cesd_chain.py

python3 scripts/build_ft_trajectory_profiles.py
python3 scripts/build_ft_trajectory_profiles_maincohort.py

python3 scripts/build_ft_fig1_cohort_selection.py
python3 scripts/build_ft_fig2_shock_thresholds.py
python3 scripts/build_ft_fig3_primary_dose_response.py
python3 scripts/build_ft_evidence_stack.py
python3 scripts/plot_ft_evidence_stack.py
python3 scripts/build_ft_manuscript_tables.py

python3 scripts/write_ft_software_versions.py

echo "Done. See results/ for outputs."

