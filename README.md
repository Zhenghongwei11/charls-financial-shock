# CHARLS Financial Shock → Functional Decline (2011–2018)

This repository contains code and aggregate outputs for a longitudinal analysis of out-of-pocket (OOP) medical spending “shocks” and subsequent functional decline (ADL) among older adults in the China Health and Retirement Longitudinal Study (CHARLS), Waves 2011/2013/2015/2018.

## What is included (and not included)
- Included: analysis scripts (`scripts/`), aggregate results tables/figures (`results/`), and reproducibility documentation (`docs/`).
- Not included: individual-level survey microdata. CHARLS data access requires application/approval; see `docs/DATA_MANIFEST.tsv`.

## Quick start (one-click)
After you obtain and place the CHARLS wave `.dta` files under the expected `data/raw/` paths (see `docs/DATA_MANIFEST.tsv`), run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/reproduce_one_click.sh
```

## Outputs
Key outputs are written under:
- `results/effect_sizes/` — model term tables (TSV)
- `results/tables/` — manuscript-facing tables (Markdown/TSV)
- `results/figures/` — figure PNGs and `*_anchor.tsv` provenance tables

Figure/table provenance is summarized in `docs/FIGURE_PROVENANCE.tsv`.

## Data and code availability
This repository is intended to be archived on Zenodo via the GitHub–Zenodo integration. Please cite the Zenodo **version DOI** corresponding to the release tag used.

