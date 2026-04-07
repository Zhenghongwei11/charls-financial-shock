# Statistical Decision Rules (Public Repro Package)

## Primary exposure definition
- “OOP shock” is defined as an interval-specific upper-tail threshold on **raw annualized OOP at t−1**.
- Main definition: q = 0.95 computed within each lag interval (2011–2013, 2013–2015, 2015–2018).
- Sensitivity: q ∈ {0.90, 0.95, 0.975, 0.99}.

## Primary outcome
- Subsequent ADL worsening is modeled as ΔADL/Δyears (per-year change in ADL limitations) over adjacent-wave intervals.

## Inference and reporting
- Regression models use clustered standard errors by participant where applicable.
- Reported intervals are 95% confidence intervals.
- P values are reported to three decimals (e.g., P = .004) when ≥.001; smaller values may be reported as P < .001.

## Missingness and structural zeros
- No silent missing→0 for spending variables.
- Structural zeros are allowed only when a utilization indicator implies no use (e.g., visit=0 and count missing → count set to 0).

