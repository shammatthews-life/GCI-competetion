# GCI v21 — tariff-distance continuation

## Why this iteration

The user-reported v20 public score was 0.7789, effectively indistinguishable at the displayed precision from the v15 champion's 0.77891. The previous candidate perturbed the champion by only 0.5%. The tariff-distance LightGBM branch was independently positive against its compact baseline in all 6 validation folds across two seeds, so the next test increases its rank contribution moderately rather than adding the rejected categorical family-ID feature.

## Evidence

* Seed 20261026: baseline OOF AUC 0.756518; tariff-distance OOF AUC 0.759753; +0.003236.
* Seed 20261027: baseline OOF AUC 0.755399; tariff-distance OOF AUC 0.757979; +0.002579.
* Both seeds: 6/6 held-out folds positive.
* The 50:50 blend of compact baseline and tariff model underperformed the tariff model alone, so it is rejected.
* The categorical tariff-family ID experiment was rejected after negative fold deltas (−0.0131, −0.0068 in the first two folds).

## Candidate files

* Primary: `GCI\_submission\_v21\_tariff\_distance\_1p5pct.csv` (1.5% tariff challenger).
* Sensitivity only: `GCI\_submission\_v21\_tariff\_distance\_2p5pct\_sensitivity.csv` (2.5%).

## Critical validation limit

The v15 out-of-fold/cache predictions are unavailable in this runtime. Thus, compact-model OOF gains are not proof that either candidate beats the full v15 ensemble. These are controlled leaderboard probes, not validated wins. The 1.5% candidate is recommended as the next single probe because it is a modest increase from v20's 0.5% contribution. Preserve v15 as fallback; submit only one candidate and record its exact score before testing the sensitivity file.

## Metrics and hashes

```json
{
  "champion": "GCI\_submission\_v15\_rankspace\_consistent\_view.csv",
  "champion\_public\_score\_reported": 0.77891,
  "v20\_public\_score\_reported\_by\_user": "0.7789 (rounded; no demonstrated gain)",
  "basis": "Average rank of two independent tariff-distance LightGBM challengers from seeds 20261026 and 20261027. Both improved their own compact baseline on all 6 validation folds; the baseline/challenger blend lost to the tariff model alone, so use tariff model alone as the challenger.",
  "results": \[
    {
      "file": "GCI\_submission\_v21\_tariff\_distance\_1p5pct.csv",
      "challenger\_weight": 0.015,
      "rank\_corr\_v15\_vs\_challenger": 0.9481651337631181,
      "mean\_abs\_rank\_change": 0.00099501179853262,
      "max\_abs\_rank\_change": 0.010987560975609711,
      "id\_order\_valid": true,
      "finite\_scores": true,
      "sha256": "f2aeab18d082a763f295449e04bafbd6a956d54aaaafe847454f59f91d4f3891"
    },
    {
      "file": "GCI\_submission\_v21\_tariff\_distance\_2p5pct\_sensitivity.csv",
      "challenger\_weight": 0.025,
      "rank\_corr\_v15\_vs\_challenger": 0.9481651337631181,
      "mean\_abs\_rank\_change": 0.0016583529975543669,
      "max\_abs\_rank\_change": 0.018312601626016223,
      "id\_order\_valid": true,
      "finite\_scores": true,
      "sha256": "6b42df3c0ec8def1ba4f869fe6895f7e761719817a7667441ff88767a644af18"
    }
  ],
  "important\_limit": "No v15 OOF/cache predictions available in this runtime. Neither candidate is proven to beat v15. The 1.5% candidate is the primary next leaderboard probe; 2.5% is sensitivity only. Submit one at a time reportGCI\_submission\_v21\_tariff\_distance\_2p5pct\_sensitivity has 0.77888 , GCI\_submission\_v21\_tariff\_distance\_1p5pct has 0.77889."
}
```

