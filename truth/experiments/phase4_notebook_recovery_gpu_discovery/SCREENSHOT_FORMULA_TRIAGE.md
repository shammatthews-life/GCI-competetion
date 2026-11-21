# Screenshot formula triage

## Method and evidence status

All 10 WhatsApp-named research screenshots were visually inspected. `tesseract` is not installed; the text below is a manual visual transcription, not automated OCR. Four other images in the scanned Downloads folder are unrelated wallpaper/collage images; `download.jpg` and `wallpaper.jpg` are byte-identical (same SHA-256). No screenshot score was recalculated and no feature was added.

Every displayed AUC is an unauthenticated screenshot claim. The screenshots do not establish fold design, direction handling, missing-value treatment, whether the number is a single-feature diagnostic or model AUC, or whether the same rows were used for selection. None is evidence of incremental performance.

## Transcription and assessment

| Screenshot | Readable formula/claim | Status and next check |
|---|---|---|
| `WhatsApp Image 2026-11-03 at 10.58.26 PM (1).jpeg` | Bureau inquiry acceleration: `AMT_REQ_CREDIT_BUREAU_QRT - AMT_REQ_CREDIT_BUREAU_YEAR / 4`; alternative ratio divides QRT by `YEAR/4 + epsilon`. | The local dictionary says MON excludes the prior week, QRT excludes the prior month, and YEAR excludes the prior three months, but the YEAR description literally contains the typo “one day year.” Exact windows/lengths are unresolved; screenshot's `/4` is not substantiated. Do not normalize/compare intervals until clarified. No score shown. |
| `WhatsApp Image 2026-11-03 at 10.58.26 PM.jpeg` | `q = AMT_ANNUITY / AMT_CREDIT`; test whether the repayment-ratio effect varies by external-score profile; the screenshot includes an incomplete fold-results table. | The interaction is testable, but exact table values are not transcribed because the screenshot is too small to read reliably. Use fold-fitted quantile bins only and compare to matched baseline. |
| `WhatsApp Image 2026-11-03 at 10.58.27 PM (1).jpeg` | `DownPaymentProxy = GoodsPrice - Credit`; ratio `(GoodsPrice - Credit) / GoodsPrice`. Screenshot claims AUC about 0.568. | Not a verified down payment: credit may include fees or exceed goods price, making the sign/semantics ambiguous. AUC direction and evaluation method are absent. Test guarded signed difference and ratio separately. |
| `WhatsApp Image 2026-11-03 at 10.58.27 PM (2).jpeg` | `FinancialStress = LTV × PTI / E`; screenshot claims AUC about 0.648. | `E` is not precisely defined in the image. If it is an external score, score direction and missingness matter. The number is not an incremental model gain. |
| `WhatsApp Image 2026-11-03 at 10.58.27 PM.jpeg` | Collateral payment stress: `CPS = AMT_ANNUITY / (AMT_GOODS_PRICE × EXT)`; screenshot says AUC about 0.67067 “on our training data.” | In-sample/training-only wording is explicit; do not compare it with OOF AUC. Confirm payment units and define a stable, missing-aware external-score aggregate before any fold-safe test. |
| `WhatsApp Image 2026-11-03 at 10.58.28 PM (1).jpeg` | External-score disagreement `sigma_E = STD(E1,E2,E3)`; normalized disagreement `sigma_E / mean(E)`; risk interaction `LTV × sigma_E / mean(E)`. | Candidate uncertainty/interactions. Mean can approach zero and missing scores alter the denominator; use guarded alternatives (absolute dispersion, valid-count-aware summaries) and compare with existing EXT features. No readable metric. |
| `WhatsApp Image 2026-11-03 at 10.58.28 PM.jpeg` | Exact fixed-rate amortization relation `A/C = r(1+r)^n / ((1+r)^n - 1)` and numerical inversion for candidate durations; also shows an unverified public-score comparison around 0.7679 vs 0.76794. | Exact equation is valid only under a specified payment frequency, fixed rate, and term. Credit and payment alone do not identify both rate and term; each chosen duration gives a different implied rate. Public-score text is not source-verified. |
| `WhatsApp Image 2026-11-03 at 10.58.29 PM (1).jpeg` | `DSCR_proxy = Income / (12 × Annuity)` and `DSCR_proxy × EXT`. | If income is annual and annuity monthly, the first is the inverse of `12 × PTI`; it adds no new information beyond PTI for a monotonic tree transform. The dictionary gives no annuity frequency, so the annualization assumption is unverified. The EXT interaction is the only distinct hypothesis. |
| `WhatsApp Image 2026-11-03 at 10.58.29 PM.jpeg` | `LTV = Credit / GoodsPrice`; `RALTV = Credit / (GoodsPrice × EXT)`. | The dictionary defines goods price for consumer loans, so this is not automatically a universal collateral LTV. The adjusted ratio is also unstable for small/absent EXT values and overlaps existing financial/EXT interactions. Use only a fold-safe, guarded variant where contract semantics support it; no metric shown. |
| `WhatsApp Image 2026-11-03 at 10.58.30 PM.jpeg` | Approximate implied annual rate `((Annuity × n / Credit)^(12/n)) - 1`, alongside the exact amortization equation. | The approximation is not the exact amortizing-loan solution. The exact solution still depends on an assumed `n`, payment periodicity, and fee treatment. Do not name this an observed interest rate. |

## Rule and validation gate

All readable formulas use provided application fields, but the local dictionary leaves annuity frequency unspecified and has ambiguous YEAR inquiry wording; a locally reviewed rule report also says unlabeled train/test-statistic fitting is unresolved. The screenshot concepts are not cleared features. Any follow-up must:

1. verify field semantics/units in the supplied competition dictionary;
2. calculate all fitted transforms within each training fold;
3. compare candidate and baseline on identical untouched folds;
4. report valid/missing/zero-denominator counts, extreme values and direction-adjusted univariate diagnostics separately from model AUC;
5. preregister the formula and avoid choosing bins/weights from the outer-validation labels.
