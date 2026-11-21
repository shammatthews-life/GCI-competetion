# Phase 4 input and dictionary audit

## Execution record

- Timestamp (UTC): 2026-11-04T03:59:24.781640+00:00
- Command: `python D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery\audit_phase4_inputs.py`
- Source SHA-256: `99fe96d979e439129cf0f0961f1bba4d86f1a06e251b7b5cebaa6575cac5060b`
- No model or feature experiment was run.

## Dataset validation

- `train.csv`: 171202 rows x 34 columns; positive rate 0.0807291971.
- `test.csv`: 61500 rows x 33 columns; TARGET is absent.
- `sample_submission.csv`: 61500 rows x 2 columns; IDs/order exactly match test.
- Train IDs and test IDs are unique within each file; predictor column names/order match after excluding train TARGET.
- Project CSV copies and the competition-package copies have matching SHA-256 for all three files. Full paths, byte counts, and hashes are in `INPUT_HASH_MANIFEST.csv`.
- Input rows are read only to validate schema, IDs, target integrity, and dictionary semantics. No labels were generated or edited.

## Dictionary definitions used in formula triage

- `AMT_ANNUITY`: Loan annuity
- `AMT_CREDIT`: Credit amount of the loan
- `AMT_GOODS_PRICE`: For consumer loans it is the price of the goods for which the loan is given
- `AMT_INCOME_TOTAL`: Income of the client
- `AMT_REQ_CREDIT_BUREAU_HOUR`: Number of enquiries to Credit Bureau about the client one hour before application
- `AMT_REQ_CREDIT_BUREAU_MON`: Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application)
- `AMT_REQ_CREDIT_BUREAU_QRT`: Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application)
- `AMT_REQ_CREDIT_BUREAU_YEAR`: Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)
- `CNT_FAM_MEMBERS`: How many family members does client have
- `EXT_SOURCE_1`: Normalized score from external data source
- `EXT_SOURCE_2`: Normalized score from external data source
- `EXT_SOURCE_3`: Normalized score from external data source
- `TARGET`: Target variable (1 - client with payment difficulties: he/she had late payment more than X days on at least one of the first Y installments of the loan in our sample, 0 - all other cases)

### Bureau inquiry intervals (verbatim local dictionary text)

- `AMT_REQ_CREDIT_BUREAU_HOUR`: Number of enquiries to Credit Bureau about the client one hour before application
- `AMT_REQ_CREDIT_BUREAU_MON`: Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application)
- `AMT_REQ_CREDIT_BUREAU_QRT`: Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application)
- `AMT_REQ_CREDIT_BUREAU_YEAR`: Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)

The local dictionary does not state a payment frequency for `AMT_ANNUITY`. Its `AMT_GOODS_PRICE` definition is scoped to consumer loans; do not interpret credit/goods as universal collateral LTV without checking contract coverage. The YEAR inquiry description literally says “one day year”; this wording is ambiguous and is not silently corrected here. Exact interval lengths must be confirmed before constructing normalized acceleration rates.

## Result

All structural checks passed. This establishes that the local modeling CSVs match the supplied competition-package copies; it does not resolve organizer permission for transductive unlabeled transforms or certify any model score.
