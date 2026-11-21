# OOF integrity report

## Development OOF

- Development rows: 136961; expected five validation folds.
- Unique OOF IDs: 136961.
- Exact ID / TARGET / fold alignment: True / True / True.
- One finite prediction per development row for all variants: True / True.
- Outer validation labels used for model fit, stopping, feature selection, or feature map fitting: **No**.
- Product parameters and maps use outer-fit rows only: **Yes**.
- OOF file SHA-256: `0a1585edf744d529aa120398857602636af857e856b6f1fd715d38224c6407bf`.

## Locked holdout

- Manifest SHA-256: `5f61f347650639d47bda74779046e1f7ab9b5397ad4b314cb43a30588fa7f56b`; development and holdout applicant IDs are disjoint.
- The raw and primary product models were fit on all development rows. The shared iteration count was selected using only an inner development split; product parameters/maps were fit on development rows and applied to the holdout.
- The two preregistered predictions were evaluated once after the primary candidate/procedure was fixed; no retuning occurred.
- **This is not independent confirmation.** Every label in the training table had already appeared in earlier Phase 2/3 OOF research before the new holdout was created. Therefore the holdout metric is a secondary one-time diagnostic only; it cannot satisfy an untouched confirmation criterion or justify model promotion.

## Files and limitations

`DEV_OOF_PREDICTIONS.csv` contains the development OOF predictions; `LOCKED_HOLDOUT_PREDICTIONS.csv` contains IDs and predictions only; `LOCKED_HOLDOUT_ONE_TIME_SCORE_INPUT.csv` adds labels for the one-time audit record. Hashes are in the manifest. The same OOF labels have been examined historically, so development fold scores are not an unbiased independent estimate.
