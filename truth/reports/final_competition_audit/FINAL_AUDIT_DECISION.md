# Final competition audit decision

## Requested status

- **Official rules read:** COMPLETE for all materials present in the local
  competition package; external/current organizer updates were not checked.
- **Winner class inspected:** COMPLETE.
- **Friend code located:** YES — standalone UMAP research notebook and
  `final_submission.rar`.
- **Friend code audit:** **BLOCKED / NOT CLEARED**. Source inspection is
  complete, but feature/prediction cache provenance is absent and the
  feature-driven test-score coupling needs organizer clarification.
- **TRUTH source audit:** **FAIL as a repository-wide certification**;
  preserved Phase 2 constants and legacy artifacts remain unresolved.
- **EXP02 notebook audit:** **PASS** for the explicit local permitted-data and
  automatic-prediction rules. Its exact source hash matches the run manifest,
  and the suspected `test_raw` overwrite is not present.
- **Feature provenance:** **FAIL overall**; EXP02's seven formulas trace to
  provided columns, but friend caches and the historical `129.5` / `0.0132`
  constants do not have complete provenance.
- **Overall audit gate:** **BLOCKED**.

## Why the friend solution is not cleared

1. The notebook's global feature construction leaks validation covariates and
   target information into the outer-fold evaluation; supervised UMAP and
   target encoding are not nested, and early stopping uses the reported
   validation fold.
2. The final stack searches blend weights on the same labels used for its
   final reported OOF AUC.
3. The reproduction script is cache replay/synthesis, not raw-data training:
   it depends on absent feature matrices, OOF/test prediction arrays, labels,
   test IDs, and metadata without their producing code or verified provenance.
4. It modifies scores for selected feature-matched test pairs. The local rule
   explicitly bans manual, EDA-rule decisions for part of predictions. This
   code is automated, but the selection-and-coupling behavior is close enough
   to that prohibition that written organizer clarification is required
   before any use.
5. Its README's public and OOF scores and submission-file hash are claims only;
   this audit did not execute the code or inspect the bundled CSV.

These findings do **not** prove the friend used external data or intentionally
violated a rule. They are enough to conclude that the code cannot be certified
as compliant or relied on as valid OOF evidence from the supplied package.

## Immediate recommendation

Do **not** copy, run, or submit the friend's archive, caches, ensemble, or
graph coupling. Keep EXP02 as a separate provisional research result; it has
not been independently replicated with a second seed, has no official
leaderboard score, and is not an accepted submission candidate. Keep the
historical champion and previous experiment evidence unchanged.

Before any reuse, request written organizer guidance on:

- Fitting unsupervised preprocessing or feature statistics using unlabeled
  test covariates.
- Supervised low-dimensional embeddings and target encoding when nested
  strictly inside training folds.
- Feature-derived repeat-entity matching and score coupling on selected test
  rows.
- Pretrained models or external prediction caches, including the provenance
  needed to prove they use only permitted data.
- Any AI-assistance disclosure policy not included in the supplied package.

No next-experiment proposal is prepared because the audit gate did not pass.
No training, code execution of the friend artifacts, submission creation,
commit, or push was performed.
