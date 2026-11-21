# EXP85 champion source audit

## Artifact identity

- Source runner: `D:\gci\champion\exp85_solo_runner.py`; SHA-256 `d6e7a398ec4f08b1ff605da4886889d476e14ab9169baa81f00c3f04fede8be9`.
- Frozen submission: `D:\gci\champion\submission_exp85_solo_champion.csv`; SHA-256 `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`.
- Historical reference: `D:\gci\champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv`; SHA-256 `78c31bb4e4534330479faa26861a49559dcadc40c7eac45620b92d7d9e6ea947`.
- The frozen submission and historical reference are byte-identical; schema is `SK_ID_CURR,TARGET`, 61,500 rows.
- The runner is recovered from the original EXP85 notebook, but relies on external cached/test artifacts selected by filename. The exact final output is reproducible only with those upstream artifacts.

## Final composition (as written)

The runner constructs or loads a test-only Anchor74 prediction, rank/probit-transforms eight upstream test predictions, uses fixed normalized weights (V17 e1Aet .2932, XGB10 .2596, V17 final .2051, V55 CatBoost product .0874, V17 e1et .0489, V55 deep ET .0442, V55 all-family .0383, V55 monotone .0233), combines 85% Anchor74 and 15% suite in probit space, then emits unique ordinal percentiles.

The final synthesis section operates on test vectors and does not construct or score a corresponding final OOF vector. Upstream V17/V55/XGB OOF arrays exist, but that is not the same artifact as an OOF vector for the exact Anchor74+suite synthesis. The Anchor74 input itself is a test-only transductive artifact. No exact final fold map or final OOF vector was located in the audited run trees.

## Score and competition-rule boundary

The source contains an `0.800850 LB Record` comment for the Anchor74 component, but this audit did not find official score evidence tied to the frozen EXP85 file hash. The local competition log's official `0.80041` record is tied to a different archived submission hash (`8e8158ae5ea9a83650e46be0edb19497f4745ef2119da39309212e42ac372d2b`), not this `78c31...` file; it must not be transferred.

The supplied local rules do not resolve whether fitting pooled test covariate statistics or feature-driven test-to-test score coupling is permitted. The upstream Anchor74 reconstruction contains a fixed, entity-pair score adjustment; this is high-risk under the no-hand-labeling language and needs organizer clarification. This audit does not certify competition eligibility.

## Comparator decision

No experiment was trained. The exact final champion OOF and matching fold map are absent. Using Phase 2/3C or an upstream member OOF as a replacement would silently change the requested baseline, so fold-level AUC and improvement are intentionally blank in the blocked result CSVs.
