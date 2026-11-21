# Next-phase recommendation

## Gate 1: Recover V16 artifacts without training

Request/export the exact `gci_cache_v16` folder from the Drive used by the executed notebook, including all LightGBM member NPZs, CatBoost fold NPZs, feature/encoding caches, and any notebook output files. Preserve archive metadata. For each file, bind SHA-256 to current train/test/sample hashes, `SK_ID_CURR` order, validation fold IDs, model/source hash, package versions, feature schema, and configuration. Do not reuse the V15 `gci_cache_final` NPZs as V16 model caches.

If that Drive cache cannot be recovered, classify historical V16 cache provenance as unrecoverable. A new V16 source rerun would be a reproduction attempt, not recovery of original predictions; do not start it until the rule gate below is resolved.

## Gate 2: Competition-rule clarification before any competition use

Obtain organizer guidance in writing on:

1. Unlabeled statistics fitted across train and test covariates.
2. Target encoding and cross-fitted supervised representations.
3. Applicant-to-applicant / test-to-test relational or score-coupling features.

The local official materials do not settle these methods. Until clarified, preserve the current files for forensic research but do not submit a pipeline relying on unresolved methods.

## Gate 3: EXP85

Do not use the 0.79467683 bundle. Recover the exact final Anchor74 OOF and its row/fold map, each suite member OOF and fold metadata, and the final blending code. The current Anchor74 surrogate is a distinct grouped-fold diagnostic and its hybrid suite blend is not an EXP85 comparator.

## Gate 4: Common-fold benchmark

Only after artifact provenance and rule scope pass, compare verified EXP85, reproduced V16, and Phase 2/3 models on a predeclared common validation protocol. Separate model selection from the final comparison labels. Keep failed runs and all previous scores append-only.

No feature expansion, parameter search, new training, or champion promotion is recommended before these gates.
