# Anomalous bundle investigation

Target file: `D:\gci\gemini\cache\v_16_blend_bundle.npz`.

## Confirmed measurements

- Bundle SHA-256 `9b1626016beb7a4a43bbfd1b38643dd20de64d6de97fb71094176d8a7ced19ac`; size 5279311 bytes; filesystem modified `2026-10-28T16:12:50.683574+00:00`.
- Keys: `oof_e1Aet, test_e1Aet, oof_cb, test_cb, final_oof, final_test_rank, final_auc`.
- `oof_e1Aet`, `oof_cb`, and `final_oof` are element-for-element identical. Shared array SHA-256: `2febb38d37455ee7875a8889a1cb6f4204c822510656904d0380602774649c1e`.
- All three score `0.7946768339` against the current train target in current CSV row order; the stored scalar is `0.782000`. The scalar is not computed from the stored arrays.
- V16 `final_oof` is byte-for-byte identical to `final_oof` in both sibling `v_6_blend_bundle.npz` and `v_8_blend_bundle.npz`. Their corresponding full-test prediction arrays are also identical. The V6 and V8 bundles store fixed scalars `.776` and `.7765`; the V16 bundle stores `.782`.
- None of these bundle arrays includes applicant IDs or fold assignments.

## Producer code explains the duplicate and invalid OOF

Both `D:\gci\gemini\scratch\generate_exp85_solo_notebook.py` (SHA-256 `77607c0fd838b31828b154131c95de7a4557b7f85cc389b95944e119fce4e374`) and `D:\gci\gemini\scratch\build_clean_exp85_solo_notebook.py` (SHA-256 `5504f009d42e809c4aa00e5728a04d1c1dafee29e8e3a63216644536d99e4dd2`) contain the same precursor code:

1. Fit one LightGBM model on `X_train_df` with `y_train` for 150 boosting rounds.
2. Set `v6_oof = bst_v6.predict(X_train_df)`, predicting the same rows used to fit that model. This is **in-sample**, not out-of-fold.
3. Save this single `v6_oof` vector as `v_6.final_oof`, `v_8.final_oof`, and all three V16 OOF keys (`oof_e1Aet`, `oof_cb`, `final_oof`); save the same test vector for several different model fields.
4. Hard-code `final_auc=0.7760`, `0.7765`, and `0.7820` rather than calculate those fields from the predictions.

This code-level reuse pattern exactly matches the recovered sibling bundle array hashes and stored scalar pattern. The source code therefore explains the anomaly: the `0.794677` score is an apparent in-sample train score from a full-fit precursor model, not an OOF estimate; `0.782` is a hard-coded placeholder. The generated scaffold is the strongly supported producer/overwriter. No process log ties a particular script invocation to the file, so exact execution chronology remains unproven; the artifact itself is nevertheless invalid as OOF evidence.

The separate `D:\gci\gemini\train_v16_and_optimized.py` has a genuine-looking OOF-training loop in its source but optimizes ensemble weights on the same OOF labels used to report the selected AUC. It is not the executed V16 notebook source, and its intended outputs do not match the observed exact V6/V8 duplicate-array lineage. Do not conflate this separate optimized script with the scaffold that explains the bundle.

See `OOF_ARRAY_IDENTITY_AUDIT.csv` for each vector's hash, AUC, identity group, and pairwise equality.
