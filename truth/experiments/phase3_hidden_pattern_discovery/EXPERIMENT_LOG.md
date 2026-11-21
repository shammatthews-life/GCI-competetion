# Phase 3 experiment log

## 2026-11-03 — Baseline verification and loan-product discovery

- Command: `python D:\gci\truth\experiments\phase3_hidden_pattern_discovery\run_phase3_hidden_pattern_discovery.py`
- Status: success; started `2026-11-03T13:57:44.871820+00:00`, finished `2026-11-03T13:59:27.328469+00:00`; 102.457 seconds.
- Inputs: supplied train/test CSV and official column dictionary. Shapes: train `(171202, 34)`, test `(61500, 33)`.
- Checks: five historical OOF references reproduced from saved predictions; IDs and targets aligned; saved five-fold assignment matched. Complete 23-line v17 feature register inspected. No test covariates used to fit product decoders.
- Result: 51 registered formula rows; 38 nonduplicate candidate columns evaluated together. Combined family pooled OOF AUC `0.7613527733366198`, delta `+0.000694322741868203` versus saved raw baseline `0.7606584505947516`.
- Code SHA-256 at execution: `d823590ea596f03ca7794755fc711bb55536bc4ab10c5bf8826560a7b803b575`.

## 2026-11-03 — Batch 1 family ablation

- Command: `python D:\gci\truth\experiments\phase3_hidden_pattern_discovery\run_phase3_batch1_family_ablation.py`
- Status: success; started `2026-11-03T14:02:14.568981+00:00`, finished `2026-11-03T14:07:11.958387+00:00`; 297.389 seconds.
- Four feature families were evaluated using the same fixed outer folds and separate inner-only early-stopping decisions.
- Product decoder only: pooled OOF AUC `0.7628264184350585`, delta `+0.002167967840306928`, improved 4/5 folds.
- Ordinary financial formulas: `0.7607239381067367`, delta `+0.0000654875`.
- Product/financial interactions: `0.7603242434265958`, delta `-0.0003342071681557979`.
- All Batch 1 features: `0.7613527733366198`, delta `+0.000694322741868203`.
- Code SHA-256: `8116132f51dfd537f790f04bfa611dbb1e73c07ee52a2366fd7a88ef82d7fdc7`.

## 2026-11-03 — Product component ablation

- Command: `python D:\gci\truth\experiments\phase3_hidden_pattern_discovery\run_phase3_product_component_ablation.py`
- Status: success; started `2026-11-03T14:08:57.203126+00:00`, finished `2026-11-03T14:13:29.866448+00:00`; 272.663 seconds.
- Fold-estimated K lattice only: AUC `0.761012090331348`, delta `+0.0003536397365964117`, improved 4/5 folds.
- Unadjusted Q ratio keys: `0.7609391210805282`, delta `+0.000280670485776624`, improved 3/5 folds.
- Fold-estimated P fee-adjusted keys: `0.7621840766299498`, delta `+0.0015256260351982354`, improved 4/5 folds.
- Combined product decoder: `0.7628264184350585`, delta `+0.002167967840306928`, improved 4/5 folds.
- Code SHA-256: `a06df2ac2d7c6914dab64fdbe5fcda53966ca1f91891a9956d4de51aa2845814`.

## 2026-11-03 — Fee constant comparison

- Command: `python D:\gci\truth\experiments\phase3_hidden_pattern_discovery\run_phase3_fee_constant_ablation.py`
- Status: success; started `2026-11-03T14:14:13.905304+00:00`, finished `2026-11-03T14:17:49.247152+00:00`; 215.342 seconds.
- Fold-estimated P keys: pooled AUC `0.7621840766299498`, delta `+0.0015256260351982354`, improved 4/5 folds.
- Fixed v17 `129.5` P keys: `0.7616556656073487`, delta `+0.0009972150125970858`, improved 5/5 folds.
- Q-key control: `0.7609391210805282`, delta `+0.000280670485776624`, improved 3/5 folds.
- Code SHA-256: `9ae0538a2bf63f7cebd40dddd2c6cd74509cddf4f467e58ed5dd98a320b9aa50`.

## Failed attempts and validation

- Initial main-run attempt failed before modeling because pandas rejected the existing `V17_FEATURE_REUSE_REGISTER.csv` (unquoted commas make its rows malformed CSV). The runner now checks the 23 physical data rows; the register is not model input.
- Second main-run attempt failed at categorical feature construction because the derived frame retained source indexes while the preserved baseline resets indexes. The transform now resets its index; a separate alignment smoke test passed.
- The first `python -m unittest <Windows absolute path>` invocation failed because unittest interpreted the path as a module name. Running the test file directly succeeded: `Ran 5 tests ... OK`.
- A post-run hash audit initially referenced a nonexistent `phase3_advanced_feature_engineering` directory; correcting it to the Phase 2 path verified the unchanged historical OOF and fold hashes.
- No failed run modified EXP85, the champion, v17, or historical Phase 2 outputs.

## Interpretation status

These are discovery results on fixed folds already used in previous project experiments. No feature is promoted; a frozen confirmation or nested evaluation is still required. No test predictions, competition submission, 0.79+ result, or later Phase 3 batch is claimed.

Post-run audit note: the generated formula catalog's `PROD_PF4_FLAG` description stated a P4 frequency comparison, while the implementation compares fit-only P5 and Q5 key frequencies (P4 is the rounded key paired with the flag). `DOMAIN_FORMULA_CATALOG_CORRECTIONS.csv` records this metadata correction and the exact duplicate flag. This correction does not change any feature values or executed model results; the executed source hashes above remain the hashes used for each run.
