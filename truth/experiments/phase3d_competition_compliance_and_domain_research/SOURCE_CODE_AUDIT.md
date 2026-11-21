# Source and notebook audit — Phase 3D

## Files inspected

- `C:\Users\SHARON\Downloads\GCI_v5_dropPF4_colab.ipynb` (10 cells; all saved stdout read).
- `D:\gci\truth\research_archive\exp85_phase2_source_snapshot\project\source\run_independent_baseline.py` (preserved model and row-wise feature builder).
- `D:\gci\truth\experiments\phase3b_deep_validation\phase3b_ablation.py` (fold-local product-family construction).
- `D:\gci\truth\experiments\phase3c_model_intelligence\phase3c_initial_benchmark.py` and `phase3c_advanced_discovery.py` (model, fold, and fixed-iteration methodology).
- `D:\gci\truth\experiments\phase3c_model_intelligence\MODEL_BENCHMARK_OOF_PREDICTIONS.csv`.
- `D:\gci\truth\experiments\phase3c_model_intelligence\MODEL_RUNTIME_RESULTS.csv`.
- `D:\gci\truth\experiments\phase2_advanced_feature_engineering\FOLD_ASSIGNMENTS.csv`.
- `D:\gci\rules and info\04. Competition (due Nov 20)\Competition\HomeCredit_columns_description.xlsx`.
- `D:\gci\truth\reports\final_competition_audit\OFFICIAL_RULES_AUDIT.md` and the preserved compliance matrix.

## Notebook and saved-score findings

- The named notebook was found at the Downloads path. A separate `D:\exp colab\GCI_v5_dropPF4_colab.ipynb` copy was not found.
- It contains ten cells. The code cells have `execution_count=None`, while saved stream outputs remain. These outputs are historical notebook text, not proof the current workspace reran the notebook.
- Cell 1 finds CSVs by recursive globbing and writes caches/results in Drive paths. The original source has machine/environment-dependent inputs.
- Cell 2 selects the expected row-count dataset and checks the sample submission/test size.
- Cell 3 constructs raw row-wise finance/external/bureau features, plus pooled category/group-frequency and supervised product target encodings. It concatenates train and test for category coding, value counts, group means, product/fee decoding and key construction. Those pooled unsupervised calculations are transductive; local competition materials do not explicitly settle their permissibility. They were not copied into the Phase 3D probe.
- Product/fee decoding uses a fixed `129.5` correction and frequency comparisons across concatenated train/test observations. The local source does not establish a train-only derivation for this constant. New Phase 3D formulas do not use it.
- Target encodings are nested inside each outer fit fold for model-training rows and use fit labels to encode outer validation rows. The design is materially better than full-data encoding, but reproducibility still depends on the absent caches/source inputs and the fold-level run outputs.
- Fixed boosting rounds are used for outer-fold OOF predictions; the notebook code does not perform outer-fold early stopping.
- Cell 9 requires `gci_cache_repro` to compare old members, then writes two test submissions. That referenced old cache was not found in the inspected local experiment locations. The OOF prediction arrays and submission CSV outputs referenced by the stdout are also absent.
- Saved stdout reports old/new/mix pooled AUC 0.77202/0.77198/0.77213. It reports mean new-minus-old fold delta -0.00005 (2/5 positive), and mean 50/50-mix delta +0.00010 (4/5 positive). The notebook's predeclared gate requires at least +0.00020 and 4/5 folds for the new model; it therefore fails its own gate. The mix does not meet the mean-gain threshold either.
- The notebook's “keep v3_A (0.77746)” figure is a code/comment claim only; no matching local vector/cache was available for independent recalculation.
- v5 output metrics do not replace or contradict Phase 3C's 0.763957 fixed blend: the underlying models, features, scores and fold setup are different. They are not a score trend.

## Formula audit decisions

The baseline builder already includes `CREDIT_TO_GOODS` (LTV), `ANNUITY_TO_INCOME`, `INCOME_PER_FAMILY`, `EXT_MEAN`, `EXT_STD`, and `DEBT_SERVICE_COVERAGE`. Those are not treated as new discoveries. “Down payment” is not a supported semantic interpretation of goods price minus loan credit; `(goods-credit)/goods` is algebraically `1-LTV`. The official local dictionary describes `AMT_ANNUITY` as “Loan annuity” but does not identify a monthly/annual period. Accordingly, annualized PTI, annuity-to-goods flow burdens, and the 12x DSCR interpretation are marked unresolved rather than accepted as domain facts.

The tested score-adjusted LTV uses `(AMT_CREDIT / AMT_GOODS_PRICE) / (EXT_MEAN + 0.05)`. The fixed 0.05 stabilizer limits extreme ratios from near-zero scores; it is not estimated from labels. Source scores in this training set range from approximately `8.17e-8` to `0.948`, so the unstabilized ratio can be extreme. The official dictionary does not define whether a higher score means lower risk; this remains an explicit interpretation caveat.

The available data have `AMT_REQ_CREDIT_BUREAU_HOUR`, `MON`, `QRT`, and `YEAR`; `DAY` and `WEEK` columns are absent. The local dictionary describes separate horizons but does not fully establish comparable count rates. The probe names its candidate a month-to-older-window count ratio, not inquiry acceleration.

The inverse amortization equation has more unknowns than supplied observations: neither number of payments nor payment frequency is in the supplied columns. With a fixed guessed term, the implied-rate solution is monotone in `AMT_ANNUITY / AMT_CREDIT`, a quantity already represented in the baseline. It is not added as a “discovered rate” feature.

## Competition compliance boundary

The Phase 3D tested features use only row-wise arithmetic on supplied training columns. They read no labels in feature creation, no outside data, and no test covariates. The existing official-rules audit notes that transductive pooled train/test statistics are unresolved; such processing in the v5 notebook is therefore not assumed eligible. The fixed folds are reused and were already studied, so the new CV is leakage-controlled but exploratory, not independent confirmation.
