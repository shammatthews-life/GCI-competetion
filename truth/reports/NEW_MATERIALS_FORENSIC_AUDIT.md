# Forensic audit of newly supplied competition materials

## Scope, method, and limits

This report covers the six user-named inputs in `D:\gci\truth`: `zip.zip`,
`notepade.txt`, `process.txt`, `GCI_v7_extra_trees_colab_1.ipynb`,
`GCI_v13_reweight_and_ET_colab_1.ipynb`, and `GCI_hgb_product_test.npy`.
The archive was read in place with Python's ZIP reader; its members were not
extracted, and no archived code/notebook was executed. Notebook source and
embedded outputs were parsed separately. Text and source files were read in
full. CSV members were streamed through their final row; labeled OOF vectors
were rescored only where their labels and row alignment were verified.

The archive has 373 top-level files (including 39 nested ZIPs). Those nested
archives reduce to 30 unique archive byte hashes and contain 429 unique-archive
members. Repeated copies are identified by their hashes; counts below
distinguish physical paths from deduplicated contents. The evidence supports
complete content review of text/source, notebooks, structured files, and CSVs.
The archive also contains opaque media: the 257-second MP4 was format/metadata
inspected, not watched; 174 PNG paths (93 distinct image payloads) were
dimension/hash inspected but not visually/OCR reviewed. One nested NPZ was
corrupt/unreadable as an NPZ despite its containing ZIP passing CRC. These
exceptions are not represented as line-by-line content review.

No experiment in either user-named notebook was run. Both notebook files have
zero execution counts and zero saved outputs. The local machine reports an
NVIDIA GeForce RTX 4050, but `torch.cuda.is_available()` is false and the
notebooks configure LightGBM for CPU. A separate XGBoost CUDA translation of
the v7-style deterministic feature block was subsequently run and is reported
in the Phase 52 addendum; that score is not attributed to either original
notebook.

## File 1 — `D:\gci\truth\GCI_v7_extra_trees_colab_1.ipynb`

1. The notebook has 9 cells: one Markdown description and eight code cells.
2. Its code cells contain 219 source lines; code cells parse successfully.
3. Every code cell has a null execution count, and every cell has no output.
4. It describes a comparison of five existing v6 LightGBM blend members against
   versions with `extra_trees=True`; this is an experiment plan, not a result.
5. The stated motivation is regularizing noisy nested target-encoded product
   keys by randomizing tree split thresholds.
6. Markdown quotes exploratory held-out AUCs: 0.76987 to 0.77065 for core
   one-hot, 0.76960 to 0.77037 in another seed/partition, and 0.76966 to
   0.77105 for all-family one-hot.
7. Those numbers are Markdown claims, not outputs saved in this notebook.
8. The proposed model is LightGBM with five members and fixed weights:
   0.24, 0.12, 0.32, 0.16, and 0.16.
9. The base feature set cleans the employment sentinel, adds missingness,
   external-score aggregates, financial ratios/logs, age/employment ratios,
   bureau inquiry aggregates, value counts, and group-relative differences.
10. A decoded credit/goods tariff feature uses the asserted constant 0.0132.
11. The product-key block uses annuity/credit `q`, a fee-adjusted `Q0`, and
    several rounded keys (`Q3` through `P5`, including `PF4`).
12. The fee estimate 129.5 is selected using pooled train-plus-test key
    frequencies; neither its permitted-data derivation nor its stability is
    established by this notebook.
13. Target encodings use outer training-fold labels, smoothing constants, and
    nested stratified inner folds for training rows; held-out labels are not
    intentionally used in the encoded features.
14. Learned-expectation residual features are generated using KFold on the
    concatenated train-plus-test predictor rows and can be loaded from a v6
    cache; the cache's provenance and exact row/seed correspondence are not
    independently established here.
15. The feature-count and group-mean functions compute statistics on combined
    train and test covariates, a transductive choice that is not explicitly
    authorized by the local competition rules review.
16. The evaluation uses five stratified folds (seed 42) and reports pooled and
    per-fold AUC in code, but this file contains none of those results.
17. The notebook compares v6 and v7 using the same fixed blend weights and
    suggests a gate of mean gain at least 0.0002 and improvement in at least
    four of five folds.
18. It also writes test submissions on a normal run and asserts 61,500 rows,
    unique test IDs, and finite predictions.
19. `THREADS = os.cpu_count()` and LightGBM has no GPU device option; the
    notebook's stated runtime is approximately 1.2 hours on Colab CPU.
20. Main reproducibility risks are required Drive caches, the absent v6
    artifacts, training/test covariate pooling, and empirical constants whose
    provenance is not documented in this notebook.
21. Verdict: a detailed, plausible ExtraTrees ablation specification, but no
    verified v7 result or GPU run is present in the supplied notebook.

## File 2 — `D:\gci\truth\GCI_v13_reweight_and_ET_colab_1.ipynb`

1. The notebook has 10 cells: two introductory Markdown cells and eight
   Python code cells; all code cells parse successfully.
2. The code contains 291 source lines; every execution count is null and all
   saved output arrays are empty.
3. The first description says to rebuild a v9 reproduction and test
   reweighting and ExtraTrees variants; it does not contain a recorded run.
4. The v9 recipe is stated as 50% old blend plus 50% new blend.
5. Each side uses rank-transformed predictions and the same five member
   weights: 0.24 / 0.12 / 0.32 / 0.16 / 0.16.
6. The reported historical local holdout figures (old 0.77202, new 0.77258,
   mix 0.77330) appear only in Markdown; none is verified by this file's
   execution output.
7. A stated public score of 0.77865 is likewise a claim in Markdown, not a
   locally reproduced leaderboard result here.
8. Cell 1 searches Colab/Drive for train/test/sample files and requires
   171,202 training rows and 34 columns; it does not use a GPU runtime.
9. Feature engineering repeats the clean, external-score, financial,
   age/employment, bureau, count, group-difference, and decoded-product
   families documented for v7.
10. The adjusted product uses the fixed fee 129.5 and the tariff key uses
    0.0132; both constants need a permitted-data derivation before reliance.
11. It computes 16 learned-expectation residual predictions from
    HistGradientBoosting regressors and may load a previously generated v6
    cache.
12. Its E1 view models use EXT, loan, and person feature subsets and
    out-of-fold predictions, then append view logits/differences to the main
    LightGBM members.
13. The main LightGBM blend has fixed rounds and several full-data seeds;
    its test prediction is a blend of fold-model and full-data rank scores.
14. Cell 7 optimizes member weights by greedy AUC hill-climbing on five
    nested folds, then uses all-row weights to write a candidate.
15. Cell 8 compares the fixed blend with old/e1/e1-extra-trees mixtures and
    uses a gate of at least +0.0002 mean nested gain and positive on 4/5 folds.
16. The printed bootstrap interval resamples positive and negative rows; it is
    described as information only, not as the decision criterion.
17. **Code/document mismatch:** Markdown says the v13 candidate files are
    written only when a rule passes, but the code writes them whenever
    `SMOKE` is false and merely labels them “submit only if PASS.”
18. The fold/OFF estimates and weight-selection procedure require the cached
    OOF predictions to match their declared folds/seeds; absent cache files
    and empty outputs prevent verification from this notebook alone.
19. The notebook's runtime estimates say approximately three hours for the
    v9 reproduction and 1.7 hours for ExtraTrees on Colab CPU.
20. No code specifies GPU training; this notebook provides no GPU execution
    evidence.
21. Verdict: useful reproduction/ablation code with a real output-gating
    mismatch, but it does not establish a v13 metric or the claimed public
    score.

## File 3 — `D:\gci\truth\notepade.txt`

1. The file is 157,929 bytes and 939 UTF-8 lines; all lines were read.
2. Its contents are a research/conversation transcript rather than a single
   canonical experiment report.
3. The opening research notes describe planned RealMLP/TabM experiments and
   fixed three-fold assignments, not evidence that the models completed.
4. The notes mention a 3,000-row smoke test and GPU detection with CPU
   fallback; that is a proposed execution path, not proof of GPU training.
5. The transcript includes LightGBM as a proposed OOF control.
6. It contains many GCI phase references, model comparisons, feature
   hypotheses, and handoffs between researchers/agents.
7. It discusses product/tariff representations, target/frequency encodings,
   external-score features, residuals, applicant fingerprints, and blending.
8. It mixes measured local OOF statements, notebook/source descriptions,
   hand-entered experiment summaries, and user-reported leaderboard claims.
9. Those evidence classes cannot be treated as interchangeable.
10. Some quoted results describe cached notebook outputs; their model inputs,
    fold identity, and cache provenance need their corresponding artifacts.
11. At least one transcript section says the desired 0.80 churn thresholds
    were not achieved; that belongs to a different project and is not GCI
    evidence.
12. GCI public-score values in the transcript are narrative claims unless
    tied to independently checkable submission history.
13. The text references notebooks, `.npy` arrays, caches, prediction files,
    and Google Drive artifacts that are not all part of this top-level file.
14. It includes proposed GPU work, but the newly supplied v7/v13 notebooks
    themselves have no GPU configuration or saved run.
15. The transcript should be treated as a provenance map and hypothesis log,
    not as a verified score ledger.
16. Useful next step: follow each important claim to a matching source file,
    saved row-aligned OOF array, fold definition, and prediction hash.
17. Do not copy transcript metrics into an official result table without
    recording whether they are measured, claimed, or public-LB reported.
18. Verdict: useful context and leads; not an executable experiment or
    self-validating record of the 0.77/0.78 leaderboard claims.

## File 4 — `D:\gci\truth\process.txt`

1. The file is 329,926 bytes and 5,122 UTF-8 lines; all lines were read.
2. It is an extended process/research transcript containing iterative
   hypotheses, prompt/agent outputs, experiment summaries, and decisions.
3. It includes generic Home Credit material and claims about conventional
   baselines, which are not automatically results on the provided competition
   split.
4. It also records later GCI feature/model work and references to prior
   notebooks, code, prediction files, and cached artifacts.
5. It contains both the author's historical claims of public scores and
   reported local OOF scores.
6. The claimed public results include values around 0.76858, 0.77283,
   0.77865, 0.77884, 0.77891, and 0.78222 in different passages.
7. These values are not authenticated leaderboard records by the transcript
   itself.
8. A separate local OOF-to-leaderboard projection is explicitly described as
   unsupported; it must not be used as evidence of a test score.
9. The transcript includes reports that v9-style models used nested
   target-encoded product keys and multiview/residual features.
10. Other passages describe feature experiments and subsequent blends; they
    should be reconciled to their exact source and OOF files before treating
    the reported deltas as measured.
11. It discusses matching applicants/fingerprints and manual score changes;
    these are compliance-sensitive and not suitable as a shortcut to adopt.
12. The local competition rules review prohibits manual test prediction edits
    and external data; that review also treats train/test covariate fitting as
    ambiguous and advises avoiding it.
13. Reports of GPU hardware or runs in a transcript are claims unless backed
    by runtime logs/model device configuration; none proves that v7/v13 ran
    on GPU.
14. The text contains material from other projects and generic research; those
    scores must not be attributed to Home Credit.
15. Some score narratives change across revisions, so the latest prose is not
    automatically a correction to earlier OOF files.
16. The full file is useful for context and sourcing leads, but does not
    replace executable, fold-aligned evidence.
17. Verdict: preserve the historical claim that scores above 0.77 were
    reported, while marking the public scores as user-reported and local
    scores as independently verified only when arrays/labels align.
18. No new model execution was done from this text.

## File 5 — `D:\gci\truth\GCI_hgb_product_test.npy`

1. This is a binary NumPy file, so it has no source-code/text lines to read.
2. It is NumPy v1 format, dtype `float64`, with shape `(61500,)`.
3. All 61,500 elements are finite.
4. The minimum is 0 and the maximum is 1.
5. Its mean is exactly 0.5 and it has 60,822 unique values.
6. The vector has no IDs and no labels, so row alignment cannot be verified
   from the array itself.
7. Its length matches the known test-set row count, which suggests it is a
   test prediction vector but does not prove which model produced it.
8. It is byte-identical to `zip.zip`'s `GCI_hgb_product_test.npy`.
9. SHA-256: `81ed436322a3cd97c32565cda2e7145b45dff210b7f748df643e5dd2af6de249`.
10. ROC-AUC cannot be computed without test labels, and no leaderboard score
    can be inferred from prediction values.
11. It is not evidence that the v7 or v13 notebook ran.
12. It is not evidence of GPU use.
13. Treat it as an orphan test-score vector until a matching training code,
    ID order, model/configuration, and generation log are found.
14. Verdict: valid numeric array, but unproven provenance and no independently
    measurable score.

## File 6 — `D:\gci\truth\zip.zip`

### Complete-archive coverage

1. The root ZIP contains 373 files: 200 CSV, 63 IPYNB, 23 PY, 39 nested ZIP,
   32 MD, 4 XLSX, 3 DOCX, 3 TXT, 2 NPY, 1 JSON, 1 PDF, 1 PPTX, and 1 MP4.
2. Its 39 nested ZIP paths represent 30 unique byte hashes; their contents
   contain 429 members across the distinct archives.
3. The nested unique contents contain 98 CSV, 7 IPYNB, 35 PY, 21 MD, 19 TXT,
   40 JSON, 11 LOG, 7 JSONL, 11 NPZ, 174 PNG, 2 PDF, 2 PPTX, 1 DOCX, and
   1 XLSX.
4. All 200 root CSVs and 98 nested CSVs were read through their final row.
5. All 63 root notebooks and 7 nested notebooks were parsed; the 23 root and
   35 nested Python files were read statically, never executed.
6. Text/Markdown/log, JSON/JSONL, PDF, OOXML Office documents, and workbook
   cells were read or parsed; no code was run.
7. Exact duplicates were grouped by SHA-256, but all paths and duplicate
   copies remain part of the archive inventory.
8. The archive combines GCI/Home Credit experiments with telecom churn,
   Company A churn/offer work, a GATE syllabus, and sign-language-avatar
   material; the latter projects are not GCI score evidence.
9. There are 179 root CSVs that are test submissions: 61,500 rows with
   `SK_ID_CURR,TARGET`, unique consecutive test IDs in consistent order.
10. Their `TARGET` column is a model score, not a known label. The files cannot
    independently establish their ROC-AUC or public leaderboard score.
11. Eight root submissions include values outside `[0,1]`, so those particular
    outputs are not probabilities even though they may be rank/blend scores.
12. The remaining root CSVs are result/feature/OOF tables; seven inconsistent
    rows occur in `experminetbygeminioct22.csv`.
13. There are 11 nested 61,500-row submissions, also with no labels.
14. The root contains 200 CSV files total; 170 distinct byte hashes and 22
    duplicate-content groups were observed. Nested CSVs have 82 distinct
    hashes and 16 duplicate groups.
15. Eight nested NPZs have `y` vectors exactly matching the local 171,202
    training labels in row order; those provide a basis for local rescoring,
    but do not by themselves prove fold integrity.
16. One NPZ is unlabeled, and one NPZ under the v24 benchmark package is
    unreadable. The containing archive CRC is valid.
17. The top-level NPY test vector duplicates the user-named NPY exactly.
18. The OOF NPY has length 171,202 but no labels/IDs in-array; its alignment is
    not established and it was not scored.
19. The saved notebook outputs establish held-out scores for several historic
    experiments, but do not validate public leaderboard scores.
20. The archive's repeated 0.77/0.78 public-score claims are not independently
    authenticated by the saved submission CSVs, because they contain test
    predictions only.
21. Root and nested member paths, sizes, and content hashes are listed in the
    complete inventory appendix below; score-like names are not treated as
    scores.

### Score evidence found in archive

The seven notebooks with saved outputs are `GCI_qfingerprint_product_colab_top_24.ipynb`,
`GCI_v16_everything_merged_o.ipynb`, `GCI_v17_optimised_O.ipynb`,
`GCI_v18_evidence_driven_o.ipynb`, `GCI_v19_NN_entity_embeddings_colab_0.ipynb`,
`GCI_v19_research_candidate.ipynb`, and `GCI_v21_gated_research_candidate_o.ipynb`.
The archive notebook/code audit counted 58 stream and 10 display output records.
The key output claims are:

| Evidence | Result in saved output | Classification |
|---|---:|---|
| q-fingerprint one-hot / all-family / monotone / three-model rank average | 0.76992 / 0.77070 / 0.77035 / 0.77158 | Saved notebook held-out metrics |
| v16 / v17 / v18 champion blend | 0.77649 / 0.77646 / 0.77666 | Saved notebook held-out metrics |
| v19 neural entity-embedding model | pooled 0.76732; fold AUCs 0.76419, 0.76999, 0.77389, 0.76250, 0.76602 | Saved notebook fold outputs |
| v16-v18 adversarial / permutation diagnostics | about 0.686 / about 0.50 | Saved diagnostic outputs; adversarial AUC indicates distribution shift, not proof of label leakage |
| v3 / v9 / v15 public score | 0.77746 / 0.77865 / 0.77891 | Markdown/source claims, not verified by saved output |
| v15 OOF vs v14G OOF | 0.77410395 vs 0.77393726 (+0.00016669) | Reported local comparison; candidate said not submitted in another file |
| v16 public result and local OOF | 0.78222 / 0.77649 | Conflicting provenance: public value user-reported; local value saved output |

The two named notebooks in this task have no saved outputs. In particular,
their Markdown's quoted historical 0.769–0.771 experiments and public 0.77865
claim must not be described as results of these notebook files.

### Independently rescored OOF evidence

The following archived OOF vectors are aligned to the local training labels
and were rescored locally. They are local metrics, not public leaderboard
scores; array alignment does not independently establish a leakage-free
validation protocol.

| Archived vector family | ROC-AUC | Log-loss |
|---|---:|---:|
| `oof_lgb_xgb.csv` LightGBM | 0.757142 | 0.246726 |
| `oof_lgb_xgb.csv` XGBoost | 0.754068 | 0.247550 |
| v18 alternative-direction package, no-interactions / no-frequency | 0.756371 / 0.756086 | 0.246672 / 0.246751 |
| v24 benchmark chunk1, baseline / annuity formula / tariff tiers / EXT geometry / all | 0.744508 / 0.751764 / 0.745695 / 0.744049 / 0.751007 | 0.250446 / 0.248618 / 0.250132 / 0.250253 / 0.248392 |
| v24 benchmark combo, baseline / annuity formula / tariff tiers / affordability / all | 0.744508 / 0.751764 / 0.745695 / 0.750727 / 0.751840 | 0.250446 / 0.248618 / 0.250132 / 0.248928 / 0.248617 |
| v26 baseline / product Q-TE / category-product TE / conditional formula-EXT / domain cluster / formula-ExtraTrees | 0.758507 / 0.763430 / 0.762489 / 0.758335 / 0.758264 / 0.755606 | 0.246008 / 0.244858 / 0.244955 / 0.246055 / 0.246172 / 0.246860 |
| v29 baseline / hierarchical-plus-formula-burden | 0.758507 / 0.764028 | 0.246008 / 0.244675 |
| v29 seed 20261032 baseline / hierarchical Q4-Q3 / risk-decile residual / income-type backoff / EXT profile / formula-quality burden | 0.758507 / 0.764664 / 0.754579 / 0.763262 / 0.759541 / 0.764098 | 0.246008 / 0.244477 / 0.247270 / 0.244908 / 0.245790 / 0.244697 |
| v29 seed 20261027 baseline / product Q-TE / category-product TE / product Q4-TE / hierarchical combo | 0.757043 / 0.762636 / 0.760887 / 0.762855 / 0.762324 | 0.246413 / 0.245088 / 0.245523 / 0.244976 / 0.245102 |

These independently scored archive artifacts demonstrate local OOF results
around 0.764 on some stored vectors. They do not prove a 0.78 public result,
do not establish a mathematical ceiling, and do not certify every historical
experiment's fold construction.

### Feature and method classification

The GCI archive's recurring feature families include raw application fields;
missingness and employment-sentinel treatment; EXT score products, means,
distances and residuals; credit/income/annuity/goods ratios; rounded
annuity-to-credit product keys; tariff/product decoding; frequency and
target encodings; applicant fingerprints; group-relative and feature-family
models; multiview stacking; ExtraTrees/LightGBM/CatBoost/XGBoost; and rank
blending. These are method categories, not a claim that every variation
improved performance.

Several scripts/notebooks load cached OOF arrays, predictions, Drive files, or
model members. Without the matching source run, hashes, folds, seeds, and
training data, saved values may be scored but not fully reproduced. Repeated
candidate testing and public-score-driven exploration increase selection
bias. The local rules review prohibits manual test-score edits and external
data, and advises against ambiguous train/test covariate pooling. Therefore
manual row-specific score shifts, unsupported external matches, and
leaderboard-picked candidate weights should not be carried forward.

## Overall conclusion and recommended use

The materials support the historical statement that the user reported public
scores in the 0.77–0.78 range, but the archive does not independently certify
those leaderboard entries. It contains genuine saved held-out/OOF evidence
around 0.7716 and 0.7767, plus separately rescored label-aligned OOF vectors
up to 0.7647; these results use different artifact lineages and must not be
combined as if they were one validation run. The two requested v7/v13
notebooks are unexecuted plans and their claims are not their own measurements.

For future work, preserve the original saved outputs, associate every OOF
vector with an explicit fold/seed and row-ID file, verify cached prediction
hashes against generating code, and report local OOF and public leaderboard
scores in separate columns. Keep rule-sensitive target encodings and
train/test pooling separate until their status is clarified. Do not claim GPU
execution without device telemetry and a GPU-configured model run.

## Integrity hashes for the six supplied inputs

| File | SHA-256 |
|---|---|
| `zip.zip` | `85bc4792fcd8c34475f8ed660d9c9f6a9eeb0866665a21e8f819347424c1e66b` |
| `notepade.txt` | `d0bd1fd4414aeda405518e85914210c343bc9ad0d3351186e06d11678318c249` |
| `process.txt` | `fd0bac538f06bbc47e5fa245b9b3a08199a3e4156521d0101616b8b2cb5813bf` |
| `GCI_v7_extra_trees_colab_1.ipynb` | `cc56101fb68fa8fcacaf3a8cccfcb5ba84140c11eb0a99daf20bd40ae7647fca` |
| `GCI_v13_reweight_and_ET_colab_1.ipynb` | `c6cf202432c99dba57ef38b8efc3c41aeddf4614f187f90d50605d1e5a0556de` |
| `GCI_hgb_product_test.npy` | `81ed436322a3cd97c32565cda2e7145b45dff210b7f748df643e5dd2af6de249` |

## Follow-up: file-by-file research catalogs and GPU run

The initial version of this report gave too much space to the archive path
inventory and not enough to what each file contained. The catalogs below add
per-content-group findings for the archive's code/notebooks, research and
document files, and CSVs. Identical copies are grouped while their paths remain
listed. Numeric file claims are labeled as embedded output, saved aligned OOF
results, or prose/source claims; they are not silently promoted to verified
public scores.

The requested notebook experiments use CPU-only LightGBM and
HistGradientBoosting components and need Colab/Drive caches not present here.
The installed LightGBM build cannot train on GPU; CatBoost GPU and XGBoost
CUDA both passed tiny device smoke tests. Following the user's direction to
port CPU-only GCI work, Phase 52 below is an XGBoost CUDA translation of the
v7 label-free feature comparison, not an exact reproduction of the original
LightGBM ensemble.

## Complete path/size/hash inventory

Root inventory lists every physical top-level ZIP member. Nested inventory lists every member from each byte-distinct nested archive once; duplicate nested archive aliases are already represented as separate top-level rows. Hashes are SHA-256. No member was extracted or executed.

### Root ZIP members

| Member path | Bytes | SHA-256 |
|---|---:|---|
| `zip/0011 second[music].txt` | 388648 | `a1c70463947ed67de934e5f49ab9f24d49913ece2507b8cfafe0234deb519863` |
| `zip/A_lgb_product_only (1).csv` | 1615507 | `c7021a645f5c69727473332d32fa79b6cf18e635e5a8212a18e7e9df3860396d` |
| `zip/A_lgb_product_only.csv` | 1615507 | `c7021a645f5c69727473332d32fa79b6cf18e635e5a8212a18e7e9df3860396d` |
| `zip/BASE.csv` | 1665272 | `7511a071c725f228537ac148441472511770b78b4b93face87ebe43e6118ae04` |
| `zip/best_submission (1).csv` | 1666283 | `fe784d4694573d137b059139d385bbbea58224ca45a8d586324dd0e97f3c69af` |
| `zip/best_submission.csv` | 1666283 | `fe784d4694573d137b059139d385bbbea58224ca45a8d586324dd0e97f3c69af` |
| `zip/brf_full_candidate.csv` | 1615216 | `4d530c407b10fb1891b89bcf20e0de73b11a57bf496f72c494170f4abdce0b29` |
| `zip/candidate_balanced.csv` | 1659236 | `d41c2977861b7f593fcdc91ef904fdd9c7b0d9d110c69057cf6d088ce8a634cf` |
| `zip/candidate_conservative.csv` | 1660938 | `925f58728eff5dad854f3fcddec6651a733e69cabfbf4e8ecf8968fed5963941` |
| `zip/candidate_rank.csv` | 1611494 | `2089ae91182097b6d3609cfab4ed12d023bedd7965f269dd08ab10c44e300384` |
| `zip/candidate_validation.csv` | 627 | `6893d5d4dfd394507a18c70f6558c980cb0234ecf8ee0a32ceddd6f97638c4de` |
| `zip/CA_EXT1_3pct_on_0.76835_champion.csv` | 1614591 | `6abfbf4c6ccf8eabaa195e1fc94fe8a40eaa92dda9de041ea28ea6836ee4d1cd` |
| `zip/ca_ext_ABCDE_results_fast2.csv` | 1633 | `465271fbeb00929c5df11a9a9c3ab20e82b7940ee20f81c0d17862f384e5bc89` |
| `zip/champion_80_rc_20.csv` | 1610803 | `5b6ff379b5402ab31baff9b1d45e0bc446b36009738e7bd6d449246000fa5173` |
| `zip/champion_90_rc_10.csv` | 1612246 | `23386d6a260d304281560698cbe5c5ff05af0f63167e28820220d04b2f2355ec` |
| `zip/champion_95_rc_5.csv` | 1613421 | `5f0bc0ab609917a33d9424e755a043c2e8fa88df22c36828ec5e2cacd505cbf4` |
| `zip/champ_lgb_domain_92_5_3.csv` | 1612537 | `7056ab823dd20209d5b473febe24c984dbc75958a4e4d1f33534924f968bc067` |
| `zip/champ_lgb_domain_95_3_2.csv` | 1612744 | `9d1c15a0d99a84540e5e85aa19cc1c182f5cec3c1696256b9eab72ea7199c70d` |
| `zip/champ_lgb_domain_97_2_1.csv` | 1612805 | `c7086a8159d6c2df8c80fe330d28a5f877fb5de83dabb7c528e5a9d5844ad1ec` |
| `zip/Churn_Optimisation_Decision_Package.zip` | 1810652 | `ff6e91c94732befd52012c37c276805b00528a8699a4d90a97aa98adcdb60b2a` |
| `zip/CompanyA_Final_Submission_Package.zip` | 1356913 | `16cfcd7281ea03928759ad70154af70e8107b323ccb75dcb0e4e9d5eaccec887` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip` | 1359960 | `7e7a599d3923a9f8a1a90890cd6e0f90d43dafed71d1e09704002052fab354fa` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (2).zip` | 1359960 | `7e7a599d3923a9f8a1a90890cd6e0f90d43dafed71d1e09704002052fab354fa` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (3).zip` | 1359960 | `7e7a599d3923a9f8a1a90890cd6e0f90d43dafed71d1e09704002052fab354fa` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (4).zip` | 1359960 | `7e7a599d3923a9f8a1a90890cd6e0f90d43dafed71d1e09704002052fab354fa` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (5).zip` | 1359960 | `7e7a599d3923a9f8a1a90890cd6e0f90d43dafed71d1e09704002052fab354fa` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (6).zip` | 1359960 | `7e7a599d3923a9f8a1a90890cd6e0f90d43dafed71d1e09704002052fab354fa` |
| `zip/CompanyA_Omnicampus_Submission_Bundle.zip` | 1359960 | `7e7a599d3923a9f8a1a90890cd6e0f90d43dafed71d1e09704002052fab354fa` |
| `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip` | 87778 | `c53911aa7a04b5497cc552a89f587e34e904dd927a6045fb37592699d29efabf` |
| `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence.zip` | 87778 | `c53911aa7a04b5497cc552a89f587e34e904dd927a6045fb37592699d29efabf` |
| `zip/DA_GATE2027_Syllabus (2).pdf` | 135687 | `2e9fed7ac0f70331c8777852ed8a1d880d2206b1dac8f741a5c925ea24bc8b0e` |
| `zip/experiment_A_CA_EXT1_5fold_results.csv` | 1210 | `c6b269f256e444c47abb1600233962f84081665ee40994063d5d1e119bd8fd2d` |
| `zip/experminetbygeminioct22.csv` | 14794 | `b81ad29ee5933b5528763f7dfefdad6ec13eff8e3fa1c82899e6bdd5a2d730a8` |
| `zip/fair_incremental_cv.csv` | 182 | `8437084fd1c5081f0406a833955a5c34275895adef8fe0b6485269f5ac005c4c` |
| `zip/figures_and_results (1).zip` | 2632195 | `3a7dd9993bd48f1da5f18ac95c7567248b8ade3c55b9bcd7538f95647a4d02e1` |
| `zip/figures_and_results (2).zip` | 2632195 | `0f9ba63a56a1d1747fa600fb03d738ea5c6caaa7d31f2fc6a000e5a359959772` |
| `zip/figures_and_results (3).zip` | 2774163 | `1ebc4c4fdd627bafe648022db7a54633febf98eaeed3aba350b75ea4125726bc` |
| `zip/figures_and_results (4).zip` | 2827608 | `82d5e66bcd946c566d39aa106999fe067e3ea9cdadb006f91c28621cb82f9d8e` |
| `zip/figures_and_results (5).zip` | 2791042 | `72fc7cc58d41110c59337d6644ea5687856fdb771cb23c18e59ac70f63fa5179` |
| `zip/figures_and_results (6).zip` | 2791042 | `4ecd8c817395ad540381231d262841d17d2284365c765e7bda5757a99202205b` |
| `zip/figures_and_results.zip` | 2588043 | `dbcd96e0ba91d44b81ba97f9a0e83ac8b63c9b4769d860866a732fd49b36c6d5` |
| `zip/figures_and_results_v10.zip` | 3111769 | `9b4f8d5627f56ea00fdf9451621aa99c5c2c9a002c11d362ccba0632a7ea7596` |
| `zip/figures_and_results_v11.zip` | 3113585 | `59f5130a0d85edb9c176ff1d656b554b9e36a6606fbebcb17638ec687c31bf1c` |
| `zip/figures_and_results_v9.zip` | 3001849 | `dcebf26dde08b30e9d410108626e0f11d19cefdfaa080b76b0579400a569654a` |
| `zip/FORMULA_FUSION_LAYER2_CA_RESIDUAL.csv` | 1666922 | `076be21f1abfbb6c0ee4f86d01edf47f279217389de80e2f8fdf1e187f0f4ba1` |
| `zip/gbc_full_candidate.csv` | 1660890 | `649f1eaefaa66b2a31aa180b47738bb8a15d65a1482ae5162cbff32348a9551c` |
| `zip/gci_0_762337_credit_term.py` | 3781 | `f927e180cef4324cabb8f7c70e82b7f67ebaab5feb57dc8d73159c8466fd22c9` |
| `zip/gci_0_76370_reconstruction.py` | 5344 | `64e15b637bf80d4038e3e93f3362f85deffc5dcd422d323a75345a70a8cb2c6d` |
| `zip/GCI_2ca_fingerprint_submission.csv` | 1612610 | `89930bbacd019a21ce78ad96d69f257159196be68e10f1644de6263a9a066ad8` |
| `zip/gci_2ca_fingerprint_submission.py` | 6205 | `6bd297b179686928bb7796ea0a4b1048c18ec9ae3c11c0bae33cecb19503a245` |
| `zip/GCI_adaptive_shape_top30_w08_gate93_protected.csv` | 1614706 | `2161f54eab8a03a2c5d8df3d85051678b4411eaecfdf331b2ec25b5675e3a2ed` |
| `zip/GCI_affordability_curve_residual_8pct_gate93.csv` | 1613931 | `fba5ba75d7ee46a908db0528979df6b5e51205bf74d1cf64916132d8a00b8c56` |
| `zip/GCI_allfam_transfer_v1_submission.csv` | 1615318 | `4f98e2532aa3e8ad927e5c3e29b21ed864aef5a860c97a3b9c1bbb2cb037fadd` |
| `zip/GCI_archived_XGB_0.77308_candidate.csv` | 1130421 | `09e1e4bc5fbb80067a6ac3b9ad03570ca4c258ed667b40698a396b0324410fd1` |
| `zip/GCI_CADOWN_fingerprint_submission.csv` | 1613959 | `83df6389efe92c11ba8b316445f4b00e3e209de6cd840a86844ccbbe633a4fb0` |
| `zip/GCI_CAG5_current_0p76837_champion_gate97_submission.csv` | 1614676 | `74c4b46c836e72b79a1d1a1a7e5d471956074b144b0d6c9680ddb38feb42263f` |
| `zip/GCI_CatBoost_Product_Experiment_Package.zip` | 680728 | `09adc937c6854c5aa012ce35a1978be7f28aa26d440e46cd07d5b44863dc5af8` |
| `zip/GCI_catboost_product_step_colab.ipynb` | 25125 | `70417aa8a8ef99e1bdf9017d2afb181b7982c0d6996cace7f39589fc99478f3b` |
| `zip/GCI_CA_EXTQ_v3_submission.csv` | 1614079 | `6940c1ce9498d29555b5b8879b0e8ecc09770cb5a84b68abcdd5e72f07b5ed57` |
| `zip/GCI_CA_EXT_2pct_champion_blend_submission.csv` | 1615516 | `7d57fc2d0ed0a5f5adb1a96944541c65d8812f99ac93dba095c0c4e39f754463` |
| `zip/GCI_CA_EXT_promising_research.zip` | 2615416 | `ababe07568b32aaa014ff3ccd4e1e45d58908f1f66eee7336ab8c689831cf1a9` |
| `zip/gci_CA_fingerprint_v2.py` | 1788 | `7562a0ec69363c67a671b2d7ad910e53dfa6da39ab41349ce6bd69c35fc14326` |
| `zip/GCI_CA_fingerprint_v2_submission.csv` | 1614560 | `ad01410dd2f41489a129d3712075e4ebace2b53acbde717a95bb0f0faa0ef262` |
| `zip/GCI_champion_catboost_rankblend_1pct.csv` | 1350044 | `6aa90d70f072296e04c4f4802ae9d548def9b0951b9b7e5d197f1c3c3dccdeba` |
| `zip/GCI_champion_catboost_rankblend_3pct.csv` | 1350078 | `c4c1a8e7bcea05ad9af49e76bfec0513bf749e4f647bcbe3e5942b0153d23a31` |
| `zip/GCI_champion_catboost_rankblend_4pct.csv` | 1349941 | `0b648d4daca5bf523b3f3430508b164ca1d50b6e4fc826a5114ff7f29f136f5f` |
| `zip/GCI_Community_Research_Full_Extraction_and_Triage.docx` | 224937 | `e306685f015b4604160f0001ef3ba7c158f85351256e96fea62b92d9583f8ed2` |
| `zip/GCI_Competition_Candidates_Package.zip` | 2544243 | `abb7c8c93936b3b077129827269393eae504925661baa1ae7190424960a3e816` |
| `zip/GCI_COMPLETE_CROSS_CHAT_RESEARCH_ARCHIVE_2026-10-20.md` | 92333 | `e7b9e75291a0923a1513419a2881b1f7a5ed86f8ec7c0981094a3e084705dc2a` |
| `zip/GCI_DE_SENT_CA_residual_submission.csv` | 1614060 | `0c9d08429fb04f4b117081adad769911d54eb26f2d20b89bb53e74ba4392ab63` |
| `zip/gci_DE_SENT_CA_residual_submission.py` | 6941 | `0f2407c7e39add288171ec8a67b073a1e40d3439ab7f75378d7e4ca7c6e39d4e` |
| `zip/GCI_exact_rate_ext_affordability_submission.csv` | 1615293 | `fe9d605ab632f9d88dc97bf8fde76c4fa357f504102b567c3118759b1610f40a` |
| `zip/gci_exact_rate_ext_affordability_submission.py` | 12224 | `e273b9d74f48d3683ec7eafeb054aa06fb05f8c9e6a245ad6f2a2f93292cd669` |
| `zip/GCI_experimental_champion_catboost_rankblend_5pct (1).csv` | 1612792 | `92fd4ad9ef6a844aaa0bcafa9bd09da9da9025c0058500de9db9d706d06b8eb4` |
| `zip/GCI_experimental_champion_catboost_rankblend_5pct.csv` | 1612792 | `92fd4ad9ef6a844aaa0bcafa9bd09da9da9025c0058500de9db9d706d06b8eb4` |
| `zip/GCI_experiment_report_catboost_product.md` | 6695 | `60dc714b2f39403faecbb5bbb4621c16db30a4f89febb9fe4de3eaf70a0f10d7` |
| `zip/GCI_ExtraTrees_Direct_Check_Report_2026-10-17.md` | 6177 | `7d8e8c62ca6be5de54c784019e8fd367dde8786659ba2b812a11c5fa3f3628e8` |
| `zip/GCI_FINAL_CANDIDATE_0p5pct_blend_NOT_model_only.csv` | 1587326 | `bcab0d33a565229c2f65486b3a4b5e0c116eb8b118920d1cc53d34529cdecb7b` |
| `zip/GCI_final_pick_bootstrap_cell.py` | 3363 | `441cfd165c82912f307992944ac4b811b217f18546651dc46fbe23309db600a9` |
| `zip/GCI_final_pick_bootstrap_colab (1).ipynb` | 4479 | `40c100cd4334dfbb138b3a67c3126387a5d1d50fbe5611850971c326665de08b` |
| `zip/GCI_final_pick_bootstrap_colab.ipynb` | 4479 | `40c100cd4334dfbb138b3a67c3126387a5d1d50fbe5611850971c326665de08b` |
| `zip/GCI_final_rebuild_v15_cell.py` | 2909 | `ab3844e94014da1ed3965792b1d109d476d55f66d9e15ae51050889500655d2a` |
| `zip/GCI_final_reproduction.ipynb` | 17288 | `781ed5a7cce891cac376228199021e65a334a082cf8b5da83c2ad1a024ef37aa` |
| `zip/GCI_final_reproduction_v9.ipynb` | 25246 | `31958e71db65115b6ba0e6d60ad7c282e243732bc2a8ce108acd56820d318433` |
| `zip/GCI_FINAL_v15_end_to_end_colab.ipynb` | 29292 | `354737859f5c81831ac232d73b1e8ede99706aeb8746a46c2d37a31ffd63599a` |
| `zip/GCI_financial_fingerprint_submission.csv` | 1613943 | `49680c0fbd502a1dd727ef708495d3e1bc44c5d139b206513c48b5d9ea22064e` |
| `zip/gci_financial_fingerprint_submission.py` | 3910 | `ce14b64cbb4fc5f11350e328859538db4ceb55815c7ccda94eeb012e6d3beeb2` |
| `zip/GCI_FULL_CHAT_RESEARCH_LOG_2026-10-18.md` | 35863 | `4c402cf0bacd22bd8fff03cd2ec5442a0d87b857eb3ba737aaeff67d8d83cf7b` |
| `zip/GCI_goss_v1_champion90_goss10.csv` | 1615444 | `c992fd79f7b9826f27e21ea8e529b4e7b850a8719195de7e502cf2ee38ce58b7` |
| `zip/GCI_harmonic_transfer_v1_7pct.csv` | 1615408 | `3ea86224b729f8e8f3b0e458ae5113620461a319bc4d1b773ee0daaa69dad766` |
| `zip/GCI_hgb_product_oof.npy` | 1369744 | `3773011eda7eb143a86f586cf38e0e4ea146906f50c75a157c94188c09857bd9` |
| `zip/GCI_hgb_product_test.npy` | 492128 | `81ed436322a3cd97c32565cda2e7145b45dff210b7f748df643e5dd2af6de249` |
| `zip/GCI_highres_gated_CA_submission.csv` | 1613586 | `bd309d93d824b1d2bf3165c1059fcb55ffb9bf44c4d21c92de41595f03fe3e25` |
| `zip/GCI_highres_lgb_submission.csv` | 1615573 | `c559fed4988ac0ebad4d68dc648a20df28763f63b1d22e32ec8cc38bc60922a9` |
| `zip/gci_highres_lgb_submission.py` | 3906 | `585b25a6505762a96e9b0bbc4e5484458f53c07083a93f51db9e2fe5ee3fb9bb` |
| `zip/GCI_HomeCredit_BestSolution.zip` | 3246909 | `cb012b6d5028bc36985a1c6939a46585188be3b8d1b557de613a61c564be1b78` |
| `zip/GCI_implied_rate_v3_submission.csv` | 1615277 | `7d53884847f8c57967cc4a811adc5072696a0e7675d6c985ef174f0b4aab8cb8` |
| `zip/GCI_MASTER_HANDOFF_2026-10-18.md` | 19656 | `8fbb38416aa25bf32dcebdecf6395e0a5018a0155afec956fc3b0ff09882baf9` |
| `zip/gci_next_rank_tiebreak.csv` | 1609847 | `410b5286ccc5eac184bdb32665be5f2237d333da757aeaabccbaafd643147329` |
| `zip/gci_peer_ablation_round2.zip` | 2713 | `a094fea54ceb1b1b8deb8b6a717d4a534fe678cceb83b31bdb23ea15bbf51071` |
| `zip/GCI_product_decode_colab.ipynb` | 14766 | `3a7132ea08eedc76bfc8b2986c65f0e1697d54cc9639d71166a1eb8c6f26c318` |
| `zip/GCI_product_decode_colab_v2.ipynb` | 15387 | `60a03a875833f6da20bcdbc1503c9ffa3f7d5ae8dc50210b966a87137ea78ce9` |
| `zip/GCI_qfingerprint_ext2bin_micro05pct.csv` | 1353018 | `b4b385c22f575f2eb2f41b3856f3004ca88f116a5293e160485070eb8959a6a6` |
| `zip/GCI_qfingerprint_ext2bin_micro10pct.csv` | 1353018 | `4d908d9eea1553a25ceb88f2baa35b7b64cac9f610a4226e023dc87cf26c7cc9` |
| `zip/GCI_qfingerprint_ext2bin_micro20pct.csv` | 1353018 | `26df18e07dfb3f021dbe7ea2e79b81dd2974aae324450a071ce0919b100d7f15` |
| `zip/GCI_qfingerprint_extmean_micro025pct.csv` | 1353018 | `6e3c64eb1cf1941c29e8e28d57dfa0ca0dbac7df47c1f1dc019a4b87054976f9` |
| `zip/GCI_qfingerprint_extmean_micro05pct.csv` | 1353018 | `c0f4541f22289c505f3c26007e0c4ff5652530dcd9a4b85617ff00ac94baf681` |
| `zip/GCI_qfingerprint_extmean_micro10pct.csv` | 1353018 | `37f22c3eed4bd04325d5a0f4332132ee04fc9b2500cde69046a77165cc9c070c` |
| `zip/GCI_qfingerprint_extmean_micro20pct.csv` | 1353018 | `fa4c5d07ee0a35ece70af6fce1694dbded469601aeca9a67c68ff802b0118cfa` |
| `zip/GCI_qfingerprint_product_colab (1).ipynb` | 16707 | `ef3f85d217c175e8cd9913698e31be054fd129319dc56ffb38cb93b0f499ced7` |
| `zip/GCI_qfingerprint_product_colab (1).py` | 13548 | `b4f1ee9adcf72c73f25e151d9dfb1975ecffb224383db3380cf9fc71e3bd8ac6` |
| `zip/GCI_qfingerprint_product_colab.ipynb` | 16049 | `f4e51d2c8bd403e9b2a95f16f34825e2bb318f993a7a448ff59daa2dda78d349` |
| `zip/GCI_qfingerprint_product_colab.py` | 13548 | `b4f1ee9adcf72c73f25e151d9dfb1975ecffb224383db3380cf9fc71e3bd8ac6` |
| `zip/GCI_qfingerprint_product_colab_top_24.ipynb` | 23660 | `b3842f3082b0e4e00d69af039b92600a3c9f830a9459cd4881f76560c79b6cba` |
| `zip/GCI_qfingerprint_submission (1).py` | 12431 | `d69fd1b440ae67f36241bd989bfb0118d29f9f715601d31725d0c33583222637` |
| `zip/GCI_qfingerprint_submission.py` | 12431 | `d69fd1b440ae67f36241bd989bfb0118d29f9f715601d31725d0c33583222637` |
| `zip/GCI_q_fingerprint_colab.ipynb` | 22603 | `61779bd534c1bbdc42eb3b18d88b72a21218e835f26b98087e2919671271fd5b` |
| `zip/gci_ratecurve_template_v1.py` | 9202 | `b063c11eb0cde04846be020e348d160f279c2f1a9afdea5fefd6885b546f9a72` |
| `zip/GCI_ratecurve_template_v1_submission.csv` | 1614544 | `fb567544f9f6aefdf9efe7ca5eaf2321b447c13b79317d34254d906b0bc8fc1f` |
| `zip/GCI_ratecurve_template_v2_submission.csv` | 1614674 | `1b217a1b30127d6430832e89d9e86f63e84eaee254286b2f227aa3158c35fac0` |
| `zip/GCI_rate_geometry_residual_gate93_w05_submission.csv` | 1614371 | `256ec510b06ac47eb157b3dc5073e9d72394dbd27bf64c3e276f4ecff7c3c499` |
| `zip/GCI_rate_shape_residual_2pct_gate93.csv` | 1615045 | `55dce05eb08f6d083a378b349d774f044959ce27c5354f4bbf136fffd24eadd7` |
| `zip/GCI_rate_shape_residual_3pct_gate93.csv` | 1614687 | `36f8cba3d2ac4278a5eab1237a8417ddba51d5aedd38baf0a7c71437c926fe38` |
| `zip/GCI_rate_shape_residual_4pct_gate93 (1).csv` | 1614878 | `ccb0dcbffa1fe1364065cd2e2d33304b04159205bd92bf4a9905790ffa0e7f9a` |
| `zip/GCI_rate_shape_residual_4pct_gate93.csv` | 1614878 | `ccb0dcbffa1fe1364065cd2e2d33304b04159205bd92bf4a9905790ffa0e7f9a` |
| `zip/GCI_RD3_micro_25pct_of_w01_gate93.csv` | 1596898 | `eee7753809603d912665a6c53761bc8a2a728086a48887c8f9e7cf797957e5b6` |
| `zip/GCI_RD3_micro_50pct_of_w01_gate93 (1).csv` | 1595632 | `97a23bac040cd3a671308893204df4aacbd878ed9a3f12cb14128fa2f064a505` |
| `zip/GCI_RD3_micro_50pct_of_w01_gate93.csv` | 1595632 | `97a23bac040cd3a671308893204df4aacbd878ed9a3f12cb14128fa2f064a505` |
| `zip/GCI_RD3_micro_75pct_of_w01_gate93 (1).csv` | 1597265 | `2d081cd7d09362911aa8846b2d500839ec58debd8a3763246734d89db502432c` |
| `zip/GCI_RD3_micro_75pct_of_w01_gate93.csv` | 1597265 | `2d081cd7d09362911aa8846b2d500839ec58debd8a3763246734d89db502432c` |
| `zip/GCI_RD3_on_076858_gate95_w05.csv` | 1614219 | `2b24eeee487a9ff1aed883f1094cc1cf61a619e9f24507552a41f88d46fdc658` |
| `zip/GCI_RD3_RangeStd_15pct_candidate.csv` | 1614708 | `afd5f52c2b9a6e343131ceae41fb4203575174868e6de56db07944bbd15b8376` |
| `zip/GCI_RD3_reverse_micro_25pct_gate93 (1).csv` | 1596989 | `68210ef7c9eaa618d8e6cd47801d75f9bff1ae70df3a7363e4c9f9ad70864150` |
| `zip/GCI_RD3_reverse_micro_25pct_gate93.csv` | 1596989 | `68210ef7c9eaa618d8e6cd47801d75f9bff1ae70df3a7363e4c9f9ad70864150` |
| `zip/GCI_RD3_reverse_sweep_m10_of_w01_gate93.csv` | 1352061 | `86eabb3659de4f32a2dc1c5e2c86f13b5cec12c2177a8f8171a74e5cbc5144f6` |
| `zip/GCI_RD3_reverse_sweep_m15_of_w01_gate93.csv` | 1352301 | `e850615eb28deae9d30ac149fd6798ff064eea09790b3eb606ae5f74361f0bfd` |
| `zip/GCI_RD3_reverse_sweep_m20_of_w01_gate93.csv` | 1352110 | `ce502dc663184c055b26919461d2d320a033c876157388d2c1df784289cf57e6` |
| `zip/GCI_RD3_reverse_sweep_m30_of_w01_gate93.csv` | 1351917 | `c6491d383490773c70509ec381713344fc094b20f7ed1379540727ebd81265a9` |
| `zip/GCI_RD3_reverse_sweep_m40_of_w01_gate93.csv` | 1352191 | `cd807dd332fd19f8fa98594791b562b3892ea49aa506043f851f54e32f1673d7` |
| `zip/GCI_RD3_reverse_sweep_m50_of_w01_gate93.csv` | 1352115 | `7a95663da3198fda004156c31c48e4d93b3e956b45d2bab55737bde614883a13` |
| `zip/GCI_RD3_transfer_w01_gate93.csv` | 1615052 | `9012f07a438f542056a9928db8cf72057edb62d9bfb39fa390744ac5d3a7dea8` |
| `zip/GCI_RD3_transfer_w02_gate93.csv` | 1615027 | `6589785c8f980ff40566fd20d1377991fc877078f1d4c94731c03c5da3245e30` |
| `zip/GCI_Research_Progress_Report_2026-10-22.docx` | 47420 | `236b561e805cc737075d2855b7f3b13b45485220907bf91f27ef7359ad0a7853` |
| `zip/GCI_Research_Progress_Report_Copy_2026-10-24.docx` | 47073 | `d52209c4c7e086e1752be5f35a2bd3bdf75ac007fd1275439e6d11f52ba1f994` |
| `zip/GCI_Research_Progress_Results_2026-10-22.csv` | 2986 | `836461bc7adc6095c3e1de6ccf471a13a5978ddba3f7bffb9a3852a8ab6a3a79` |
| `zip/GCI_research_report_detailed.md` | 18279 | `d7b6ef71c23b037791e0a1a2807b99b358da70243ee81736b17a977562e8198e` |
| `zip/GCI_RESEARCH_STATUS_2026-10-17_evening.md` | 8380 | `152f30bd7ee9d7f0b29a7da4f07c6ad8e4f772fdf915b3a8bdef2bab235f49d4` |
| `zip/GCI_research_summary.xlsx` | 16617 | `9dad11944e6b349f80ae75b1a1547f99f3c0de5f975bcf3b2cdbe392a8c4418f` |
| `zip/GCI_residual_ranking_5pct_v15_leaderboard_probe.csv` | 1611979 | `b008470ddd7332397a8e4856310b47e7bb38eafd3947240cc5b34b522f35a62a` |
| `zip/GCI_risk_gated_fingerprint_submission.csv` | 1613455 | `c384c3179f87feb13b8009d92023d1c1e92caeed758e6ac6ab2cc0971d755dc1` |
| `zip/gci_risk_gated_fingerprint_submission.py` | 4986 | `9ac63f1f2703dac036960e3112a0fb51bece7d3ba18b56d8976ac645f8eb7ce1` |
| `zip/GCI_submission_candidate_extmean_2pct.csv` | 1353018 | `fa4c5d07ee0a35ece70af6fce1694dbded469601aeca9a67c68ff802b0118cfa` |
| `zip/GCI_submission_catboost_product_blend.csv` | 1615463 | `c3fc40999a7fd90fc2c660e48cad2f3b138126aafdf2b4d6d1ca7053df6a9cc4` |
| `zip/GCI_submission_catboost_product_blend_reproduced.csv` | 1615490 | `6cc397b262fe4b550cf562b5902e3de8964c4f3bfb045758aebd12b98879b454` |
| `zip/GCI_submission_current_champion_0.76745 (1).csv` | 1615543 | `e8b33aa094a253a9349884cffb1e8b58a01a3184f528e7e8687621dfdeb633d3` |
| `zip/GCI_submission_current_champion_0.76745.csv` | 1615543 | `e8b33aa094a253a9349884cffb1e8b58a01a3184f528e7e8687621dfdeb633d3` |
| `zip/GCI_submission_product_blend30.csv` | 1615501 | `4a6eb7cfb6411336b72a15b7b4dfbf2d1d7a51161384e631add1c9d114e42e00` |
| `zip/GCI_submission_product_blend50.csv` | 1615306 | `f7ebd9e9565bee10294d1afb846a0a7039a33797fcd788efb831fb60df8418bc` |
| `zip/GCI_submission_qfingerprint_cl.csv` | 1614478 | `531090e0974a166d291db2027e4b19761b00c633c6348b69e0c10aa018a8401c` |
| `zip/GCI_submission_qmodels_only (1).csv` | 1615403 | `3a71990c4034a4416b936d840271e0ce2cb3686099d1cecb8de771e5f877804d` |
| `zip/GCI_submission_qmodels_only.csv` | 1615403 | `3a71990c4034a4416b936d840271e0ce2cb3686099d1cecb8de771e5f877804d` |
| `zip/GCI_submission_reproduced.csv` | 1615335 | `0c5e7f17f3a97532c3ea6d4bff062f94b6431de8141a837b7f164307e0cae1d6` |
| `zip/GCI_submission_reproduced_final.csv` | 1615405 | `8fbc351fbef1206ad5a8180e5526afc3d6f0ceea9249871a0cdfc9bf12f86691` |
| `zip/GCI_submission_reproduced_final_oct_17.csv` | 1615405 | `8fbc351fbef1206ad5a8180e5526afc3d6f0ceea9249871a0cdfc9bf12f86691` |
| `zip/GCI_submission_v14_G_B_plus_10pct_CatBoost.csv` | 1615783 | `ac43cb2e9e73c98405356c452948cb0b58da1450c494d17d4d82ea2edf7800bf` |
| `zip/GCI_submission_v15_rankspace_consistent_view.csv` | 1615628 | `6464b2144949df573dc59a9fcde42959401c6c977ed53d263e53018f21bdcdc7` |
| `zip/GCI_submission_v15_reproduced.csv` | 1615597 | `189003a9c8ebf440567453b748e615a4ae29869bfab61debb260ad38f53ec378` |
| `zip/GCI_submission_v16.csv` | 1615332 | `4c12865f2f3523cb8f965f35a08c5b6034572b276dcb1036a623295538537980` |
| `zip/GCI_submission_v17 (1).csv` | 1615454 | `078a14243f3625363cc9e084083aad2c16747897ccf066889ec84b4e23eaa88c` |
| `zip/GCI_submission_v17.csv` | 1615454 | `078a14243f3625363cc9e084083aad2c16747897ccf066889ec84b4e23eaa88c` |
| `zip/GCI_submission_v18.csv` | 1615349 | `871fe745f373ffeebff77c1eb75373a50c0e0e3fddcbbaadd7a83bc884fea3d1` |
| `zip/GCI_submission_v18_alt_direction_probe.csv` | 1615246 | `87a44764689e5e498ac218d90db18361eaa1411096e079c5025a8f2ae1d66586` |
| `zip/GCI_submission_v20_tariff_distance_probe.csv` | 1612541 | `ff62ef29af29b4833b3dd0b08e80d2880f1c26e174eace07111144c1e6858d44` |
| `zip/GCI_submission_v21_candidate.csv` | 1615466 | `9a4432e17895ff80f1fb1e933a036a878ce104ec1bebb1c4cc565c2222ff3aa7` |
| `zip/GCI_submission_v21_tariff_distance_1p5pct.csv` | 1612884 | `f2aeab18d082a763f295449e04bafbd6a956d54aaaafe847454f59f91d4f3891` |
| `zip/GCI_submission_v21_tariff_distance_2p5pct_sensitivity.csv` | 1612835 | `6b42df3c0ec8def1ba4f869fe6895f7e761719817a7667441ff88767a644af18` |
| `zip/GCI_submission_v22_tariff_bureau_combo_10pct.csv` | 1612356 | `ade1a2520d7c0109932c34dc9e084246cb2921357f5b864b6945a3cbc4356f9b` |
| `zip/gci_targeted_second_order_attack.py` | 7567 | `a99d8d156af8ca220fad84d75425725953ee59047416330745e0c066d65e7f97` |
| `zip/GCI_template_meta_local5_submission.csv` | 1615136 | `e41a97eb4f18da5e155a9a930a3daa5d11cc89487b1ef18e9e80526ad41c01e2` |
| `zip/gci_template_meta_local5_submission.py` | 6293 | `666ce937b3d9cd61d4700098c007f762391186588a7b2b9c4b92c0cf6cd7f434` |
| `zip/GCI_testlike_specialist_submission.csv` | 1615529 | `8d9afc723eb40b93293929cee624e89c4a06c38d1bdafbd8b43f820ef577e279` |
| `zip/gci_testlike_specialist_submission.py` | 6085 | `841e0db15c4c4d81b9fb5ee94b6ef2b04514204460d70def97383a607454dcd3` |
| `zip/GCI_top1_research_report.md` | 12313 | `a3d96a7d0cf8df394d5070606407c6d8534b9067a170753a422ed58a25a6416e` |
| `zip/GCI_v10_candidate_blocks_colab.ipynb` | 11181 | `1ae60770e0b27e96153381e7faa6948c87525dd4af4bf4345c69fe7d3286eafd` |
| `zip/GCI_v10_standalone_colab (1).ipynb` | 31414 | `f8920f6ff7c2079cd48ed60b3a0f203f34cf28e7e7b1f20b296b3502601b2763` |
| `zip/GCI_v10_standalone_colab.ipynb` | 31414 | `f8920f6ff7c2079cd48ed60b3a0f203f34cf28e7e7b1f20b296b3502601b2763` |
| `zip/GCI_v11_family_bagging_colab (1).ipynb` | 24165 | `fb06994cb1d9e4f1217a2ec1545dcdf1d7044cd061a8f24982deca1276888171` |
| `zip/GCI_v11_family_bagging_colab.ipynb` | 22997 | `9fba1261f6171082563f3142d4c42872d50db3219d45c64a876fd95ffc2970b9` |
| `zip/GCI_v12_product_aware_AB_colab.ipynb` | 24830 | `96f5d2071fdc465b9110bbfc82a4f82ddff1d07999437da0d52d5b950f6f9b97` |
| `zip/GCI_v13_reweight_and_ET_colab_cc.ipynb` | 32422 | `c6cf202432c99dba57ef38b8efc3c41aeddf4614f187f90d50605d1e5a0556de` |
| `zip/GCI_v14G_submission_report_2026-10-17  score 77884.md` | 4182 | `168d498411679dc622960db29d774f7ac61521f268cbb2f9490ead05826dac9a` |
| `zip/GCI_v14G_submission_report_2026-10-17.md` | 4182 | `168d498411679dc622960db29d774f7ac61521f268cbb2f9490ead05826dac9a` |
| `zip/GCI_v14_ET_deepening_colab.ipynb` | 25214 | `43121233a28b318536c8c4a7b05d9f86de73ea5de441deb1f56f4a1051ffbefc` |
| `zip/GCI_v15_consistent_views_ET_colab.ipynb` | 26611 | `98d4d87f32692040451e39386f6c8e4f20046030adb22fc0b3c284965274d999` |
| `zip/GCI_v15_rankspace_candidate_report_2026-10-17 (1).md` | 3198 | `2dc960022478c93b83b2ac7dc0816a8f9b79ae49793a19d6ca9bce9711905231` |
| `zip/GCI_v15_rankspace_candidate_report_2026-10-17.md` | 3198 | `2dc960022478c93b83b2ac7dc0816a8f9b79ae49793a19d6ca9bce9711905231` |
| `zip/GCI_v16_everything_merged.ipynb` | 46731 | `9264c76f4da88bcb23a715b881171e2751616344b3591bf7a838c26921cfce4a` |
| `zip/GCI_v16_everything_merged_o.ipynb` | 87903 | `ca77c6a6dbf6f5d1ea80b4f0f23de1f0f0c73ea7b4ac6fab937fd32ed8881306` |
| `zip/GCI_v16_regulariser_screen_colab.ipynb` | 29020 | `1ba6c09a31d696ce1480cd75bbcae3b6cb52bf4ee9e9eaef07d5a431d89ad4f2` |
| `zip/GCI_v17_catboost_E1_stack_colab.ipynb` | 29338 | `c1136dcf7eb1f32c3b2185202c087d7062095f0dba9b04529064353ad20e8b1f` |
| `zip/GCI_v17_notebook_review_and_next_direction_2026-10-20.md` | 5142 | `624692bb0492511566e93957fb440c2ac2e7c26637d152fe8fff933b06b881c7` |
| `zip/GCI_v17_optimised.ipynb` | 52657 | `f9628cb43427a3abec711bef14694848a99bf2aaa5a893e0bfe90d371f806a7b` |
| `zip/GCI_v17_optimised_O.ipynb` | 95828 | `39f931983604fe74899b61a1f116991e0e342c4509e2d7f80eeaf92117e5e6d3` |
| `zip/GCI_v18_alt_direction_package (1).zip` | 1263732 | `71364dcb107d1fb1f7168d81199115b9ef6248ed6a15cc83b50c39905e460b94` |
| `zip/GCI_v18_alt_direction_package.zip` | 1263732 | `71364dcb107d1fb1f7168d81199115b9ef6248ed6a15cc83b50c39905e460b94` |
| `zip/GCI_v18_alt_direction_probe.ipynb` | 10117 | `50558cafbdf040a7abd7bb31661e1c71ba5408ce6bf01a42dbf72cfc3206107b` |
| `zip/GCI_v18_evidence_driven (1).ipynb` | 62146 | `1e96adbca6889b8368646a0dedc195f6ff73e720e52d29fcd02a009a06aa2a9f` |
| `zip/GCI_v18_evidence_driven.ipynb` | 62146 | `1e96adbca6889b8368646a0dedc195f6ff73e720e52d29fcd02a009a06aa2a9f` |
| `zip/GCI_v18_evidence_driven_o.ipynb` | 111224 | `4c70a10af3306ad6f19a0967dff60d18e8aa3012bb395e285550df232c31b3ea` |
| `zip/GCI_v18_ExtraTrees_diversity_colab.ipynb` | 28003 | `593c00d2952c5d07c2e69c6ad64936104b395cc045d80922ddb1fc7f6cfa851a` |
| `zip/GCI_v19_NN_entity_embeddings_colab.ipynb` | 28832 | `daa379c8cb32eedbe5b434615a7be14e6599a0ebbe73aa0ff731ed978b3d7832` |
| `zip/GCI_v19_NN_entity_embeddings_colab_0.ipynb` | 40241 | `a0b3fad94e7e94ce72b9895e24d71a14351e27f2f7656d823ee5d06efcd13688` |
| `zip/GCI_v19_research_candidate.ipynb` | 88007 | `c17ecde55eec91dd479cee235bd96708f5a37661a12101e079f1fbabfd53fe67` |
| `zip/GCI_v19_research_rebuilt.ipynb` | 74471 | `ede91f2a3ea122d4b525e62320b2c9137487d2cfabe34db310af8f7feb415131` |
| `zip/GCI_v20_evidence_driven_candidate.ipynb` | 94957 | `928c5be0c6a59c2f5f517d77a9ecb60f1ddda6c0690de18ba12b5db1f376cafe` |
| `zip/GCI_v20_research_report0__7789.md` | 2695 | `4d1365407f492403e10a67045c9f090355b28e16eeffa5367aaee8b4a3da7c0f` |
| `zip/GCI_v21_gated_research_candidate.ipynb` | 78079 | `8079830c4053a1195e630a5a34c3628a1738d6708d699e4bd3c7162fe4ef9ddf` |
| `zip/GCI_v21_gated_research_candidate_o.ipynb` | 99217 | `1da0a48cc190a12396174b5648eff96dcefc421ff9737306b166c5f7d972fd66` |
| `zip/GCI_v21_research_report.md` | 3503 | `43cfb81e322084eeb3f04432546e2614efbb5d481b1828864800aaffb4144de3` |
| `zip/GCI_v22_research_expansion_gated.ipynb` | 80894 | `0cc8bda2e9a2120d0644e00962ba279fddcab4ed319ad8394eacc7f8fb9bb774` |
| `zip/GCI_v22_tariff_bureau_combo_research_package.zip` | 3511151 | `1a8b4fadd6433c6f3c4b220d181565c7a04367405f558e4feac90a30dd23589d` |
| `zip/GCI_v24_evidence_lab.ipynb` | 71084 | `564c639cb6f1305bf2dffde056a07163571a1dbbff35688e150f24d1b495d2bb` |
| `zip/GCI_v24_feature_family_results.zip` | 16910298 | `01ee00a1be62a43110e623d446d2517abdd06e92d2c95f15bbe2d8181563c714` |
| `zip/GCI_v25_block_tester.ipynb` | 60400 | `e0f5139aa07867a9f429cb92764fc6262e721faa91c3913056ee6f4d12e08109` |
| `zip/GCI_v26_optimized_five_research (1).zip` | 15594252 | `272ff8ee9c9e3cc681f138dd949e4d34f0378ddf316b954e41564671da338384` |
| `zip/GCI_v26_optimized_five_research.zip` | 15594252 | `272ff8ee9c9e3cc681f138dd949e4d34f0378ddf316b954e41564671da338384` |
| `zip/GCI_v29_five_data_mechanisms_research.zip` | 12861787 | `5ebafa872105cd03f6055970a837c45de9f05cede60ba2c59f6ad9fba18a7f85` |
| `zip/GCI_v33_product_conditional_experiment (1).py` | 7951 | `d383cd0952340fea6f64e6d1a0958faa1eb5d3583c1129ffeb58e4991dfbc240` |
| `zip/GCI_v33_product_conditional_experiment.py` | 7951 | `d383cd0952340fea6f64e6d1a0958faa1eb5d3583c1129ffeb58e4991dfbc240` |
| `zip/GCI_v33_product_conditional_model_only.csv` | 1668177 | `99d6f4554ba9ac211e2d4d5ea1007468f005240dfae79bc5d2a8f31f83442eae` |
| `zip/GCI_v3_blend_colab.ipynb` | 17683 | `2a18ca24b476333769154effa988bcdebaa14ce5704adf9dab2a2eb713aeb871` |
| `zip/GCI_v4_diversity_colab.ipynb` | 21244 | `60c0aa890ea352d3a49db904b2c61cced84ab8f6c65626124f5c4b5b437cb8be` |
| `zip/GCI_v4_results_summary.xlsx` | 16171 | `bce11b1c380164de3aeb49c2564cbd830aae3c78ae52352162a21be10c5fa6e9` |
| `zip/GCI_v5_champion95_CAG5_gate97_submission.csv` | 1614367 | `5b0674a257dd0197b401626d8e0577756b9948f25841a5ea2609dee5c4452abb` |
| `zip/GCI_v5_dropPF4_colab.ipynb` | 17948 | `b2cafab6159fae44739d13866eab09d89b06321c5548340c162097fd620561b3` |
| `zip/GCI_v6_champion93_CADOWN10_gate93_submission.csv` | 1613989 | `f66ae16019240ae73c2c78405e853fc10dc466e63be650c42ea1f73fe63d2852` |
| `zip/GCI_v6_final_blend_cell.py` | 3215 | `69cea936dd272e8604ee3f15edf048c1f02f7bc254e263a83feaf6c722bd9c05` |
| `zip/GCI_v6_residual_features_colab.ipynb` | 21482 | `6e97b0acd26ad59bebd2fad9fe4bc9dcec321a48e2f9006fbc3b097f7e85346c` |
| `zip/GCI_v7_champion67_xgbrich33.csv` | 1615671 | `31d5be5deafa54f8cb5f065bb78de0a39aaf795a1d545e34e6b1a0d956bc9ec7` |
| `zip/GCI_v7_extra_trees_colab.ipynb` | 24376 | `cc56101fb68fa8fcacaf3a8cccfcb5ba84140c11eb0a99daf20bd40ae7647fca` |
| `zip/GCI_v7_screening_results (1).csv` | 1826 | `c06aa673a63730d536160615e62b2fc0e30be8b9d8a96ba49ded87c541612799` |
| `zip/GCI_v7_screening_results.csv` | 1826 | `c06aa673a63730d536160615e62b2fc0e30be8b9d8a96ba49ded87c541612799` |
| `zip/GCI_v8_conditional_XGB_disagreement30_w20.csv` | 1615405 | `1dfdc4b01a74ec55467f052185988471b374b02a08af6de1a7b499a9741c0b8e` |
| `zip/GCI_v8_pairwise_ranker_colab.ipynb` | 20865 | `67821225f9e6a1520edbd667bd40f275ac3ab9c9c7690563dd37337b8f3b2674` |
| `zip/GCI_v9_all_experiments_colab.ipynb` | 28764 | `e4556471b8de1490a6c6171517e68cabbd858cf9e3a5a2eefb9f65d0be5c269e` |
| `zip/GCI_v9_tabicl_context_colab.ipynb` | 12645 | `52a7839a29f9bc25cfda36e3079e59fd3be7bdc6e57a2fdd3beac65f4e0649d2` |
| `zip/GCI_v9_tabicl_standalone_colab.ipynb` | 33640 | `1494aba03e23d8ff465f3dc14507bb0f255cb79a6b3089bc274e5a3b50c7cd79` |
| `zip/hard_error_oof_results.csv` | 2911 | `93e2b5545f901e15c5cfdf92d7ca87bcc45b97155bd09032c85e59a2d8a78af7` |
| `zip/interim_feature_research_evidence.zip` | 28301 | `157000dfa8e4a8466ee2e50a2279065ae3129f02c8ad515fc4d3feb8f91f9b3b` |
| `zip/INTERIM_FINDINGS_REPORT.md` | 14064 | `d7878e561904d73b0a724cb56cc0de86172d6f3e20403e1a179b900054cb0193` |
| `zip/lgb_a.csv` | 1667568 | `670e0fd30551fb8aec676c3a689674c8f3c25a6a22b2a2306dfd5fc8438c32bc` |
| `zip/marginal_drift.csv` | 2503 | `eba40f5344921d344bbc742382786c3aeaff619f22b8d49187e0d4635fee331d` |
| `zip/meta.json` | 16012 | `cdb08890d4127492d89b4864fe1ef093d1c07e8d64747cb25d480a7a546e4100` |
| `zip/mlp_full_candidate.csv` | 1130789 | `8fb7faef47716a81afb72cfbebc114588ec3f9005d1677a4133b8acf2d0018e0` |
| `zip/model_benchmark.csv` | 946 | `a8255a2b5b21ad0cef6bf9c95fe499779acaf9de61343b61d7073817c345c5af` |
| `zip/NN_test_summary (1).xlsx` | 11089 | `bb7771704b72895810fbada63c85e12fea716e390bd92e4c952167f652e1b77c` |
| `zip/NN_test_summary.xlsx` | 11089 | `bb7771704b72895810fbada63c85e12fea716e390bd92e4c952167f652e1b77c` |
| `zip/oof_lgb_xgb.csv` | 7232886 | `d1503ca04b43a71019b04d5be72d006e22d1b0e9beb1467ea751ef8a7c151f31` |
| `zip/past reseach.txt` | 54105 | `8ddf094fdcc9cd46ff5e351e190c2c4f6ba3dceb5c9de72acc6e27b87fd50929` |
| `zip/Pasted markdown(20261008-171218).md` | 121264 | `9c053e8739400bf9c152943c715cdfbff28029f9fc62e63c44edfa4dd5998999` |
| `zip/peer_ablation_results.csv` | 990 | `196c40b05055e673a5cf273ead6465896e80724c7ea4b0aa5f2909ac3a1ca9b4` |
| `zip/peer_ablation_summary.csv` | 1087 | `43372c8f8c5d71389c0cc012a429e3952837fde4c67b1c1b8e829631074adb5b` |
| `zip/phase4_domain_view_rank15.csv` | 1612011 | `ec5a19e2a53a8ad89d231b48e5bdc4d6ee2c65222e1436a97cef22ac594b34f3` |
| `zip/q4_interaction_blend_screen.csv` | 882 | `0c38e2b73ed7f63eb763f396d17b6af8c7119c4a73f950a4c8babcd7c6b25b09` |
| `zip/RANKBLEND_10.csv` | 1615471 | `44d95183ec130627eefc52c76880ccc4b7ea2878cb7126bb2b9bd1ec666ca47c` |
| `zip/RANKBLEND_15.csv` | 1615398 | `1e9069313be61d81784f2a8a4e0e8176211523c6dd4c225313fe445cb1914351` |
| `zip/RANKBLEND_5.csv` | 1615562 | `009c27b95bc3cf935d8ad25d0ec826dd459f9b4542c85a15db4accd887966d8d` |
| `zip/rate_residual_3holdout_confirmation.csv` | 1771 | `5688da1df4ca4bb8e99b8a75dac548dbabb156f72d5552469e7754e18931f620` |
| `zip/rate_shape_strong_anchor_results.csv` | 1057 | `8459dc77e274c993ba12ea587dd9ddb7fa63439f7ad95e6b987ba435b49fa16b` |
| `zip/rc_xgb_beta125_prob.csv` | 1591159 | `d924ca010ae16619ed35d7085b17dfa91fdcbc06c07fb35793cf5df1a69f4877` |
| `zip/README.md` | 676 | `bcf018e9676e69a0e4543dfc055ab5d3959549621b0754f2be749751b91f8b9c` |
| `zip/README_FINAL (1).md` | 9373 | `4804e2ea0fa7a5cd0d48e38e5890342e22abab23e07aa8e013a5ecac8558b0ba` |
| `zip/README_FINAL.md` | 9373 | `4804e2ea0fa7a5cd0d48e38e5890342e22abab23e07aa8e013a5ecac8558b0ba` |
| `zip/README_HONORS_FINAL.md` | 7761 | `dc07149af092623e5d20a807910c7e7602ed8dfd88ed8085d4a09773a62ea94b` |
| `zip/README_HONORS_FINAL_2.md` | 9055 | `5983e564744c27d64c06082bbc6fdfde38e91eb4794f8ace93b0c652b285c3c3` |
| `zip/README_HONORS_V15.md` | 7714 | `57a935aafcd4b785b15870f907c61ee17b88deabfb4ff69e8602e6adc87d0f25` |
| `zip/REPORT.md` | 4068 | `bc450239e1deddf51e2c68396a186b9396f4efe41b56af44547eb0fd8b454fa0` |
| `zip/reseach_15_10.txt` | 64203 | `8afd5b4c0b363370cf91e71e509d436925cc36f827f53c5125977118ca15c38d` |
| `zip/reseach_@.md` | 23831 | `44690b3c1a4166772441926e701bcff65fa681366beb328e22ccf0bef6d98718` |
| `zip/research.md` | 17693 | `9a0b7094d2437fe1eda7f38bf8ade62ef2f81467aa7a2e06bfb8fb3da5801ad0` |
| `zip/research2.md` | 23831 | `44690b3c1a4166772441926e701bcff65fa681366beb328e22ccf0bef6d98718` |
| `zip/research_10_15 pm.md` | 21645 | `812aa53fcec739a4a42ac8f8e4e3f3b2fb90e889c32d3da5257e5691fa17d242` |
| `zip/RESEARCH_REPORT.md` | 4590 | `847e60872f3d99d9956c58af53111d86cc27bf41ef6c2ff67426adef0778d990` |
| `zip/research_section13_weakness_analysis.md` | 4544 | `a3c4e314e34e2c6b9ee6849616011788685079e9b05d02893a114a16a8ca1512` |
| `zip/residual_rate_geometry_screen.csv` | 3271 | `ebb9a772a7621ea92edd69c7e2da05a86b277c7b5bfd9573e8062b59cc7ba6c9` |
| `zip/results (1).csv` | 3943 | `e65a7d67e42f91e8e2d2ec944103dc273c12b7e9433099ca2af208c2408ede83` |
| `zip/results.csv` | 380 | `eea1f6c9e3af961356df1a30fd80034494a44292dadcbea0aa9d181a2fc76108` |
| `zip/Right_Customer_Right_Offer_Right_Time.ipynb` | 70070 | `9cbd3cc804971f80ab790c976e93869e8553fecdce232ed2b45b3078f3162617` |
| `zip/Right_Customer_Right_Offer_Right_Time_v10.ipynb` | 125470 | `a0bb9f5595fc042902e349390ecc6759a4c75524b9c6b76eb682d5b11b5641f8` |
| `zip/Right_Customer_Right_Offer_Right_Time_v11.ipynb` | 128582 | `880268d8975747272e492bcd2de1e9307ca3eaac871efaf3483c19606c28cab5` |
| `zip/Right_Customer_Right_Offer_Right_Time_v12.ipynb` | 138324 | `7b5956dabebaa74b9fd3106b1c71a0e22679167c0ec9b7cffbdc21edc148a4e6` |
| `zip/Right_Customer_Right_Offer_Right_Time_v13_Optimized.ipynb` | 140893 | `2dd02ee096126143b868598845b7e09a9c3d50b124d7400c0354b3392f9391a2` |
| `zip/Right_Customer_Right_Offer_Right_Time_v2.ipynb` | 88239 | `d8ee514f7e268f83385bcb2b16995e5c8831e8c9c3d5bf6d5aafc96078d59fdf` |
| `zip/Right_Customer_Right_Offer_Right_Time_v3.ipynb` | 94162 | `a8e57729431420fbd599149b51da78e6905566c5c9432d5d526c024e3fff3922` |
| `zip/Right_Customer_Right_Offer_Right_Time_v4.ipynb` | 97390 | `2ad5cb21aba6c8a2ce3bc5524a68ca9b435b5704d10009a56e7b0e24a90d2862` |
| `zip/Right_Customer_Right_Offer_Right_Time_v5.ipynb` | 97722 | `78a1079ce699474bb10339493229101f774861bee95bf4a0186392b68fcd0a45` |
| `zip/Right_Customer_Right_Offer_Right_Time_v6.ipynb` | 99266 | `31c12fd3eacee2e6b7626e623a4f39bf1010c1fb24bc32c71dc63998dc5fe4a1` |
| `zip/Right_Customer_Right_Offer_Right_Time_v7.ipynb` | 100606 | `d8c792177c06a28f236d00adc6f274ad75bba7959d3fd7135b8b8af4c07abd00` |
| `zip/Right_Customer_Right_Offer_Right_Time_v9.ipynb` | 115341 | `499d51bd8a4c6a2c720845521f1da86ad2a3c923c51ce42ad838170796861f9f` |
| `zip/ROUND1_CA_EXT2_RESID_submission.csv` | 1614213 | `39c682199a11762d4af99b606ded4ec86311b68a0d4a2c0f8a3690db91021eb7` |
| `zip/Session5-20261008T161139Z-1-001.zip` | 178 | `f0e006c00006c61c0f1ac4d18d68cf89b6954b7e39ee2a9f82d955fbcfa674bf` |
| `zip/solution_v5_gci_research.py` | 140761 | `4f0f1e9208f99ef2350af05f9c3da98eaae922de729295f2e0c410186924ae74` |
| `zip/Speech-and-Text-to-Indian-Sign-Language-with-a-3D-Avatar.pptx` | 1188627 | `102e26e304a7c6937de38c9d8cd33929c66042396c7e0a5162650eb7771ad71a` |
| `zip/submission (1).csv` | 1614674 | `1b217a1b30127d6430832e89d9e86f63e84eaee254286b2f227aa3158c35fac0` |
| `zip/submission_01_rank_equal.csv` | 1615615 | `f5afbc4e1dc06709fcedae57e2e8a0dd3e03e79ce93f4524a257a2060dea7db6` |
| `zip/submission_03_best_single.csv` | 1667667 | `1d07a52fb6aa8aefd5f92b4482fc6c565ada87c9ab7f0e2bf8f13573edea3bda` |
| `zip/submission_04_stack_LR.csv` | 1667435 | `d1a30fee1eb5de192b22c469f326a766286687735011e4900d6eeb627f10196d` |
| `zip/submission_10_blend_10.csv` | 1615287 | `1babc45d13edb04822824b8d5e3d6fa957acda12c88f00e1ff7369b9f633da35` |
| `zip/submission_10_blend_15.csv` | 1615695 | `00fd95f467d8d61414b7546597ec080e5a7d3e36bcafb71ba5c83ce5df1bd389` |
| `zip/submission_10_blend_3.csv` | 1615740 | `94f21664a37f00a162f6c3db3933b4a1600cf583813f444abfdb05648743bfe5` |
| `zip/submission_10_blend_5.csv` | 1615619 | `264bd377510dab4143637977b3e190f670e7c173761917334ea2d527de2c9a1b` |
| `zip/submission_10_blend_7.csv` | 1615476 | `8143e244475952ede91eff7fda4afd20a1a4def5d163cda2b3f29ddcac196ed5` |
| `zip/submission_10_xgb_onehot_no_extagg.csv` | 1130617 | `3a7d90f696e6a034ce37f85e511bc1f8a6a8265e5c0d897ca71a6fd8e6aa9ebc` |
| `zip/submission_13_old_anchor_80_file12_20.csv` | 1615365 | `50faadf3e53c4f2b003dc3bf826bf46460a46042d7be5b150ef7253670c3bbb7` |
| `zip/submission_13_old_anchor_85_file12_15.csv` | 1615672 | `331ddda62ae7a06a6ba13260b5ce68d685b5981578a235aa1e52796d62cd8162` |
| `zip/submission_13_old_anchor_90_file12_10.csv` | 1615479 | `098acf6943d75384e45a399e57127c502a455f140bc559eb36f34eb2d6eda0ea` |
| `zip/submission_13_old_anchor_95_file12_5.csv` | 1615602 | `eef8a3d9d2fadf7a96826603a8e0f55dab19deb3f9e7c72e81444aa8c6601454` |
| `zip/submission_13_tri_85_10_5.csv` | 1615546 | `551bb873436ddd3fda08098397c1f61807943d1eeb01a660979fabc0e5f417e4` |
| `zip/submission_archived_xgb_0.77308_candidate.csv` | 1130421 | `09e1e4bc5fbb80067a6ac3b9ad03570ca4c258ed667b40698a396b0324410fd1` |
| `zip/submission_BEAT_CLAUDE_cat7_lgb3.csv` | 1615585 | `b57e67b8b000d86da7978fe7423e0544eee7fe58c207312058ae95451b2bec50` |
| `zip/submission_cat.csv` | 1666325 | `e4f128b3b63fcf63704ee594b8c3ec3b88327826cb4e1ac8cc00d6e0a000f334` |
| `zip/submission_FINAL_76745_downsample7.csv` | 1615560 | `e21ea8b353606fe4bd5089efe751877108c6e2ec8d3261d621421e52f2392b65` |
| `zip/submission_FINAL_BEAT_CLAUDE_maxbin25_cat5.csv` | 1615614 | `5a6821b12df38fe9cdf140c1c62bd5378b70e8e38d1faae8402db5176382cc07` |
| `zip/submission_lgb.csv` | 1667211 | `ad62d971dc9112658096837261ccb65f3171da334a40517ae748355deae09cb1` |
| `zip/submission_log_credit_term.csv` | 1130292 | `8caf037244b795b9e4bdb2cc08ad256e51e5daa012225644f06796111af6673c` |
| `zip/submission_NEXT_BEAT_CLAUDE_oblique18.csv` | 1615577 | `5aadada0727064a21e53be133a81fbd09b4fd9bd410be757c92fe57db7adffab` |
| `zip/submission_rf.csv` | 1620006 | `9fba83f468d296ab99858fa306103bf4870a86eb4185bb0704075c82d21efd6c` |
| `zip/submission_selected_lgb.csv` | 1666970 | `c3f2ac211d8785df6ce717694cad0766012d98119ba6f9c651295f1fe4ecc671` |
| `zip/submission_targeted_second_order_75_25.csv` | 1615618 | `2f5d7c801387f4a22920c8970345fac002cdc4e0720795e87ab61e080da1ffc3` |
| `zip/submission_te_xgb_97_champ_3.csv` | 1662583 | `c2c3dfe440c65a7d5e6a8017ad83893483a2f52481604c359a039afec0e39b9c` |
| `zip/submission_te_xgb_seedavg.csv` | 1129937 | `d9fdf72fd003a8cc9abe1041d916d150d67006ab35b844cf7623b1dd18216fec` |
| `zip/submission_xgb(1).csv` | 1130421 | `09e1e4bc5fbb80067a6ac3b9ad03570ca4c258ed667b40698a396b0324410fd1` |
| `zip/submission_xgb.csv` | 1130421 | `09e1e4bc5fbb80067a6ac3b9ad03570ca4c258ed667b40698a396b0324410fd1` |
| `zip/submission__regime5.csv` | 1615671 | `b8f2f791033617ec367930780f7d84b327de30ab8cfe3311422d0552c1de314b` |
| `zip/sub_lgb_rich_full.csv` | 1666867 | `4048526ad37ce9e58c5a0272b6597b575ab35afb229ae4881bdcb4ecb30b54f9` |
| `zip/sub_xgb_rich_seedavg.csv` | 1130294 | `87f10ff4b23a5deb66430b2c6c665755af4cbb2fc2c84b3b2bcde34272366aa8` |
| `zip/target_encoding_screen.csv` | 5300 | `bb15cd41ca8adfe07607aa90078c39cde4f54dd34e0ae8cf3cc6e072e5f20d71` |
| `zip/Telecom_Churn_Final_Research_Package.zip` | 1814302 | `41da51eee601dfadf84a88cf892f7a370fad4e2c5a2d13486050bd7d049bf044` |
| `zip/telecom_churn_research_package.zip` | 31006 | `12ef65e16f06e9eb463b49128276fb69b52f6f075383a498fc42d0a52418e1e8` |
| `zip/telecom_churn_research_round3.zip` | 38417 | `6b777a63d869cc3e5649e10b3dbb4090f9011b84441d60fc90d0e4e3735d2142` |
| `zip/TOP10_COMBINED.csv` | 1665470 | `ec61f60d03fdd3b2a2113589d3c42ea4d2a6b33fdda0269046d8062966d95054` |
| `zip/V10_ANALYSIS_TO_DATE_REPORT.md` | 16897 | `346cc617cd6b6483a053f73ee637e57b7fe6376bb02013862eaca5d495b9b49b` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip` | 353249 | `362d90a486bc5d3d12863c40007ddb4e15daeb1deaf5aa99dd00cb9fe2359e5d` |
| `zip/v11_A_v6B_plus_family_bagged (1).csv` | 1615133 | `4b50943bd22fdf2ae41318a230f462de593491e1408b47d6b1ead8b0be3937f4` |
| `zip/v11_A_v6B_plus_family_bagged (2).csv` | 1615133 | `4b50943bd22fdf2ae41318a230f462de593491e1408b47d6b1ead8b0be3937f4` |
| `zip/v11_A_v6B_plus_family_bagged.csv` | 1615133 | `4b50943bd22fdf2ae41318a230f462de593491e1408b47d6b1ead8b0be3937f4` |
| `zip/v2_A_lgb_product_k.csv` | 1615383 | `b96843b24f61397af65f300d1675872fcdec9f717efb8132cecd51c87e230eab` |
| `zip/v2_D_lgb50_prevrun50.csv` | 1615540 | `928daef0caa6d0e9de6d55a7f22482b2672e87691f5fc031c963d4abb4e1407f` |
| `zip/v3_A_greedy_oof_weights (1).csv` | 1615335 | `0c5e7f17f3a97532c3ea6d4bff062f94b6431de8141a837b7f164307e0cae1d6` |
| `zip/v3_A_greedy_oof_weights.csv` | 1615335 | `0c5e7f17f3a97532c3ea6d4bff062f94b6431de8141a837b7f164307e0cae1d6` |
| `zip/v3_B_plain_average_all (1).csv` | 1615518 | `34d067a696458801eb25767ec588f93511d6e2f46444bb559b9f809517c0f81e` |
| `zip/v3_B_plain_average_all.csv` | 1615518 | `34d067a696458801eb25767ec588f93511d6e2f46444bb559b9f809517c0f81e` |
| `zip/v3_C_average_lgb_members_only (1).csv` | 1615518 | `34d067a696458801eb25767ec588f93511d6e2f46444bb559b9f809517c0f81e` |
| `zip/v3_C_average_lgb_members_only.csv` | 1615518 | `34d067a696458801eb25767ec588f93511d6e2f46444bb559b9f809517c0f81e` |
| `zip/v3_submission.csv` | 1613768 | `bcd763c85f49ca9b5abf941c416d5a0ab3625151ee428c0f6615570599efb9e6` |
| `zip/v4_A_greedy_all_members.csv` | 1615193 | `e2196cbbfb813ea4161ce95956f3aa096d77143272b4e79366a441533efcd8f1` |
| `zip/v4_B_greedy_old_members_only.csv` | 1615184 | `0806f456154861b6868287c4946f9100feeb338c841b8dc09516a60319432a6b` |
| `zip/v5_A_new_members_same_weights (1).csv` | 1615314 | `39a1d0a7e154f2ec31470faa2e4f9bf6d8981a5b35e2a3a98891fd4254886848` |
| `zip/v5_A_new_members_same_weights.csv` | 1615314 | `39a1d0a7e154f2ec31470faa2e4f9bf6d8981a5b35e2a3a98891fd4254886848` |
| `zip/v5_B_mix_old_and_new (1).csv` | 1615550 | `d481c039fb6ecdf804111bcbb26f79b0f0a31ccc70182906af516c79425eafb3` |
| `zip/v5_B_mix_old_and_new.csv` | 1615550 | `d481c039fb6ecdf804111bcbb26f79b0f0a31ccc70182906af516c79425eafb3` |
| `zip/v6_A_new_members_same_weights.csv` | 1615501 | `6a92117ac5e871bf318acbb3588e8a3403475a81717aa4a3eca21ddea223ea3b` |
| `zip/v6_A_new_members_same_weights_modified.csv` | 1615501 | `6a92117ac5e871bf318acbb3588e8a3403475a81717aa4a3eca21ddea223ea3b` |
| `zip/v6_B_mix_old_and_new (1).csv` | 1615314 | `bf2d41f17b4656c7e41d17e18462c1dbcebcc8010c7fa171923ca9cafe98b688` |
| `zip/v6_B_mix_old_and_new (2).csv` | 1615314 | `bf2d41f17b4656c7e41d17e18462c1dbcebcc8010c7fa171923ca9cafe98b688` |
| `zip/v6_B_mix_old_and_new.csv` | 1615314 | `bf2d41f17b4656c7e41d17e18462c1dbcebcc8010c7fa171923ca9cafe98b688` |
| `zip/v6_B_mix_old_and_new_modified.csv` | 1615314 | `bf2d41f17b4656c7e41d17e18462c1dbcebcc8010c7fa171923ca9cafe98b688` |
| `zip/v7_A_et_members_same_weights.csv` | 1615540 | `619d636c627693a6db61c565af0e5d70940b7562fbe1609176e4aa017e0d2029` |
| `zip/v7_B_mix_v6_and_et.csv` | 1615719 | `d5e3055b40f8af06fea58570b4dad2ec027bbd8acb6b9cee4f868b61ce049e5b` |
| `zip/v8_A_v6B_plus_ranker.csv` | 1615266 | `63d5956382d67fa74688f576d0c13b4b6e5d6e71c608884f913c8f5cd4df0490` |
| `zip/WhatsApp Video 2026-10-11 at 5.09.33 PM.mp4` | 179745841 | `dd262c1b008f6fc757d24d120b762a7d34cc42aa53c30110ec6eb9660cce3007` |
| `zip/WHY_RESIDUAL_SUBMISSION_DROPPED.md` | 4563 | `728bbb83c5f3d3909c700ca33a616ce4a6d546bbd1fb1b40bac6e6dcc9ea8ede` |
| `zip/xgb_scale2_seedavg.csv` | 1114736 | `6cdf24024585bcef16539385985d7399610592aac7f5a310d197425c226632fc` |

### Nested archive members (one representative for each identical nested ZIP)

| Member path | Bytes | SHA-256 |
|---|---:|---|
| `zip/Churn_Optimisation_Decision_Package.zip!/Churn_Risk_and_Right_Offer_Time_Decision_Report.md` | 4013 | `63775e4c70c0064cbfd6c6d4d348cd0d4b0673d42d7b89690cf05d3fa8220f36` |
| `zip/Churn_Optimisation_Decision_Package.zip!/Churn_Risk_Ranking_and_Segmentation.csv` | 839 | `860493dffe284cc412e885da3dcc9fd49e23894e197ae4bfff597d144fbfd787` |
| `zip/Churn_Optimisation_Decision_Package.zip!/Churn_Threshold_Tradeoff_Analysis.csv` | 2343 | `00243a81fee5af1f8d6d040bf1dee1026a667f6443f244cf8683606d718e2438` |
| `zip/Churn_Optimisation_Decision_Package.zip!/Deep_Experiment3_Fold_Results.csv` | 695 | `36cd97bb23924a5d88f56c5ddab953684776b7bdb97ce44ba8a91e88a1d40258` |
| `zip/Churn_Optimisation_Decision_Package.zip!/Deep_Experiment3_Blend_Results.csv` | 712 | `064de3d03c4aa67dbe153f58e4053b8cc2d2747006bea78a0f5b4ffc4d401ebb` |
| `zip/Churn_Optimisation_Decision_Package.zip!/Deep_Experiment3_Paired_Bootstrap.csv` | 271 | `9e5f18c04e0a4cb3732b9521225455243e59fb17a27d17272834c507307330ad` |
| `zip/Churn_Optimisation_Decision_Package.zip!/Deep_Experiment3_OOF_Predictions.csv` | 4030816 | `45caacf9ff2a32dbf9655421eb6a4c730e4dc2890bd50dd24781fffcd54e0831` |
| `zip/Churn_Optimisation_Decision_Package.zip!/Telecom_Churn_Final_Research_Experiment.ipynb` | 12026 | `0081f3d21497b5e8127d821939e8f839ca74ae6df5d1d944853f47a845c8d785` |
| `zip/CompanyA_Final_Submission_Package.zip!/CompanyA_FinalSubmission.pdf` | 839542 | `75d961ad455b8c3304629f0d9b48a68f2d11edfe4b8827fc08f9db13a3820566` |
| `zip/CompanyA_Final_Submission_Package.zip!/CompanyA_FinalSubmission.pptx` | 335317 | `fb1de4ab7b0330c3fa0b8ac04489b3663f41b843d93084fb9d1bc3d9699030e3` |
| `zip/CompanyA_Final_Submission_Package.zip!/CompanyA_References_for_Omnicampus.txt` | 1316 | `5153e7aa37324d8414da6e9105c1d80e3d09fff9f3fdba49c8f4f5c81be07e72` |
| `zip/CompanyA_Final_Submission_Package.zip!/CompanyA_Reproducible_Analysis.ipynb` | 29588 | `299003bdb1ab58dfafec3fa5efdbe4c70a2c7020a8c282428b3f67c454c4187a` |
| `zip/CompanyA_Final_Submission_Package.zip!/README_SUBMISSION.md` | 1680 | `882fa9187ed5fdc0ef805ee3db6a709333cfc820ad48778238c208b30dcb7e3c` |
| `zip/CompanyA_Final_Submission_Package.zip!/RESEARCH_AND_LIMITATIONS.md` | 2405 | `f8d4cc012e88f4457a790ecae755957c054e9deba0fc721ea649f9427314d842` |
| `zip/CompanyA_Final_Submission_Package.zip!/assets/domain_signals.png` | 87203 | `3e196f08b3fc63fbcb497e4f5cd1657448cf7fefaf03f9cf492e495baf43fb7f` |
| `zip/CompanyA_Final_Submission_Package.zip!/assets/finance_sensitivity.png` | 109967 | `c30eeb49faf6db1978810ccdc5ecd10677cc30d894ba3959475e6f2671548efa` |
| `zip/CompanyA_Final_Submission_Package.zip!/assets/model_comparison.png` | 69576 | `6de030ca88cc11bd27168dd9e0949dd67f1557a36ca9b3be3277ead95cbe84c9` |
| `zip/CompanyA_Final_Submission_Package.zip!/assets/risk_deciles.png` | 74884 | `f6239a5b8c2464094e9a0fc90521d20b8f967cbaffc6bf9b5096b5dd6d7b3f86` |
| `zip/CompanyA_Final_Submission_Package.zip!/assets/tenure_churn.png` | 79822 | `1500eb994898a6107ab2d0cd36e2867bc8ae6e8b44765cb1692800db59a82684` |
| `zip/CompanyA_Final_Submission_Package.zip!/model_outputs/holdout_model_comparison.csv` | 384 | `810f27c11a0da7c10208fbd63e69b2bb869224ed01a718a32c2ba638c7d5154b` |
| `zip/CompanyA_Final_Submission_Package.zip!/model_outputs/risk_decile_churn.png` | 43773 | `81de88ae0694b8c4305960d3be81c853f2989fbf6b8ef9fd30086d801baec3e7` |
| `zip/CompanyA_Final_Submission_Package.zip!/model_outputs/risk_deciles.csv` | 838 | `076be49586672c98ddda23a02ad94e2cd27382434b64b7bbd3072fec029332b2` |
| `zip/CompanyA_Final_Submission_Package.zip!/model_outputs/tenure_churn_curve.png` | 75875 | `221b680178425dc3618165f78b52817415b7bafb8fca3331442547fc8da92fb5` |
| `zip/CompanyA_Final_Submission_Package.zip!/results/financial_sensitivity.csv` | 1809 | `15a774b42f125445ea14dabf4520050d370a77f20339c211680e2518a7367d97` |
| `zip/CompanyA_Final_Submission_Package.zip!/results/notebook_code_cell_check.json` | 1195 | `9e3d00adfcbae3ccdbdfc8f4a74ba4a02c42f8fff89422c1cc0d7baec7b57e81` |
| `zip/CompanyA_Final_Submission_Package.zip!/results/recreated_submission_stats.json` | 1457 | `14abbfec2732e993caf2f6eab4108e77d55b63809a18bf943f7df5dc0f97639b` |
| `zip/CompanyA_Final_Submission_Package.zip!/results/reproduced_model_comparison.csv` | 398 | `ac728f183d2aaae7c9836ef4b976552d136f511e30793791d163f767ef797473` |
| `zip/CompanyA_Final_Submission_Package.zip!/results/risk_deciles.csv` | 786 | `619c0bf666e9730e99522ea822fdd7be285c130bffe973b6c71be42524fb9821` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/CompanyA_FinalSubmission.pdf` | 839542 | `75d961ad455b8c3304629f0d9b48a68f2d11edfe4b8827fc08f9db13a3820566` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/CompanyA_FinalSubmission.pptx` | 335317 | `fb1de4ab7b0330c3fa0b8ac04489b3663f41b843d93084fb9d1bc3d9699030e3` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/CompanyA_References_for_Omnicampus.txt` | 1316 | `5153e7aa37324d8414da6e9105c1d80e3d09fff9f3fdba49c8f4f5c81be07e72` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/CompanyA_Reproducible_Analysis.ipynb` | 29588 | `299003bdb1ab58dfafec3fa5efdbe4c70a2c7020a8c282428b3f67c454c4187a` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/README_SUBMISSION.md` | 1680 | `882fa9187ed5fdc0ef805ee3db6a709333cfc820ad48778238c208b30dcb7e3c` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/RESEARCH_AND_LIMITATIONS.md` | 2405 | `f8d4cc012e88f4457a790ecae755957c054e9deba0fc721ea649f9427314d842` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/assets/domain_signals.png` | 87203 | `3e196f08b3fc63fbcb497e4f5cd1657448cf7fefaf03f9cf492e495baf43fb7f` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/assets/finance_sensitivity.png` | 109967 | `c30eeb49faf6db1978810ccdc5ecd10677cc30d894ba3959475e6f2671548efa` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/assets/model_comparison.png` | 69576 | `6de030ca88cc11bd27168dd9e0949dd67f1557a36ca9b3be3277ead95cbe84c9` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/assets/risk_deciles.png` | 74884 | `f6239a5b8c2464094e9a0fc90521d20b8f967cbaffc6bf9b5096b5dd6d7b3f86` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/assets/tenure_churn.png` | 79822 | `1500eb994898a6107ab2d0cd36e2867bc8ae6e8b44765cb1692800db59a82684` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/model_outputs/holdout_model_comparison.csv` | 384 | `810f27c11a0da7c10208fbd63e69b2bb869224ed01a718a32c2ba638c7d5154b` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/model_outputs/risk_decile_churn.png` | 43773 | `81de88ae0694b8c4305960d3be81c853f2989fbf6b8ef9fd30086d801baec3e7` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/model_outputs/risk_deciles.csv` | 838 | `076be49586672c98ddda23a02ad94e2cd27382434b64b7bbd3072fec029332b2` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/model_outputs/tenure_churn_curve.png` | 75875 | `221b680178425dc3618165f78b52817415b7bafb8fca3331442547fc8da92fb5` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/results/financial_sensitivity.csv` | 1809 | `15a774b42f125445ea14dabf4520050d370a77f20339c211680e2518a7367d97` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/results/notebook_code_cell_check.json` | 1195 | `9e3d00adfcbae3ccdbdfc8f4a74ba4a02c42f8fff89422c1cc0d7baec7b57e81` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/results/recreated_submission_stats.json` | 1457 | `14abbfec2732e993caf2f6eab4108e77d55b63809a18bf943f7df5dc0f97639b` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/results/reproduced_model_comparison.csv` | 398 | `ac728f183d2aaae7c9836ef4b976552d136f511e30793791d163f767ef797473` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/results/risk_deciles.csv` | 786 | `619c0bf666e9730e99522ea822fdd7be285c130bffe973b6c71be42524fb9821` |
| `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/Omnicampus_Form_Answers_and_Upload_Guide.txt` | 5756 | `6ef3ba79312df5272c5ebb3741d731c6942c5aa826863de17df8e5d8a1b23130` |
| `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/CompanyA_Telecom_Churn_Complete_Chat_Experiment_Log.docx` | 49444 | `542a522c1fee2426ec8fbd705ecdc9080eb90498ce3e2890bbf3b9bcd35a1719` |
| `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/CompanyA_Telecom_Churn_Complete_Chat_Experiment_Log.md` | 28472 | `4b6aafba8ad25336f91ba92a5e8e9867cc37449342e4f61864841701c9731641` |
| `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/domain_feature_consolidated_results.csv` | 3280 | `258e5c31970b23c641eccd6d2807534ca7daead572d51e9400081df5cf8d65ab` |
| `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/v11_cv_feature_family_comparison.csv` | 609 | `d2b02538b874767f3ba35103999d7644472d3fda7c3aa6b7a870aa4b97f1ad14` |
| `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/v10_feature_finance_continuation_report.md` | 22991 | `af1f52879462c6e35240d4c35b4f3c45d56c589bb1b4e3184db4690cf41ac70e` |
| `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/domain_feature_family_results.csv` | 1648 | `6e027a1807dc440fe3c1fd5470f9722e056d37672ea8191bbbef0dbd6a741a2e` |
| `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/v9_technical_audit_and_slide_fixes.md` | 12139 | `46bff6ddf2781140f76c35834bf68a71dbc5c43f5885fc715267d87e57a8ed9f` |
| `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/targeted_feature_optimization_results.csv` | 727 | `6275a20d57a5a14f8edbbb236d8119099d87ec2b6fe0968c0a03cfdc6b9703bc` |
| `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/v10_morning_to_now_report.md` | 27109 | `eadcbb7ef5f0f8b8c057948463de26709ad8e0604a318a9dc36af4cfd3760048` |
| `zip/figures_and_results (1).zip!/figures/01_missing_values.png` | 68184 | `c84142f4399d117bc74cd035714b867c2fba58a6b98f4df7a3a7f9427697b836` |
| `zip/figures_and_results (1).zip!/figures/02_churn_patterns.png` | 211590 | `e8804e52f5c764bf686fafc4ed2dc8bc56ad26de2625b3ac4498515cf7092735` |
| `zip/figures_and_results (1).zip!/figures/03_tenure_x_equipment_heatmap.png` | 92690 | `7b4f38e01f4571e214968fd210ea8a73f47935fca435d096f30ea77ca1eb4db6` |
| `zip/figures_and_results (1).zip!/figures/04_feature_correlations.png` | 106764 | `74233208401e4e4f0a02137ab6c5568168f32d147ad72f16e974f1f96ecbf725` |
| `zip/figures_and_results (1).zip!/figures/04b_hidden_structure.png` | 254733 | `dba5952788129e36efcaad013663ac6281cd718ac30d3506a768e0cc1c7870e8` |
| `zip/figures_and_results (1).zip!/figures/05_model1_performance.png` | 224892 | `2e270730a5ea92dd4fe47d12ef4cfd9ad4e29faa0ba8e36c33d1cd35821c33a9` |
| `zip/figures_and_results (1).zip!/figures/06_feature_importance.png` | 88244 | `c5c2a40548d06571a4bcdca00ee9bcde665cccd43fde096c8c7129f1e2c896e1` |
| `zip/figures_and_results (1).zip!/figures/07_driver_screening.png` | 154687 | `b0e8f31616dfed268adadfb728e375c128105a107004bdc11f68b2ac7f267259` |
| `zip/figures_and_results (1).zip!/figures/08_risk_clusters.png` | 455211 | `684f3d3f218bfa2fc18d39317b210faaa74ad900a0a930152ecfc4db98f4349b` |
| `zip/figures_and_results (1).zip!/figures/09_time_hazard_curves.png` | 268665 | `19586ee9eb0a8ac50977f7cf52967d045d3a1a478fdf9038ce3c9824b4198f3b` |
| `zip/figures_and_results (1).zip!/figures/10_timing_model_comparison.png` | 88217 | `fb4ee2c155d61934092820137758673fd7b616dbd04ce384e278991795151dc5` |
| `zip/figures_and_results (1).zip!/figures/11_profit_curve.png` | 182574 | `c73eceff47e8c878d45f82d58356f6a50bbadb387e8d50f285777512146c483b` |
| `zip/figures_and_results (1).zip!/figures/12_sensitivity_heatmap.png` | 191654 | `46bf9fbf437a12058e14af8acda6a36f92c02493cebad43538ca70921c8a7bd7` |
| `zip/figures_and_results (1).zip!/figures/13_action_plan.png` | 102679 | `6aa402c96a98b0ec4ae88b3f752f12a6d7e9a4018ecfe552620f434d93e8bd55` |
| `zip/figures_and_results (1).zip!/results_summary.json` | 6795 | `abddb3cf89c6c8dc7da2867d07732d46c7420ef8a6e724eaca06520071940c98` |
| `zip/figures_and_results (1).zip!/slide_summary.txt` | 1745 | `c9142b25c17c52927df58063658c3a6c36b47b09dbba077619804b4b41ad1a05` |
| `zip/figures_and_results (1).zip!/action_list_top1000.csv` | 130539 | `ac3c01b62cfa445eadf27b547adca5f99eb8b8fd3bbcc10ea03b777c9f92066f` |
| `zip/figures_and_results (2).zip!/figures/01_missing_values.png` | 68184 | `c84142f4399d117bc74cd035714b867c2fba58a6b98f4df7a3a7f9427697b836` |
| `zip/figures_and_results (2).zip!/figures/02_churn_patterns.png` | 211590 | `e8804e52f5c764bf686fafc4ed2dc8bc56ad26de2625b3ac4498515cf7092735` |
| `zip/figures_and_results (2).zip!/figures/03_tenure_x_equipment_heatmap.png` | 92690 | `7b4f38e01f4571e214968fd210ea8a73f47935fca435d096f30ea77ca1eb4db6` |
| `zip/figures_and_results (2).zip!/figures/04_feature_correlations.png` | 106764 | `74233208401e4e4f0a02137ab6c5568168f32d147ad72f16e974f1f96ecbf725` |
| `zip/figures_and_results (2).zip!/figures/04b_hidden_structure.png` | 254733 | `dba5952788129e36efcaad013663ac6281cd718ac30d3506a768e0cc1c7870e8` |
| `zip/figures_and_results (2).zip!/figures/05_model1_performance.png` | 224892 | `2e270730a5ea92dd4fe47d12ef4cfd9ad4e29faa0ba8e36c33d1cd35821c33a9` |
| `zip/figures_and_results (2).zip!/figures/06_feature_importance.png` | 88244 | `c5c2a40548d06571a4bcdca00ee9bcde665cccd43fde096c8c7129f1e2c896e1` |
| `zip/figures_and_results (2).zip!/figures/07_driver_screening.png` | 154687 | `b0e8f31616dfed268adadfb728e375c128105a107004bdc11f68b2ac7f267259` |
| `zip/figures_and_results (2).zip!/figures/08_risk_clusters.png` | 455211 | `684f3d3f218bfa2fc18d39317b210faaa74ad900a0a930152ecfc4db98f4349b` |
| `zip/figures_and_results (2).zip!/figures/09_time_hazard_curves.png` | 268665 | `19586ee9eb0a8ac50977f7cf52967d045d3a1a478fdf9038ce3c9824b4198f3b` |
| `zip/figures_and_results (2).zip!/figures/10_timing_model_comparison.png` | 88217 | `fb4ee2c155d61934092820137758673fd7b616dbd04ce384e278991795151dc5` |
| `zip/figures_and_results (2).zip!/figures/11_profit_curve.png` | 182574 | `c73eceff47e8c878d45f82d58356f6a50bbadb387e8d50f285777512146c483b` |
| `zip/figures_and_results (2).zip!/figures/12_sensitivity_heatmap.png` | 191654 | `46bf9fbf437a12058e14af8acda6a36f92c02493cebad43538ca70921c8a7bd7` |
| `zip/figures_and_results (2).zip!/figures/13_action_plan.png` | 102679 | `6aa402c96a98b0ec4ae88b3f752f12a6d7e9a4018ecfe552620f434d93e8bd55` |
| `zip/figures_and_results (2).zip!/results_summary.json` | 6795 | `abddb3cf89c6c8dc7da2867d07732d46c7420ef8a6e724eaca06520071940c98` |
| `zip/figures_and_results (2).zip!/slide_summary.txt` | 1745 | `c9142b25c17c52927df58063658c3a6c36b47b09dbba077619804b4b41ad1a05` |
| `zip/figures_and_results (2).zip!/action_list_top1000.csv` | 130539 | `ac3c01b62cfa445eadf27b547adca5f99eb8b8fd3bbcc10ea03b777c9f92066f` |
| `zip/figures_and_results (3).zip!/figures/01_missing_values.png` | 68184 | `c84142f4399d117bc74cd035714b867c2fba58a6b98f4df7a3a7f9427697b836` |
| `zip/figures_and_results (3).zip!/figures/02_churn_patterns.png` | 211590 | `e8804e52f5c764bf686fafc4ed2dc8bc56ad26de2625b3ac4498515cf7092735` |
| `zip/figures_and_results (3).zip!/figures/03_tenure_x_equipment_heatmap.png` | 92690 | `7b4f38e01f4571e214968fd210ea8a73f47935fca435d096f30ea77ca1eb4db6` |
| `zip/figures_and_results (3).zip!/figures/04_feature_correlations.png` | 106764 | `74233208401e4e4f0a02137ab6c5568168f32d147ad72f16e974f1f96ecbf725` |
| `zip/figures_and_results (3).zip!/figures/04b_hidden_structure.png` | 254452 | `02f3b799795dd2adbd43462e24331a5772b9566448a87bf0b9ec11ad33e68f0c` |
| `zip/figures_and_results (3).zip!/figures/05_model1_performance.png` | 181155 | `a622b704aa490d67b0fa81ee21a27f12e2f47e9af86403746ad9fe4fb87165df` |
| `zip/figures_and_results (3).zip!/figures/05b_learning_curve.png` | 88601 | `49297a8b3db6195e9a3402c0e167aa3f4f17f5103a7be27b9af9bf0ebab1efa0` |
| `zip/figures_and_results (3).zip!/figures/05c_confidence_tiers.png` | 108383 | `eb2b4661e4a758a31e05abfcfd64ac801cc56730d0bdd1ac1a72aaba06e46475` |
| `zip/figures_and_results (3).zip!/figures/06_feature_importance.png` | 85218 | `52488059c2555a4696db489e325365d22c5ec933041f92b7795f5fcc4788149f` |
| `zip/figures_and_results (3).zip!/figures/07_driver_screening.png` | 154687 | `b0e8f31616dfed268adadfb728e375c128105a107004bdc11f68b2ac7f267259` |
| `zip/figures_and_results (3).zip!/figures/08_risk_clusters.png` | 448726 | `8e9b4262ca9e25b2c4aaf21105e80b3909b706bda44094a551df1db43bc7107a` |
| `zip/figures_and_results (3).zip!/figures/09_time_hazard_curves.png` | 268665 | `19586ee9eb0a8ac50977f7cf52967d045d3a1a478fdf9038ce3c9824b4198f3b` |
| `zip/figures_and_results (3).zip!/figures/10_timing_model_comparison.png` | 87843 | `4d2ae5bc0ea83a4c75cddea33e1ba933b5193a0cb6477215322b9e4e36bda64a` |
| `zip/figures_and_results (3).zip!/figures/11_profit_curve.png` | 181412 | `d56d0eda62028f7119f1f419e93b577cdb814a0a7e2896b52fe69fe72cbd64aa` |
| `zip/figures_and_results (3).zip!/figures/12_sensitivity_heatmap.png` | 189128 | `8537b6eeb432c1c73a549988bc98ff3b4dde50b5a1b289a01ae8ad3e9b034f10` |
| `zip/figures_and_results (3).zip!/figures/13_action_plan.png` | 102643 | `14962edf8ba9ae1590d8d281201742eab36fd1354970809ca596dabda375646f` |
| `zip/figures_and_results (3).zip!/results_summary.json` | 8932 | `f1b3139e3750bfcec86cb06f4c0f6fbb34fc0f31a49adc12f0f168f1af80d31c` |
| `zip/figures_and_results (3).zip!/slide_summary.txt` | 1961 | `16e535210576efeab26bf0a2fcc668c66005ae66e6d5942b78b0664937803262` |
| `zip/figures_and_results (3).zip!/action_list_top1000.csv` | 130521 | `2bf9d2badc8c5304116bbbb8712b46856e92f6e1fdb4b332b818a2074dc62d89` |
| `zip/figures_and_results (4).zip!/figures/01_missing_values.png` | 65600 | `c069cfb0232abc70761900e700943830f3a5072b53ef2fc60a24f3820e8693ff` |
| `zip/figures_and_results (4).zip!/figures/02_churn_patterns.png` | 216065 | `6b00073ba344c54cfc86698b32525169aca3a57a56bdb851328d42d20487e83d` |
| `zip/figures_and_results (4).zip!/figures/03_tenure_x_equipment_heatmap.png` | 91505 | `97724b4a097e93a265067905ba6911aada5cb426167f7803b5fb4ab9e1b362cb` |
| `zip/figures_and_results (4).zip!/figures/04_feature_correlations.png` | 106674 | `999b382c689912a2955a52c1fd7efb8c74a143b3150842f1a3a4cc4480e08cab` |
| `zip/figures_and_results (4).zip!/figures/04b_hidden_structure.png` | 207699 | `29bd3ccd57f747981733c659474ca58194540f2c5d6ca8a359def3d91a660e8d` |
| `zip/figures_and_results (4).zip!/figures/05_model1_performance.png` | 186589 | `7549cf3118a697ee9ddabf1a8b0873c2412ae874029f1199ed0c2b2da5f56b5a` |
| `zip/figures_and_results (4).zip!/figures/05b_learning_curve.png` | 89690 | `db4e82a51e07e868fbb768cd3d165534e0c4d07aef88a75ca1c3d9b00973cbbc` |
| `zip/figures_and_results (4).zip!/figures/05c_confidence_tiers.png` | 108086 | `d613b7b3af9a67ad08ff233cc649a8872c1ea859c3ef70c86d89305e3c0c4fe7` |
| `zip/figures_and_results (4).zip!/figures/06_feature_importance.png` | 90659 | `2b97a607993403b49ad9d56031fe51e64d39f61b922d78105a66664759c4cf56` |
| `zip/figures_and_results (4).zip!/figures/07_driver_screening.png` | 173057 | `4b76528b68229397232b3201812e53d5d5bd93cb6cd0ad2561d5507d061b2dc7` |
| `zip/figures_and_results (4).zip!/figures/08_risk_clusters.png` | 517478 | `da59805444ec36ea2b13dc41d2a41d66d30b419a0ba9679362acad28e5999ad3` |
| `zip/figures_and_results (4).zip!/figures/09_time_hazard_curves.png` | 261990 | `dd6127d1c36944cca0b49ad98dd6dc75f8854fcb0a7eff4acc6b7604fbd54453` |
| `zip/figures_and_results (4).zip!/figures/10_timing_model_comparison.png` | 89760 | `019d4a3cefe3f16bc8ea3119432c7d4da6f9f90ea1904b37b265f23af3e346c8` |
| `zip/figures_and_results (4).zip!/figures/11_profit_curve.png` | 193722 | `2737e36adeb41aabbe729317f7a5e210a9cbc90bd5305a0e39f6ebe958e0af49` |
| `zip/figures_and_results (4).zip!/figures/12_sensitivity_heatmap.png` | 191804 | `dc9ce6ebfda3ab359fd3c146aaec9cad8d74842ff351015edced72aea0cde129` |
| `zip/figures_and_results (4).zip!/figures/13_action_plan.png` | 101687 | `9f6587b402f29e97bfa8e9f277854fed33728303fbabb90ff16cefb79847d65f` |
| `zip/figures_and_results (4).zip!/results_summary.json` | 8417 | `f638901529d6b5286d8c91f49775e7b573e36a83b21d7d1f7c500572bd1f5da2` |
| `zip/figures_and_results (4).zip!/slide_summary.txt` | 2000 | `ef9dd1472afcb0bccd2ee149d78c4744e25a8ed3b6c3f684e46d72c8351db553` |
| `zip/figures_and_results (4).zip!/action_list_top1000.csv` | 122518 | `47943f0bf34cd7ce6b1df4931750dca7b69aad4dcd84eaf8cce1660f80d7e83c` |
| `zip/figures_and_results (5).zip!/figures/01_missing_values.png` | 68184 | `c84142f4399d117bc74cd035714b867c2fba58a6b98f4df7a3a7f9427697b836` |
| `zip/figures_and_results (5).zip!/figures/02_churn_patterns.png` | 211590 | `e8804e52f5c764bf686fafc4ed2dc8bc56ad26de2625b3ac4498515cf7092735` |
| `zip/figures_and_results (5).zip!/figures/03_tenure_x_equipment_heatmap.png` | 92690 | `7b4f38e01f4571e214968fd210ea8a73f47935fca435d096f30ea77ca1eb4db6` |
| `zip/figures_and_results (5).zip!/figures/04_feature_correlations.png` | 106764 | `74233208401e4e4f0a02137ab6c5568168f32d147ad72f16e974f1f96ecbf725` |
| `zip/figures_and_results (5).zip!/figures/04b_hidden_structure.png` | 254452 | `02f3b799795dd2adbd43462e24331a5772b9566448a87bf0b9ec11ad33e68f0c` |
| `zip/figures_and_results (5).zip!/figures/05_model1_performance.png` | 181155 | `a622b704aa490d67b0fa81ee21a27f12e2f47e9af86403746ad9fe4fb87165df` |
| `zip/figures_and_results (5).zip!/figures/05b_learning_curve.png` | 88601 | `49297a8b3db6195e9a3402c0e167aa3f4f17f5103a7be27b9af9bf0ebab1efa0` |
| `zip/figures_and_results (5).zip!/figures/05c_confidence_tiers.png` | 108383 | `eb2b4661e4a758a31e05abfcfd64ac801cc56730d0bdd1ac1a72aaba06e46475` |
| `zip/figures_and_results (5).zip!/figures/06_feature_importance.png` | 85218 | `52488059c2555a4696db489e325365d22c5ec933041f92b7795f5fcc4788149f` |
| `zip/figures_and_results (5).zip!/figures/07_driver_screening.png` | 173700 | `317d07230fddb375fd02d1e5f4a11bda699d1817d016af11ccda002eaa36f361` |
| `zip/figures_and_results (5).zip!/figures/08_risk_clusters.png` | 448726 | `8e9b4262ca9e25b2c4aaf21105e80b3909b706bda44094a551df1db43bc7107a` |
| `zip/figures_and_results (5).zip!/figures/09_time_hazard_curves.png` | 268665 | `19586ee9eb0a8ac50977f7cf52967d045d3a1a478fdf9038ce3c9824b4198f3b` |
| `zip/figures_and_results (5).zip!/figures/10_timing_model_comparison.png` | 87843 | `4d2ae5bc0ea83a4c75cddea33e1ba933b5193a0cb6477215322b9e4e36bda64a` |
| `zip/figures_and_results (5).zip!/figures/11_profit_curve.png` | 174411 | `033c599ab91be453327afca48acdf0e3ede91816c2433d101c5b04b587e6b8b1` |
| `zip/figures_and_results (5).zip!/figures/12_sensitivity_heatmap.png` | 193952 | `b4bbe9785b693f647909577ba5ef1e64ce8d3d2bfe49b7c6b62e9ac897d8d1b9` |
| `zip/figures_and_results (5).zip!/figures/13_action_plan.png` | 102364 | `44d7e41cc97a6ac2f5499d52e810b322a0594658878a85a8b9130698be4839f9` |
| `zip/figures_and_results (5).zip!/results_summary.json` | 9389 | `839b9b41de2ca51935152b1190c046e37d579c700db2633ce3d3c7a55ad8583c` |
| `zip/figures_and_results (5).zip!/slide_summary.txt` | 2111 | `2e53d3040460c5b90b7d9aff93be848f7f32ed87edb18cc745dc075c1e81fccc` |
| `zip/figures_and_results (5).zip!/action_list_top1000.csv` | 130236 | `21b5b639a3764defd1b44d66083e6c1f309bfae993db15c943a0ecb252e5c2d4` |
| `zip/figures_and_results (6).zip!/figures/01_missing_values.png` | 68184 | `c84142f4399d117bc74cd035714b867c2fba58a6b98f4df7a3a7f9427697b836` |
| `zip/figures_and_results (6).zip!/figures/02_churn_patterns.png` | 211590 | `e8804e52f5c764bf686fafc4ed2dc8bc56ad26de2625b3ac4498515cf7092735` |
| `zip/figures_and_results (6).zip!/figures/03_tenure_x_equipment_heatmap.png` | 92690 | `7b4f38e01f4571e214968fd210ea8a73f47935fca435d096f30ea77ca1eb4db6` |
| `zip/figures_and_results (6).zip!/figures/04_feature_correlations.png` | 106764 | `74233208401e4e4f0a02137ab6c5568168f32d147ad72f16e974f1f96ecbf725` |
| `zip/figures_and_results (6).zip!/figures/04b_hidden_structure.png` | 254452 | `02f3b799795dd2adbd43462e24331a5772b9566448a87bf0b9ec11ad33e68f0c` |
| `zip/figures_and_results (6).zip!/figures/05_model1_performance.png` | 181155 | `a622b704aa490d67b0fa81ee21a27f12e2f47e9af86403746ad9fe4fb87165df` |
| `zip/figures_and_results (6).zip!/figures/05b_learning_curve.png` | 88601 | `49297a8b3db6195e9a3402c0e167aa3f4f17f5103a7be27b9af9bf0ebab1efa0` |
| `zip/figures_and_results (6).zip!/figures/05c_confidence_tiers.png` | 108383 | `eb2b4661e4a758a31e05abfcfd64ac801cc56730d0bdd1ac1a72aaba06e46475` |
| `zip/figures_and_results (6).zip!/figures/06_feature_importance.png` | 85218 | `52488059c2555a4696db489e325365d22c5ec933041f92b7795f5fcc4788149f` |
| `zip/figures_and_results (6).zip!/figures/07_driver_screening.png` | 173700 | `317d07230fddb375fd02d1e5f4a11bda699d1817d016af11ccda002eaa36f361` |
| `zip/figures_and_results (6).zip!/figures/08_risk_clusters.png` | 448726 | `8e9b4262ca9e25b2c4aaf21105e80b3909b706bda44094a551df1db43bc7107a` |
| `zip/figures_and_results (6).zip!/figures/09_time_hazard_curves.png` | 268665 | `19586ee9eb0a8ac50977f7cf52967d045d3a1a478fdf9038ce3c9824b4198f3b` |
| `zip/figures_and_results (6).zip!/figures/10_timing_model_comparison.png` | 87843 | `4d2ae5bc0ea83a4c75cddea33e1ba933b5193a0cb6477215322b9e4e36bda64a` |
| `zip/figures_and_results (6).zip!/figures/11_profit_curve.png` | 174411 | `033c599ab91be453327afca48acdf0e3ede91816c2433d101c5b04b587e6b8b1` |
| `zip/figures_and_results (6).zip!/figures/12_sensitivity_heatmap.png` | 193952 | `b4bbe9785b693f647909577ba5ef1e64ce8d3d2bfe49b7c6b62e9ac897d8d1b9` |
| `zip/figures_and_results (6).zip!/figures/13_action_plan.png` | 102364 | `44d7e41cc97a6ac2f5499d52e810b322a0594658878a85a8b9130698be4839f9` |
| `zip/figures_and_results (6).zip!/results_summary.json` | 9389 | `839b9b41de2ca51935152b1190c046e37d579c700db2633ce3d3c7a55ad8583c` |
| `zip/figures_and_results (6).zip!/slide_summary.txt` | 2111 | `2e53d3040460c5b90b7d9aff93be848f7f32ed87edb18cc745dc075c1e81fccc` |
| `zip/figures_and_results (6).zip!/action_list_top1000.csv` | 130236 | `21b5b639a3764defd1b44d66083e6c1f309bfae993db15c943a0ecb252e5c2d4` |
| `zip/figures_and_results.zip!/figures/01_missing_values.png` | 68184 | `c84142f4399d117bc74cd035714b867c2fba58a6b98f4df7a3a7f9427697b836` |
| `zip/figures_and_results.zip!/figures/02_churn_patterns.png` | 211590 | `e8804e52f5c764bf686fafc4ed2dc8bc56ad26de2625b3ac4498515cf7092735` |
| `zip/figures_and_results.zip!/figures/03_tenure_x_equipment_heatmap.png` | 92690 | `7b4f38e01f4571e214968fd210ea8a73f47935fca435d096f30ea77ca1eb4db6` |
| `zip/figures_and_results.zip!/figures/04_feature_correlations.png` | 106764 | `74233208401e4e4f0a02137ab6c5568168f32d147ad72f16e974f1f96ecbf725` |
| `zip/figures_and_results.zip!/figures/05_model1_performance.png` | 194761 | `8694c23ccea63fea00e53bb6838f20574d30b4aeec6ee3202a98638460f79a4f` |
| `zip/figures_and_results.zip!/figures/06_feature_importance.png` | 89557 | `ebb077421c7fc61b11cbb05059de1640d4b35c9781a579bf6374549e9699f6a0` |
| `zip/figures_and_results.zip!/figures/06b_shap_summary.png` | 257636 | `eae37d0e876ce8358fb1cbc091eb713c39feed2b0789211849ed42a96f11c1dd` |
| `zip/figures_and_results.zip!/figures/07_driver_screening.png` | 154687 | `b0e8f31616dfed268adadfb728e375c128105a107004bdc11f68b2ac7f267259` |
| `zip/figures_and_results.zip!/figures/08_risk_clusters.png` | 446695 | `4424056f06a55a0c39de107d3abb2ae4496405dd623442bb61fd69bed0e522aa` |
| `zip/figures_and_results.zip!/figures/09_time_hazard_curves.png` | 268665 | `19586ee9eb0a8ac50977f7cf52967d045d3a1a478fdf9038ce3c9824b4198f3b` |
| `zip/figures_and_results.zip!/figures/10_timing_model_comparison.png` | 86645 | `02db214042b665a4fd6dcb237ce79683ef1f53cec5d09d9d860e78e606853c50` |
| `zip/figures_and_results.zip!/figures/11_profit_curve.png` | 182635 | `ef4f5336e3b8da7ed13939ab08b8cd6318335528ec4a08b45272a07bf415ff47` |
| `zip/figures_and_results.zip!/figures/12_sensitivity_heatmap.png` | 193058 | `5f1e877885e457f2f009585c4d5213f589282cc49660904b0b71099d38662b10` |
| `zip/figures_and_results.zip!/figures/13_action_plan.png` | 102674 | `7362502aacaeb730cc53a6d9adf70502fdaaf7caaec679ef7362ed0ea89496ac` |
| `zip/figures_and_results.zip!/results_summary.json` | 5671 | `dcf33c8351ce8e16f430381667492be0bde1790f59e72ef4b64bbac6fccbe6f9` |
| `zip/figures_and_results.zip!/slide_summary.txt` | 1731 | `e73ff5976afea576b7eacbafb3794503f07641f290ca519c8d57b8df7a5ce8b7` |
| `zip/figures_and_results.zip!/action_list_top1000.csv` | 122076 | `667de49b595b252ebbd340ad3affd1f73e9bee95da5603478f9a8cb946458b49` |
| `zip/figures_and_results_v10.zip!/figures_v10/01_missing_values.png` | 68184 | `c84142f4399d117bc74cd035714b867c2fba58a6b98f4df7a3a7f9427697b836` |
| `zip/figures_and_results_v10.zip!/figures_v10/02_churn_patterns.png` | 211590 | `e8804e52f5c764bf686fafc4ed2dc8bc56ad26de2625b3ac4498515cf7092735` |
| `zip/figures_and_results_v10.zip!/figures_v10/03_tenure_x_equipment_heatmap.png` | 92690 | `7b4f38e01f4571e214968fd210ea8a73f47935fca435d096f30ea77ca1eb4db6` |
| `zip/figures_and_results_v10.zip!/figures_v10/04_feature_correlations.png` | 106764 | `74233208401e4e4f0a02137ab6c5568168f32d147ad72f16e974f1f96ecbf725` |
| `zip/figures_and_results_v10.zip!/figures_v10/04b_hidden_structure.png` | 257825 | `42d3e92eaead62b7fa0fa313235994b5dac82599a165b0333137c2ed28212d89` |
| `zip/figures_and_results_v10.zip!/figures_v10/04c_rates_of_change.png` | 191148 | `346033b0a6c6cb3df88f7d621faad40db0354ca91c727183a654ea729984b49c` |
| `zip/figures_and_results_v10.zip!/figures_v10/05_model1_performance.png` | 183275 | `ad2cbdcfe4423b77800476dda4df87785a1ef1ef40b5bf8e90c26c710fcb10d3` |
| `zip/figures_and_results_v10.zip!/figures_v10/05b_learning_curve.png` | 88270 | `799dd21fa325e194247356f5eaa82afb8d0a3697da5b0f3bf345049766bee970` |
| `zip/figures_and_results_v10.zip!/figures_v10/05c_confidence_tiers.png` | 108230 | `511ee65e144dfb2dc1d5948d8c24cf034add3a759a82b78e39ec3d4219de8fc5` |
| `zip/figures_and_results_v10.zip!/figures_v10/05d_calibration.png` | 109105 | `aa80f3f18ec1c779aadd3a4df1086fdda0c4a0007f012f1a9f9bce5bbe0dbc88` |
| `zip/figures_and_results_v10.zip!/figures_v10/06_feature_importance.png` | 81238 | `dec1c46aefc834d4de772fb12e91cf268f9ba64b230fbea62d1169a688f49e4c` |
| `zip/figures_and_results_v10.zip!/figures_v10/07_driver_screening.png` | 175219 | `fb921a89def863e09b88abe082840f3ac7f4f9420694b356b1d2883477710925` |
| `zip/figures_and_results_v10.zip!/figures_v10/08_risk_clusters.png` | 451917 | `52addbff4b78a560eebf1c6d7336a142a92f31485fb0eeb6efe2530b9ebcbe8e` |
| `zip/figures_and_results_v10.zip!/figures_v10/09_time_churn_rate_curves.png` | 268665 | `19586ee9eb0a8ac50977f7cf52967d045d3a1a478fdf9038ce3c9824b4198f3b` |
| `zip/figures_and_results_v10.zip!/figures_v10/10_timing_model_comparison.png` | 88204 | `9bff728e517dbbb87bbc818c8aa8b7c4bbe171ff2b7e1b1c643b25a2ef762593` |
| `zip/figures_and_results_v10.zip!/figures_v10/11_profit_curve.png` | 178397 | `47f0f3f0e6f5635de6ae1c0bd899183125b1c3c5522dc4f05744070c8936b0a4` |
| `zip/figures_and_results_v10.zip!/figures_v10/12_sensitivity_heatmap.png` | 199757 | `02c8110ed85d13a088af7ffc89f76671dbb4dac575cb6ca476441f4267e08718` |
| `zip/figures_and_results_v10.zip!/figures_v10/13_action_plan.png` | 102833 | `7c33d0a65e19f350afbfb9cebecaf4977c68befe9f91941cb56fa9e4f6688564` |
| `zip/figures_and_results_v10.zip!/results_summary_v10.json` | 12569 | `581c5265c945d0ed0993f0495d7614a4e340200732cf1d0f54286dd8b5483849` |
| `zip/figures_and_results_v10.zip!/slide_summary_v10.txt` | 2941 | `2317c53b933eb0736fe33f9a7d21795776e1dbf7059c7cf476c83a8c4bd2850b` |
| `zip/figures_and_results_v10.zip!/action_list_top1000_v10.csv` | 129896 | `f6854129c8a3d6340e8f1d2cc653c8a49b6f105ef61f7184485bd93a0b1e7108` |
| `zip/figures_and_results_v11.zip!/figures_v11/01_missing_values.png` | 68184 | `c84142f4399d117bc74cd035714b867c2fba58a6b98f4df7a3a7f9427697b836` |
| `zip/figures_and_results_v11.zip!/figures_v11/02_churn_patterns.png` | 211590 | `e8804e52f5c764bf686fafc4ed2dc8bc56ad26de2625b3ac4498515cf7092735` |
| `zip/figures_and_results_v11.zip!/figures_v11/03_tenure_x_equipment_heatmap.png` | 92690 | `7b4f38e01f4571e214968fd210ea8a73f47935fca435d096f30ea77ca1eb4db6` |
| `zip/figures_and_results_v11.zip!/figures_v11/04_feature_correlations.png` | 106764 | `74233208401e4e4f0a02137ab6c5568168f32d147ad72f16e974f1f96ecbf725` |
| `zip/figures_and_results_v11.zip!/figures_v11/04b_hidden_structure.png` | 257825 | `42d3e92eaead62b7fa0fa313235994b5dac82599a165b0333137c2ed28212d89` |
| `zip/figures_and_results_v11.zip!/figures_v11/04c_rates_of_change.png` | 191148 | `346033b0a6c6cb3df88f7d621faad40db0354ca91c727183a654ea729984b49c` |
| `zip/figures_and_results_v11.zip!/figures_v11/05_model1_performance.png` | 183011 | `3772e1c1fc8dcf3af329425cdc6c1a20b06552960728347dce53030d3dd5c71d` |
| `zip/figures_and_results_v11.zip!/figures_v11/05b_learning_curve.png` | 88913 | `2ed292856340fd008c9f7476bd60c12b12413b9993f8090290c87c4c0e1507e3` |
| `zip/figures_and_results_v11.zip!/figures_v11/05c_confidence_tiers.png` | 108425 | `b1acb70c4ca0c401410717268d9b5f4c1a8f8d1c3da57683f4977a6cdd831d7e` |
| `zip/figures_and_results_v11.zip!/figures_v11/05d_calibration.png` | 108766 | `433a5d2e1c7915199d4224f10047d0bf8f42e1dfb24b167e4d803afad2235e1a` |
| `zip/figures_and_results_v11.zip!/figures_v11/06_feature_importance.png` | 86488 | `b89a94210b1d08352334a640c0925276bbfc64fdf9f6bc96538a2c3f8318239f` |
| `zip/figures_and_results_v11.zip!/figures_v11/07_driver_screening.png` | 175219 | `fb921a89def863e09b88abe082840f3ac7f4f9420694b356b1d2883477710925` |
| `zip/figures_and_results_v11.zip!/figures_v11/08_risk_clusters.png` | 447157 | `00d5c869a73e7d6defec51d48b48e795413ecacb8e2b1be285821f7e3abede3f` |
| `zip/figures_and_results_v11.zip!/figures_v11/09_time_churn_rate_curves.png` | 268665 | `19586ee9eb0a8ac50977f7cf52967d045d3a1a478fdf9038ce3c9824b4198f3b` |
| `zip/figures_and_results_v11.zip!/figures_v11/10_timing_model_comparison.png` | 88204 | `9ae7e3ee121f60569244b9a364c7762a3c8e1009f06ed5f5fafb89c23ca7ece2` |
| `zip/figures_and_results_v11.zip!/figures_v11/11_profit_curve.png` | 178076 | `aa93ffce4d137a784b1c0d676e31eb45f957231c12dbc2c7ecdf79f17b48edb3` |
| `zip/figures_and_results_v11.zip!/figures_v11/12_sensitivity_heatmap.png` | 200751 | `23d0f10005a261b8c54cdc0acc02dcd437fee63566dab066148bf62b85c6c19f` |
| `zip/figures_and_results_v11.zip!/figures_v11/13_action_plan.png` | 102765 | `60c040da2076d0dc93df57621f565b84354194db57ecdc64a2e7f202c1f548fb` |
| `zip/figures_and_results_v11.zip!/results_summary_v11.json` | 12861 | `d787859cca8e3f3750936e7be909b41588dee63ef1f27567605be67a9871f75b` |
| `zip/figures_and_results_v11.zip!/slide_summary_v11.txt` | 3115 | `ebb7b4e7a6c38ea75f12f033fd4c7390595d460bc0fe2ab752b46ee763775427` |
| `zip/figures_and_results_v11.zip!/action_list_top1000_v11.csv` | 129916 | `8315bcd2724b941ef0402494cca86c7c6dfe89c37199dbb965d951f46862f416` |
| `zip/figures_and_results_v9.zip!/figures_v9/01_missing_values.png` | 68184 | `c84142f4399d117bc74cd035714b867c2fba58a6b98f4df7a3a7f9427697b836` |
| `zip/figures_and_results_v9.zip!/figures_v9/02_churn_patterns.png` | 211590 | `e8804e52f5c764bf686fafc4ed2dc8bc56ad26de2625b3ac4498515cf7092735` |
| `zip/figures_and_results_v9.zip!/figures_v9/03_tenure_x_equipment_heatmap.png` | 92690 | `7b4f38e01f4571e214968fd210ea8a73f47935fca435d096f30ea77ca1eb4db6` |
| `zip/figures_and_results_v9.zip!/figures_v9/04_feature_correlations.png` | 106764 | `74233208401e4e4f0a02137ab6c5568168f32d147ad72f16e974f1f96ecbf725` |
| `zip/figures_and_results_v9.zip!/figures_v9/04b_hidden_structure.png` | 254452 | `02f3b799795dd2adbd43462e24331a5772b9566448a87bf0b9ec11ad33e68f0c` |
| `zip/figures_and_results_v9.zip!/figures_v9/04c_rates_of_change.png` | 191148 | `346033b0a6c6cb3df88f7d621faad40db0354ca91c727183a654ea729984b49c` |
| `zip/figures_and_results_v9.zip!/figures_v9/05_model1_performance.png` | 182012 | `0ba23f3fb1c9ecf640b97c2aa809968997eb5e956d206307a0ebce01de947eec` |
| `zip/figures_and_results_v9.zip!/figures_v9/05b_learning_curve.png` | 89029 | `f1447ad0e670239051758fb0e9213cb8e072bc88a70cb18dfa79b569eac7e7c6` |
| `zip/figures_and_results_v9.zip!/figures_v9/05c_confidence_tiers.png` | 108733 | `a405f94cc98d27848b6664f498465fb7416e30184b60e30bd211ca1bcf56cfec` |
| `zip/figures_and_results_v9.zip!/figures_v9/06_feature_importance.png` | 84291 | `2d61449ffcf8f290654b5693a3ecfd05953601be7a9499aa990989884983aeda` |
| `zip/figures_and_results_v9.zip!/figures_v9/07_driver_screening.png` | 175149 | `7cfbda4e6fbc4ca1fddfbf464e50bd649237b1ca5f9f09755504d3c66e283c8d` |
| `zip/figures_and_results_v9.zip!/figures_v9/08_risk_clusters.png` | 447132 | `a2d4ecdffafd71c196adb5cc9a4c0d0ffdfcc996058bda61ee82e930f9ea9072` |
| `zip/figures_and_results_v9.zip!/figures_v9/09_time_hazard_curves.png` | 268665 | `19586ee9eb0a8ac50977f7cf52967d045d3a1a478fdf9038ce3c9824b4198f3b` |
| `zip/figures_and_results_v9.zip!/figures_v9/10_timing_model_comparison.png` | 88257 | `6e5409f74fab3d3a43954960a2bb0e31c555dccf0d50de7e78a202d2b864d623` |
| `zip/figures_and_results_v9.zip!/figures_v9/11_profit_curve.png` | 184031 | `05d00d60aa399c2880b242e9b7303cbfa05fac970f590b4acab4cb8c299819d3` |
| `zip/figures_and_results_v9.zip!/figures_v9/12_sensitivity_heatmap.png` | 201524 | `2a876630f1fb4cb7de234b12a55a0bd9e2468b4c5cc59a2e35344af00bcd3a52` |
| `zip/figures_and_results_v9.zip!/figures_v9/13_action_plan.png` | 102789 | `71d419c9c5f6f5b5133671c8af8764e8c02e81a2d11f3457900019b550792478` |
| `zip/figures_and_results_v9.zip!/results_summary_v9.json` | 10380 | `6859a5687ca4e52c1d91890a1364a6210893050d1480584f348d3a01e7ce9093` |
| `zip/figures_and_results_v9.zip!/slide_summary_v9.txt` | 2110 | `5ce3f4bb9bf15951072561526268e4442010b4cb800e65b63e30ab0216730813` |
| `zip/figures_and_results_v9.zip!/action_list_top1000_v9.csv` | 130053 | `8b57503d7506b5935501d8b96ee69d14148e4404215193b79c6226d9b2b22a90` |
| `zip/GCI_CatBoost_Product_Experiment_Package.zip!/GCI_submission_catboost_product_blend.csv` | 1615463 | `c3fc40999a7fd90fc2c660e48cad2f3b138126aafdf2b4d6d1ca7053df6a9cc4` |
| `zip/GCI_CatBoost_Product_Experiment_Package.zip!/GCI_experiment_report_catboost_product.md` | 6695 | `60dc714b2f39403faecbb5bbb4621c16db30a4f89febb9fe4de3eaf70a0f10d7` |
| `zip/GCI_CatBoost_Product_Experiment_Package.zip!/v9_full_oof_catboost_comparison.json` | 1513 | `3d196469bacd72b3903eeb59b493064fa1fa7272494f123dc2db17fee426028f` |
| `zip/GCI_CatBoost_Product_Experiment_Package.zip!/v9_full_oof_catboost_comparison_folds.csv` | 364 | `2a23ec9db0f1759f05f714b5d6527491edc13a8cdeee69c0650f7f0158a9b252` |
| `zip/GCI_CatBoost_Product_Experiment_Package.zip!/catboost_product_experiment_results.json` | 3538 | `9f427bd537e3511cbbbf0d084863a219819dae339943b7d0b627985315b70231` |
| `zip/GCI_CatBoost_Product_Experiment_Package.zip!/catboost_product_experiment_folds.csv` | 811 | `b7dc77655d22c8d9c2dd50d16f47560205d2be3e4ef51fc4c7dd8cf738a672da` |
| `zip/GCI_CatBoost_Product_Experiment_Package.zip!/run_gci_catboost_product_experiment.py` | 10802 | `bd9932db6c7a23dcd5b5c396ce6e3fc3207a0b6026bfcbc71bcbd61be1464fc7` |
| `zip/GCI_CatBoost_Product_Experiment_Package.zip!/run_v9_full_oof_for_catboost_optimized.py` | 9477 | `0967db91b9c8c604f968b09ebe4459c37eddc33aa1894f519d7b58a7d54dcf56` |
| `zip/GCI_CatBoost_Product_Experiment_Package.zip!/candidate_submission_validation.json` | 915 | `2742a54ebfb239e3eb4bb4c50776f7d66aece8b7059c05915f1a6edbddb11938` |
| `zip/GCI_CA_EXT_promising_research.zip!/gci_promising_ca_ext_test.py` | 4362 | `999bdcc6a7b108f39a2ea78c7cb48e4d77ab5354a3a5b36fe0c69dc8153f360e` |
| `zip/GCI_CA_EXT_promising_research.zip!/metrics.json` | 956 | `2881f5ac6a99356f8f65981b178edb63d1fe50f6ed701732dc2aec0732837e55` |
| `zip/GCI_CA_EXT_promising_research.zip!/folds.csv` | 334 | `3e285318fdd085b3c3ac363366b49316c82241a402db34785724c26edfb170f8` |
| `zip/GCI_CA_EXT_promising_research.zip!/oof.npz` | 2612170 | `93706f491c384786195641b4247877207611c5d49e364a4405ebb70dfd86b7e8` |
| `zip/GCI_Competition_Candidates_Package.zip!/gci_submission_notebook.ipynb` | 5281 | `09bc7d030f43901e6401fa012d62e5fb25e98c6adafbbf6332dce06c580817ee` |
| `zip/GCI_Competition_Candidates_Package.zip!/README_candidates.txt` | 683 | `2e6e86f0dd17d1144b41b9882f2f87686af8e3379bde402510708bd804f6af22` |
| `zip/GCI_Competition_Candidates_Package.zip!/candidate_conservative.csv` | 1660938 | `925f58728eff5dad854f3fcddec6651a733e69cabfbf4e8ecf8968fed5963941` |
| `zip/GCI_Competition_Candidates_Package.zip!/candidate_balanced.csv` | 1659236 | `d41c2977861b7f593fcdc91ef904fdd9c7b0d9d110c69057cf6d088ce8a634cf` |
| `zip/GCI_Competition_Candidates_Package.zip!/candidate_rank.csv` | 1611494 | `2089ae91182097b6d3609cfab4ed12d023bedd7965f269dd08ab10c44e300384` |
| `zip/GCI_Competition_Candidates_Package.zip!/best_submission.csv` | 1666283 | `fe784d4694573d137b059139d385bbbea58224ca45a8d586324dd0e97f3c69af` |
| `zip/GCI_HomeCredit_BestSolution.zip!/best_submission.csv` | 1666283 | `fe784d4694573d137b059139d385bbbea58224ca45a8d586324dd0e97f3c69af` |
| `zip/GCI_HomeCredit_BestSolution.zip!/submission_xgb.csv` | 1130421 | `09e1e4bc5fbb80067a6ac3b9ad03570ca4c258ed667b40698a396b0324410fd1` |
| `zip/GCI_HomeCredit_BestSolution.zip!/submission_lgb.csv` | 1667211 | `ad62d971dc9112658096837261ccb65f3171da334a40517ae748355deae09cb1` |
| `zip/GCI_HomeCredit_BestSolution.zip!/submission_cat.csv` | 1666325 | `e4f128b3b63fcf63704ee594b8c3ec3b88327826cb4e1ac8cc00d6e0a000f334` |
| `zip/GCI_HomeCredit_BestSolution.zip!/submission_rf.csv` | 1620006 | `9fba83f468d296ab99858fa306103bf4870a86eb4185bb0704075c82d21efd6c` |
| `zip/GCI_HomeCredit_BestSolution.zip!/competition_solution.py` | 4192 | `3e56060ef90a557837b44db0d845665088ba3eaf8231835ef92ea7b2c84f8ac5` |
| `zip/GCI_HomeCredit_BestSolution.zip!/model_comparison.csv` | 663 | `bf3a8a49b1011ec98ada24524676fd9bfe8ae349eb19b06d933e75f23f9c84a5` |
| `zip/GCI_HomeCredit_BestSolution.zip!/README_results.txt` | 423 | `a0fe53c7035a5a2db78b961356725f6dc68bac2f4815b6353c3862db4f389829` |
| `zip/gci_peer_ablation_round2.zip!/peer_ablation_results.csv` | 990 | `196c40b05055e673a5cf273ead6465896e80724c7ea4b0aa5f2909ac3a1ca9b4` |
| `zip/gci_peer_ablation_round2.zip!/peer_ablation_summary.csv` | 1087 | `43372c8f8c5d71389c0cc012a429e3952837fde4c67b1c1b8e829631074adb5b` |
| `zip/gci_peer_ablation_round2.zip!/PEER_ABLATION_REPORT.md` | 3178 | `e03c503174460ed01a9bc1d052028fcf9efcedbb782237797e50ae1f027eb97f` |
| `zip/GCI_v18_alt_direction_package (1).zip!/GCI_v18_alt_direction/v18_progress.csv` | 204 | `080855f01cff51788b6d3e05b75b300b2bad40930fcac196ebb96238f42dff66` |
| `zip/GCI_v18_alt_direction_package (1).zip!/GCI_v18_alt_direction/v18_fast_progress.csv` | 318 | `986ed53ccafe17933e3372e45260d834add40d77be9b6ff4a279d5e05e58fb2e` |
| `zip/GCI_v18_alt_direction_package (1).zip!/GCI_v18_alt_direction/v18_fast_notes.json` | 624 | `e5a1639891264e012fcb583409645b2508b7469a0b37f13120ce8b11e5745779` |
| `zip/GCI_v18_alt_direction_package (1).zip!/GCI_v18_alt_direction/run_v18_independent_confirmation.py` | 7029 | `641ab77a40feb694b7bcebe43623177c563c632f722cf87d414fbb67bf0d308f` |
| `zip/GCI_v18_alt_direction_package (1).zip!/GCI_v18_alt_direction/v18_fast_summary.csv` | 368 | `62b4e7de125ecef02618711a9f903d1879e19f7860c137097e87fff2147992ed` |
| `zip/GCI_v18_alt_direction_package (1).zip!/GCI_v18_alt_direction/v18_fold_comparison.csv` | 322 | `b927ea972479b004f0b0360a996b4216c9261a65f4c2482d19504398571c86ac` |
| `zip/GCI_v18_alt_direction_package (1).zip!/GCI_v18_alt_direction/v18_fast_oof.npz` | 1255004 | `525c080a559504ecad6610165fd522ceca3463d8ff560be872dc5872861431d8` |
| `zip/GCI_v18_alt_direction_package (1).zip!/GCI_v18_alt_direction/run_v18_fast_confirmation.py` | 7049 | `74dacf3c7beb64d957d80bca616db9f1f427754196f83b2de6bea6e95b695467` |
| `zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/predictions.npz` | 3054007 | `1a91a55313eb3cfd5df5030e02208176809c95ea5e31e06a84a4f2086b3625a0` |
| `zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/GCI_v22_tariff_bureau_combo.ipynb` | 2208 | `6d3a7a97d4103524589a3668e79779f7dcbc9af234dc2a7f703d404a3d8b88c2` |
| `zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/metrics.json` | 2218 | `b1d9e9c14d521fe32b216c0cabf8e56d1a7ee41cd3f4b45c9fa718554974fba9` |
| `zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/GCI_submission_v22_tariff_bureau_combo_10pct.csv` | 1612356 | `ade1a2520d7c0109932c34dc9e084246cb2921357f5b864b6945a3cbc4356f9b` |
| `zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/combined_fold_results.csv` | 442 | `bf4526ab4a3e2e30196be29dc7564061d0158f54728cefa0b33ecffdc237dfe6` |
| `zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/run_gci_v22_combo_screen.py` | 5040 | `d3109933a0d26a2e7f8de07afc6d04427539f91714f8f4bcec1be434e88739e5` |
| `zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/run_gci_v22_combo_screen_seed2.py` | 5046 | `08f1f558cb0fa0db184b83533cdae48b06eccfc9b00959f899f76564d3a0eeea` |
| `zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/fold_results.csv` | 247 | `4c75006e1851a0eb00341f58453737725933b1b42ff3b1352b6276cca3eb1db8` |
| `zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/GCI_v22_research_report.md` | 1974 | `4510b6bd9d95d21fd8a5a1cab297e31e3e56f81d75bf2b9fbb188d8d8e358238` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk1/results.csv` | 1129 | `697eb34a813ecb2ac42e13cc914df9defd0f60b18d011354f6eb69e0567017b0` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk1/metrics.json` | 2668 | `96bad1aebd23f4410623a403ecc1bd720202927c6c55e07678dd814432c146b2` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk1/run.log` | 3595 | `9cefa1362819281f833af335ad4ab3f2cc5dba02be642c8627611ab115de26fa` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk1/oof.npz` | 6492095 | `9696daed809b7d707339514705ef506753cb3681e1621a21a7eabeedd48d2a7b` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk2/results.csv` | 1373 | `5c68a669f9b3adf299bca0d6bf112a31320320f9258d1888f863fe142e4cdd1b` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk2/metrics.json` | 3198 | `7315fd5cf4942d1911a600d7064462633b7838bc1361d6b3bdbccd97563130ca` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk2/run.log` | 1235 | `c0b018b95fe1840ba2585e05e04ed6dfd284759f62e85dacd4cab391568f1d3b` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk2/oof.npz` | 3903998 | `54a39bf8be64193310108ff62e596bb203926f75c52a4237e7b4e3071c8f7fea` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_combo/results.csv` | 1114 | `ca70ce1a894ee8c4fd01d16420e452b3f0cdcc15c0392ebc9bd6ed343d30233c` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_combo/metrics.json` | 2653 | `e203c2d18b2d58b1a7102c5f99136b3177b04f67ce192325e555c6127fc6a7c2` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_combo/run.log` | 3569 | `98bacf25f8b619fa7150e353b92a7de80650bd9ec09e10bf91ca794a6bb6a7b2` |
| `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_combo/oof.npz` | 6491429 | `11c6fbf74d85b6cd32546115fb4247abef6fc8a3ba771ad6d9aa81e3ada64976` |
| `zip/GCI_v24_feature_family_results.zip!/run_gci_v24_feature_family_benchmark.py` | 8640 | `0a01e8d1207aa10afe2723da72185f14a19f96ce080dcefa272d852bde107590` |
| `zip/GCI_v24_feature_family_results.zip!/run_gci_v24_chunk.py` | 8912 | `02e20558abf1a44273667d1de110eb971dbef1f1e12a104fd31b977b8a0d3fa2` |
| `zip/GCI_v24_feature_family_results.zip!/run_gci_v24_combo.py` | 8955 | `ccc2727bd6e14ed302754a8543a323b33e111e7703b6d5134a0f3c000dc4357b` |
| `zip/GCI_v26_optimized_five_research (1).zip!/REPORT.md` | 3347 | `f88ce12905751f77e408141e357fd5df05fb9c3c0b2acf2ac0a059c6053d5dfd` |
| `zip/GCI_v26_optimized_five_research (1).zip!/metrics.json` | 1851 | `b966a82a0cea07b69c0d5e3d5b672f6ea9fc93030d12770c33e81234e4a45c92` |
| `zip/GCI_v26_optimized_five_research (1).zip!/results.csv` | 781 | `3f98ea7f4abad074e628a2f90702ef090811f311e7d2c121ba6e5115d5c35cd6` |
| `zip/GCI_v26_optimized_five_research (1).zip!/oof.npz` | 7790611 | `523d5eb7938dac59e82abaae95cdde958a45f6a04694009bd68a1b9f60e01d8c` |
| `zip/GCI_v26_optimized_five_research (1).zip!/checkpoint_oof.npz` | 7790611 | `523d5eb7938dac59e82abaae95cdde958a45f6a04694009bd68a1b9f60e01d8c` |
| `zip/GCI_v26_optimized_five_research (1).zip!/run_gci_v26_optimize_five.py` | 9623 | `0dc5ce7939eab2632987fddbef0b94968c6c64a3dc5042059fd5bc4b9147ef4a` |
| `zip/GCI_v26_optimized_five_research (1).zip!/GCI_v26_optimize_five.log` | 2773 | `cceed02ae8ffedfe5ec649a59222f60e31b74f2a96e6646410a6ffd481803077` |
| `zip/GCI_v26_optimized_five_research (1).zip!/REPORT.md` | 1723 | `f88ce12905751f77e408141e357fd5df05fb9c3c0b2acf2ac0a059c6053d5dfd` |
| `zip/GCI_v26_optimized_five_research (1).zip!/five_hypothesis_summary.csv` | 688 | `eb99b74532c6a20fc160284134b935a9ca912861c263232ed0b97a0de4a7251f` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/oof_combination_followup.npz` | 1934741 | `4e5982e3d1a5b14db060b9951daa4e386ebb481a8e599bd3368857c1e47683c8` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/GCI_v29_fresh_confirmation.log` | 2321 | `650b0049569daf28bc7b91144771879ac299d823c04783bce76b80f2ae7afe54` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/metrics_seed20261032.json` | 10257 | `60ee93deb46a5b9fcf22793059676c15b0ccaebab71ae2d51c72d86d3d393027` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/run_gci_v29_fresh_confirmation.py` | 11130 | `217e323c32b888a9ab24de4b55611736d185f290a2c73dfe90cc42e084c3b402` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/REPORT.md` | 5631 | `abe5c7fe32c0bfdf248715814530fd7236c3dbb64000c566410107114a365d3d` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/results_combination_followup.csv` | 1077 | `03f09e2d5f975f6e9584d36f4ad3a2646f2887d7f528f56e5a2cc9965a5cb522` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/oof_seed20261032.npz` | 4402094 | `555c7333d79f9d5a18f3d8d60c9441a2b63b058e051bf6f342308cd060ca34d5` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/oof_seed20261027_confirmation.npz` | 6496458 | `54e93c443cd96fc9d271b825eca1b2988c1719be4476f82344fc2e6234b6d001` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/metrics_seed20261027_confirmation.json` | 1556 | `5232a585a2856b22f0364bc1f30f60e5359601d06cba6fd43d422ad78918a285` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/results_seed20261027_confirmation.csv` | 625 | `3224f38c85ccfd7f4c7f3f1f975cae24369f91d908ef5349321d87dd81bbe5c1` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/run_gci_v29_five_data_mechanisms.py` | 9572 | `395867565f46dc60925b5fa8c4abc09d1d51f642324e66d211a360656ede9d3b` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/run_gci_v29_combo_followup.py` | 9706 | `9ebf9dc87023c4bd492ef671d5f627fcdd002cb3de579df25d88fddc3b729bf5` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/GCI_v29_combo_followup.log` | 2521 | `73e47ce786b7310bfec49f88fe699caad03eb0ae0bae25ed1fe9dd40325c5a5b` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/GCI_v29_five_data_mechanisms.log` | 11384 | `6c7c429e2ca05a804f5fecc55d3dbc7c98e9e91203020779494a5f32784a67b2` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/metrics_combination_followup.json` | 2278 | `0bcd5f17ad90e7785d019f93ae05d650942488e385763ece2341ec28efeacfc3` |
| `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/results_seed20261032.csv` | 4943 | `6c628ab9280e0b85a609deb4b44d8a071bda58b63e65fe2a04c54434070130da` |
| `zip/interim_feature_research_evidence.zip!/INTERIM_FINDINGS_REPORT.md` | 14064 | `d7878e561904d73b0a724cb56cc0de86172d6f3e20403e1a179b900054cb0193` |
| `zip/interim_feature_research_evidence.zip!/consolidated_feature_experiment_results.csv` | 3280 | `258e5c31970b23c641eccd6d2807534ca7daead572d51e9400081df5cf8d65ab` |
| `zip/interim_feature_research_evidence.zip!/domain_feature_resume_results.csv` | 1648 | `6e027a1807dc440fe3c1fd5470f9722e056d37672ea8191bbbef0dbd6a741a2e` |
| `zip/interim_feature_research_evidence.zip!/feature_importance_with_definitions.csv` | 21480 | `48a94930308c3c3fd7e82bbe08a92770fa89f34c08b742f4e04f23efd66b3506` |
| `zip/interim_feature_research_evidence.zip!/feature_importance_stability.csv` | 13065 | `814336c8652e6c08ca894ba1bc8f342f8ca3d5b07e936fd4443e86d996f121bf` |
| `zip/interim_feature_research_evidence.zip!/exploratory_patterns_min1000.csv` | 12193 | `fa614c0b1869ee66e725398f13142bd934fe8be48d4de8eda6ef603719b57bad` |
| `zip/interim_feature_research_evidence.zip!/nested_selected_top8_results.csv` | 927 | `5743516ad5a8bc47c6e50b484fc27d0e7863cda1a5f71a54faf2dd40b7d96c64` |
| `zip/interim_feature_research_evidence.zip!/targeted_domain_optimization_summary.csv` | 727 | `6275a20d57a5a14f8edbbb236d8119099d87ec2b6fe0968c0a03cfdc6b9703bc` |
| `zip/interim_feature_research_evidence.zip!/core5_cv_raw.csv` | 467 | `5c3547f613385e9e8476cec90d5de6f4eb129a2d2ef2c07400a6a689ed9bf65f` |
| `zip/Telecom_Churn_Final_Research_Package.zip!/Telecom_Churn_Final_Research_Experiment.ipynb` | 12026 | `0081f3d21497b5e8127d821939e8f839ca74ae6df5d1d944853f47a845c8d785` |
| `zip/Telecom_Churn_Final_Research_Package.zip!/Telecom_Churn_Final_Research_Experiment.py` | 9666 | `67c04b8768859681f2d8364e73b1e17d949842fa7689bc6ece6a1ba360683746` |
| `zip/Telecom_Churn_Final_Research_Package.zip!/Telecom_Churn_Final_Research_Report.md` | 6861 | `f556990f41bd9fb619e033fe4d9d7b3983c34eb4cd2b3178b4c1e25a8550761a` |
| `zip/Telecom_Churn_Final_Research_Package.zip!/Deep_Experiment3_Fold_Results.csv` | 695 | `36cd97bb23924a5d88f56c5ddab953684776b7bdb97ce44ba8a91e88a1d40258` |
| `zip/Telecom_Churn_Final_Research_Package.zip!/Deep_Experiment3_Summary.csv` | 328 | `533130128d6a81a51be3748de4f126e6ae94e997add6b0a72523ac77e9e3f33a` |
| `zip/Telecom_Churn_Final_Research_Package.zip!/Deep_Experiment3_Blend_Results.csv` | 712 | `064de3d03c4aa67dbe153f58e4053b8cc2d2747006bea78a0f5b4ffc4d401ebb` |
| `zip/Telecom_Churn_Final_Research_Package.zip!/Deep_Experiment3_Paired_Bootstrap.csv` | 271 | `9e5f18c04e0a4cb3732b9521225455243e59fb17a27d17272834c507307330ad` |
| `zip/Telecom_Churn_Final_Research_Package.zip!/Deep_Experiment3_OOF_Predictions.csv` | 4030816 | `45caacf9ff2a32dbf9655421eb6a4c730e4dc2890bd50dd24781fffcd54e0831` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/ML_Experiment_Report.md` | 8136 | `763e2074bb0a6f2ddb0b8e1199e47acfbb77533687a6c920c3a302091ec4e242` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/README.txt` | 1013 | `acca52bbecbbf772271ab01c947e9187842c31031687af0d3b4efa08db89f8c8` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/code/fast_cat.py` | 2049 | `61741dc736c670c6f9e841b4fd05aec397be453699a19d1547c4ecc65f79333b` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/code/feature_select_experiments.py` | 2734 | `6d98b10051efa5b001412f4ed4a1161c6d3a228d4d56bbaf8901ad33667f4fbe` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/code/optuna_lgbm.py` | 2877 | `c7fbc6b6728264d70620f9d02c215c43fbf77c1ab8a309fdadd38756794cbf87` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/code/xgb_extended.py` | 2202 | `e4e1caa117bdaddd59312ba9d375ee5acfad40e881cf8ac85b2b0ea4e2a20c88` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/code/lgbm_auc_earlystop.py` | 2087 | `0bc792f127bd10a9e07221f6a40f72c2c432d4bf496a1f3ed6d77e4bce934ce6` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/code/engineer_lgb.py` | 3777 | `d764718c360e4fbf3c14725cc7026c404324063f3c94baa709fcb9de4c7f53bd` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/code/reproduce_3fold_cv.py` | 3614 | `217885dca95cb59029ab91461aaaaebabf4671e0ddf2c69aa5db3a73df449fac` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/code/fast_xgb.py` | 2299 | `1ce8338b6cd1a0e0a187677360c11bcc6c98bef18a489203df13d228d0d1f9cf` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/xgb_fast_results_raw.jsonl` | 713 | `defa264f9c7fadc5e16484840b8709bcbd4a209eeca705821ee028a8df89f33b` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/heldout_comparison_exploratory.csv` | 924 | `2d4b7d2f838c10ae88c409d16ea1b810cd955dff524427fa0bf523d7c7396965` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/lgbm_fast_results_raw.jsonl` | 470 | `e5ff0759ba4746310865d88045aa11cdf3075d2dea4b2641d5fe025b11e7959c` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/diagnostic_id_ethnicity.json` | 224 | `0a4abe059a80720336c937007fc83da40556c991654f820a57d80e0e7ce6fdfd` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/feature_select.json` | 3900 | `3e2d1ac8415ea45042782214caf8c7ec70536355adbf6f4fb32465f03c06f612` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/final_holdout_result.json` | 1044 | `d25272daa42a1a3eefc18cc39ebc3699da81da69c7f0d044f0b3ee27a8e66d0c` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/feature_selection_and_engineering.csv` | 632 | `dd2ee9004d4be6d66c24ac09ac2eda34d94c8c37dcf7fbd18a67828aaf915f0c` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/cat_fast_results_raw.jsonl` | 454 | `af9d6d5aa8fa85f3d2c81fec0f709dd35d7b8d20cb331c0c7c9cd7b3d3c7cd70` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/risk_decile_business_metrics.csv` | 362 | `112dfc5786737f6db525fe0a163cf00d03bd1e41f27fcd2e0df0c9f127b2c37b` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/lightgbm_feature_importance.csv` | 3183 | `adb24dc43a849bbb95b0324bd539a0716317d96a0750eeb9b5eb19b75d430264` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/lgb_auc_earlystop_raw.jsonl` | 1018 | `9dfc1d4b078fd02107e4686ce7e9c506f53578bd12f2b7c0870916743627679e` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/xgb_extended_raw.jsonl` | 348 | `108fa6078db0d9b287fcbccf869fdd93bbd42501ffffffbed2822e2038c979d5` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/lightgbm_3fold_oof_result.json` | 833 | `8848a8932143843a9731c46de783d288de26fdda10aaf43fbfc0560481b0265b` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/cat_higher_raw.jsonl` | 388 | `e015bcd25951241dc165395bf63aef14421baae7577c123af04b7f89780bcdcc` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/engineered_lgb_result.json` | 260 | `e2323479bb9b4f071f801f52e7d56a1ff6e12df44d3649ccbf4dea5feb6429d5` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/ensemble_validation_variants.csv` | 882 | `f454cba10955f2245153270231bd9836b4bbdf7a0570eae9171cb82a7d542d6c` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/data_audit.json` | 382 | `ee9f67351ffdff180c9bc50478fd33b49fd2fe915d71052cb82267a6f43996fa` |
| `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/validation_model_comparison.csv` | 1517 | `80b1f506856703b6e6dd3a323b85da001c5ea02c25d21f35b348f1d3ec533942` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/README.txt` | 818 | `ef64d77528f5b2504ba02d6a3c3df4b3bd815166dae4fe4d5de846631cc0f325` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/FINAL_FINDINGS.md` | 9718 | `ac3fc2056bfa61e6edb276efb7481a1b2a3541de2babf7dee238d80dcccd650c` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/code/final_retest_holdout_fast.py` | 2818 | `c652cf33dc28bae985441964dce04fee72ece65352b70d5be32efb45c2edb1c8` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/code/mrmr_models_fast_partial_execution.py` | 6733 | `fdd39f83cbb64811158f06449d66fba08cf18d03fa3edf3cfd6e1c5a09e810c2` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/code/rotation_forest_partial_execution.py` | 3778 | `e54b5eb467574a4573ac09596fcdfa5a80f24c5aa988f14101e05499b70c24bc` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/code/kernel_neural_small_partial_execution.py` | 5592 | `95fb71ef4ee2257c1c8210104c5d4ddaae892c85e60c93b97701320bf73c23ef` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/code/missingness_strategies.py` | 2626 | `1e8602c05dce58ebf15ea92b7890ce8fc4e5d72da502311f970995c9c0f184fb` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/code/rotboost_only_approximation.py` | 3230 | `f034d5ca709810f207f4a014aa99f83f0c0c4fee66ba0ecb5b7e1881617e7c2f` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/code/audit_telecom.py` | 2634 | `8088fc9e01a1b37b2a26943bf643f54b7abfb3b495edba7852190b476da4e5d1` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/code/optuna_retest_12_trials.py` | 2955 | `1d19f9665a42f4b859235861886c59e8c9457b4e8029f95d630de0a7f9e8b3fe` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/code/xgb_native_cat_partial_execution.py` | 2400 | `660524bc4220289ae23571cb4824b1cf71bdfe0793e43a524bda1f4733a6478b` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/code/id_sensitivity.py` | 1714 | `58ffbcc5cc9c8b9c6fcf2585d45715d0d8723b5a4bfffece27c86c1fb2ccba22` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/rotboost_only_result.json` | 384 | `0e8f326daff3fbb7c11e430674941d43a3b6283d9b3c27284556420fce3a953c` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/previous_ensemble_validation.csv` | 882 | `f454cba10955f2245153270231bd9836b4bbdf7a0570eae9171cb82a7d542d6c` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/data_and_hash_audit.json` | 461 | `a654288546eb4381d977fa3291400212e354b641cd1c13d05d34c19a49febc29` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/mrmr_approx_rank.csv` | 3884 | `606e2e6e0b8d2ca70d6d6c411478e821ab6d89c54d2eed50ffc8df460e30c8fa` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/kernel_neural_partial_execution.log` | 1665 | `796268b872004036bd3da21327973dd43c38fdf1015b22c9de83cc73b605d3f1` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/previous_heldout_comparison.csv` | 1112 | `5e6d10b24d85a46772f7b6d1088ab191434191636a27b67b584084051bdc282a` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/id_sensitivity.jsonl` | 328 | `ff8cf4687d4266926c88ce19ecbcbb311a63ac1e45e97bf703e96d734888f0ee` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/xgb_native_cat_partial_execution.log` | 351 | `1e2049578cd12ba9bbdad1c4f2ba32bfffe439eed104b3909ccb7fb31a68fc40` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/missingness_strategy_results.csv` | 468 | `98614d94bab64275466ea9497febe595c837e2dc2d03267689b85863b70ca531` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/rotation_forest_partial_execution.log` | 961 | `e9a0336653323b3609dc8d267b136673f1c647a7093c68bf5ab302d79291a1bc` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/consolidated_experiment_results.csv` | 4719 | `4161b2b3318d1d135b6414b3671e964ce29c30e648ca428a2b4dacbc3d5e070b` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/optuna_retest_partial_3_of_12.log` | 1858 | `0af9a0ea2aecaceb4383a75398057610d3b5145585502bd18a467ebd1b9e0e71` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/data_audit.txt` | 4559 | `5e3de56bf532d587c6d4a7b7e31d902c3ea4281753193cbd82c9a7e0f226e933` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/previous_feature_selection_results.json` | 3900 | `3e2d1ac8415ea45042782214caf8c7ec70536355adbf6f4fb32465f03c06f612` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/final_retest_holdout_fast.json` | 861 | `061afa8ec6473401be4391db5d1ec5da88766007f4077bd8ffbac124e0f2a48d` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/previous_3fold_cv_result.json` | 833 | `8848a8932143843a9731c46de783d288de26fdda10aaf43fbfc0560481b0265b` |
| `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/references/SOURCES.md` | 1900 | `f38ba248abab318bc8c678e7a044950ea52335ec6454d0bb4699f90734a5b744` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/V10_MORNING_TO_NOW_RESEARCH_REPORT.md` | 27109 | `eadcbb7ef5f0f8b8c057948463de26709ad8e0604a318a9dc36af4cfd3760048` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/Right_Customer_Right_Offer_Right_Time_v10.ipynb` | 125470 | `a0bb9f5595fc042902e349390ecc6759a4c75524b9c6b76eb682d5b11b5641f8` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/Right_Customer_Right_Offer_Right_Time_v10_2_Financial_Feature_Pilot_Workbook.xlsx` | 134454 | `21f9e2eefe4ea0fb91dcae72903133b969bf8468a4926a82d74ae4dd00f560f7` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/v10_feature_engineering_patch.py` | 4391 | `487f2d8f9984032c3216c3fca9ed03c6d8af42df69a8104c5f77133f2a8eb774` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/association_risk_diff_development_only.csv` | 1317 | `19310546a6fd3231f61b6c4f0551b073b84b1f459868fe5d54f7e87d1c54ca20` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/adjusted_no_id_odds_ratios.csv` | 3472 | `64faddfebaf97141b89eaf9179837c549e598a42d859d8b73762367990120666` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/adjusted_plus_id_decile_diagnostic_only_odds_ratios.csv` | 3846 | `7d37f50c99eb205515725d55f046c25c63277e74b56837063973dc45e1494062` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/area_churn_missingness_development_only.csv` | 1265 | `9d06984be74a31e92bd5d1a1c695416fe7779ea297bae041b5ed637e75fa5a51` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/id_decile_churn_missingness_diagnostic_only.csv` | 268 | `f90a21d11ec6b5c611b0c64d498e8466a2f9f352ee86797b5c33bb17d5d06e10` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/zero_voice_state_breakdown_dev.csv` | 441 | `b768516676ff645ffe7eddf73716d906094695d8675e2eed269788316e295dea` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/revenue_semantic_reconciliation.json` | 485 | `1c5eb3313b2123b28b39093aac4f21963ef1fbfd66ad0f959aec45da2f4c450a` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/v10_feature_bug_impact.json` | 245 | `16215a84d565b7461465d008cf3d91a5cf8def204c709d82089557715c64c062` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/v10_fixed_policy_horizon_sensitivity.csv` | 1114 | `014e42fba06248faed523f68e4854c59ead762777a7cbddc1f89a096b115db2b` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/v10_2_fixed_policy_sensitivity_162.csv` | 40911 | `75be7f0f5d0a3334d390dc9ee253426a7e673dcfb5f9953fc9c7e4883f923913` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/v10_2_break_even_sensitivity.csv` | 3734 | `60079f591751fd80a1fde735967981c1337a64a0e9db65a54442e41875662613` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/v10_2_fixed_policy_sensitivity.png` | 168228 | `52094fed1210ddb12d0f393a23e3cbd78f4e70e69ddce0d45f997c78b5fa8dd6` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/development_only_feature_associations.csv` | 13314 | `8fc159770f46f398e2c9ee079721b13c88832774b6903c5d357d5383d3395a57` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/catalogue_and_data_quality_summary.csv` | 487 | `6055107006eda731d585f6e3fb7b5c9c501ec851ec42a4d862a3856f2fa538bf` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/data_quality_focus.csv` | 845 | `dd5563f371de6391b71f579fdddf9cf50c2fdf0514ccfc1fe327868b99d3afa7` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/financial_horizon_sensitivity_fixed_policy.csv` | 724 | `845d39599df9c29ff34d894764f553dbe55d06c6874784d4787c40b989b2f946` |
| `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/data_overview_extract.txt` | 6134 | `4a6bf9c78d79ee2e737b4e4f556ef2de453c366b121d91d568599241236248bc` |


# ARCHIVE CODE AND NOTEBOOK CATALOG

## Reconciled scope

I recursively read members of the 39 nested ZIP paths using `zipfile`/`BytesIO`; I did not extract or execute archive code. After collapsing byte-identical archive copies, the count is **128 member paths: 70 notebooks and 58 Python files**—the expected 63+23 root members plus 7+35 nested additions. Some files are themselves identical across distinct paths, so the 128 paths represent **119 distinct member payloads**. Identical paths are grouped as aliases below.

All 128 paths decoded and all notebooks parsed. The notebooks contain **68 saved output records**, in seven notebooks; the remaining files’ metrics are source/Markdown claims or absent. **“No score” below means no numeric score found in source/Markdown or saved output**, not that the model was run and scored zero. Runs were not executed for this audit.

## Per-file catalog

**Key for evidence:** `logged` = present in a notebook’s embedded output; `claim` = Markdown or source text; `none` = no numeric result found. GCI notebooks commonly use five-fold stratified OOF and local prediction caches; individual entries specify exceptions or more specific validation evidence. Colab/Drive and external CSV/NPY/NPZ dependencies are noted where visible. Absent saved outputs do not verify a claimed score or cache provenance.

### GCI / Home Credit experiments and submissions

1. **`zip.zip/zip/Churn_Optimisation_Decision_Package.zip/Telecom_Churn_Final_Research_Experiment.ipynb`**; alias: `zip.zip/zip/Telecom_Churn_Final_Research_Package.zip/Telecom_Churn_Final_Research_Experiment.ipynb` — Telecom churn feature ablation: corrected compact hybrid versus added service-quality/lifecycle/account features; LightGBM, fixed three-fold stratified comparison, training-fold-only catalogue transforms, paired bootstrap. **Claim:** baseline ~0.7058 CV AUC; no output. Needs `Client.csv`, `Record.csv`; CPU/data environment unspecified.
2. **`zip.zip/zip/CompanyA_Final_Submission_Package.zip/CompanyA_Reproducible_Analysis.ipynb`**; aliases: `zip.zip/zip/CompanyA_Omnicampus_Submission_Bundle.zip/CompanyA_Reproducible_Analysis.ipynb` and identical `(1)`–`(6)` bundle paths — Reproducible Company A churn, risk ranking and retention pilot; domain features and boosted classifier. No logged score; notebook describes fold-based held-out evaluation, but no run output. Requires Company A data and notebook dependencies; ID-derived signup ordering is an assumption to disclose.
3. **`zip.zip/zip/gci_0_762337_credit_term.py`** — XGBoost experiment adding `CREDIT_TERM = AMT_CREDIT / AMT_ANNUITY`; local OOF/holdout AUC **~0.762337, claim**, explicitly not the public 0.76370 champion. Validation details are in script; CSV inputs under configured `input`; no GPU evidence.
4. **`zip.zip/zip/gci_0_76370_reconstruction.py`** — XGBoost reconstruction of public champion feature/model family; **0.76370 public leaderboard, claim**, not an embedded validation output. Requires train/test/sample CSVs in configured input; no saved execution evidence or GPU indication.
5. **`zip.zip/zip/gci_2ca_fingerprint_submission.py`** — LightGBM submission with financial ratios (credit/income, annuity/income, credit/annuity, goods/credit) and fingerprint-like groups; rank-space prediction/output. No score or explicit CV result found; searches external prediction/submission paths, so provenance matters. No GPU evidence.
6. **`zip.zip/zip/GCI_CatBoost_Product_Experiment_Package.zip/run_gci_catboost_product_experiment.py`** — Tests CatBoost member on decoded product keys atop v9, with rank blending; fixed-round LightGBM v9 baseline and stratified folds. No numeric result recorded in source; expects v9/cache predictions and competition inputs. CPU/Colab context, not a confirmed device.
7. **`zip.zip/zip/GCI_CatBoost_Product_Experiment_Package.zip/run_v9_full_oof_for_catboost_optimized.py`** — Memory-optimized OOF-only reproduction of v9, preserving feature logic, folds, target encodings, rounds and weights; HGB used in reconstructed pipeline. No score claim/output; requires exact training data and reproducible fold/config setup; no GPU evidence.
8. **`zip.zip/zip/GCI_catboost_product_step_colab.ipynb`** — CatBoost depth-7, 900-iteration product-key member and fixed 10% rank blend on v9; five-fold comparison. Markdown describes a pre-registered gain/fold/bootstrap promotion rule; **no saved output**. Depends on Drive `gci_cache_final` and prior v9 predictions; CPU claimed in Markdown.
9. **`zip.zip/zip/GCI_CA_EXT_promising_research.zip/gci_promising_ca_ext_test.py`** — LightGBM test of CA fingerprints and conditional EXT-score quantile groups with nested target encoding; three-fold stratified CV. No exact score found; requires competition CSVs, sklearn/LightGBM, output JSON; no GPU evidence.
10. **`zip.zip/zip/gci_CA_fingerprint_v2.py`** — CA fingerprint rank-space submission built from competition features and cached champion predictions; not an estimator-training validation script. No numeric score; depends on `/mnt/data`-style files/current champion; no device claim.
11. **`zip.zip/zip/GCI_Competition_Candidates_Package.zip/gci_submission_notebook.ipynb`** — Home Credit reproducible submission using supplied data and deterministic domain feature engineering; LightGBM/XGBoost candidate models. No saved outputs or exact score identified; data-only claim, but package/runtime dependencies needed; no GPU evidence.
12. **`zip.zip/zip/gci_DE_SENT_CA_residual_submission.py`** — Narrow employment-sentinel/CA residual added to Submission-15, rank blend **95% champion + 5% residual**; champion public AUC **0.76794, claim**. No validation score logged in script; needs prior champion/input CSV and feature data; leakage safety is claimed, not independently established.
13. **`zip.zip/zip/gci_exact_rate_ext_affordability_submission.py`** — LightGBM submission with exact amortization/rate and EXT affordability ratios. No score found; intended to use train/test/sample files via `GCI_INPUT`; dependency on LightGBM and input folder, CPU configuration not explicit.
14. **`zip.zip/zip/GCI_final_pick_bootstrap_cell.py`** — Paired stratified bootstrap comparing v9 candidates against v6_B on pooled held-out predictions; AUC rank calculation, no new model/features. Prints candidate bootstrap results at runtime but no saved score; requires Drive caches `gci_cache_repro`, `gci_cache_v6` etc.; CPU.
15. **`zip.zip/zip/GCI_final_pick_bootstrap_colab.ipynb`**; alias: `zip.zip/zip/GCI_final_pick_bootstrap_colab (1).ipynb` — Same bootstrap pick workflow as 14; held-out paired resampling, no model retraining. No embedded output; requires Drive prediction caches and labels; no GPU requirement stated.
16. **`zip.zip/zip/GCI_final_rebuild_v15_cell.py`** — Cache-only v15 blend reconstruction: champion formula combines ranks, including 0.75 B / 0.25 E1Aet, then 0.90 blend and 0.10 branch. Rebuild, not fresh validation; public **0.77891, claim** in referenced source file. Requires `gci_cache_final`; no GPU.
17. **`zip.zip/zip/GCI_final_reproduction.ipynb`** — Rebuilds v3_A submission with fixed blend `0.24 qf_shallow + 0.12 qf_reg + 0.32 v1_allfam + 0.16 v2_allfam + 0.16 v2_mono`; **public AUC 0.77746, Markdown claim**. Five-fold OOF/cached model workflow; no saved output. Requires CSVs and model caches; Colab CPU-oriented.
18. **`zip.zip/zip/GCI_final_reproduction_v9.ipynb`** — Rebuilds v9 E1 multiview submission as 50% OLD + 50% NEW blend; **public 0.77865, Markdown claim**. Five-fold OOF code, no saved output; depends on competition CSVs and `gci_cache_final`; CPU/Colab, no GPU evidence.
19. **`zip.zip/zip/GCI_FINAL_v15_end_to_end_colab.ipynb`** — End-to-end v15 champion reconstruction, rank-space consistent views plus cached HGB/product branch; **public 0.77891, Markdown claim**. Five-fold stratified training, no saved outputs. Requires `GCI_hgb_product_test.npy`, OOF companion, Drive cache and CSVs; no confirmed GPU.
20. **`zip.zip/zip/gci_financial_fingerprint_submission.py`** — Final rank blend: 93% champion + 7% hierarchical financial-fingerprint rank, using train TARGET/features and test/sample. No explicit validation score; depends on champion CSV; no GPU evidence.
21. **`zip.zip/zip/gci_highres_lgb_submission.py`** — High-resolution LightGBM (`num_leaves=15`, `max_bin=511`, `learning_rate=.025`, `num_threads=10`) and rank output. No score or CV result stated; hard-coded `/mnt/data` paths and champion/fingerprint work files; CPU threading configured.
22. **`zip.zip/zip/GCI_HomeCredit_BestSolution.zip/competition_solution.py`** — Home Credit feature engineering and model comparison across LightGBM, CatBoost, XGBoost; stratified 80/20 holdout (`random_state=42`), reports validation AUC when run. No embedded score found; uses supplied train/test only; XGBoost hist, no CUDA stated.
23. **`zip.zip/zip/GCI_product_decode_colab.ipynb`** — Three LightGBM product-decode models with cached five-fold OOF and test predictions; product keys plus q-fingerprint family. No embedded output; Markdown/cells compare against references **0.76713/0.76678/0.76746**, source reference only. Requires CSVs, prior qmodel CSV/HGB NPY and Drive cache; CPU Colab.
24. **`zip.zip/zip/GCI_product_decode_colab_v2.ipynb`** — Product-decode v2 adds `k` credit/goods tariff tier to three LightGBM members; cached five-fold OOF. References prior model values **0.76992/0.77070/0.77035, source claims**, not saved outputs. Requires product-v2 cache, prior product CSV/HGB NPY and data; CPU Colab.
25. **`zip.zip/zip/GCI_qfingerprint_product_colab (1).ipynb`** — q-fingerprint models derive rate-curve features from `q=AMT_ANNUITY/AMT_CREDIT`; smoothed nested OOF target encodings of q rounded 3/4/5 decimals, three LightGBM variants. **Source claims** held-out **0.76713/0.76678/0.76746**, average **0.76843**; no outputs. CPU/Colab; needs CSVs and optional champion.
26. **`zip.zip/zip/GCI_qfingerprint_product_colab.py`**; alias: `zip.zip/zip/GCI_qfingerprint_product_colab (1).py` — Same q-fingerprint submission implementation as 25; three one-hot/encoded LightGBM views and rank blend. Same **source-claimed** metrics, no logged output; relies on CSVs and optional champion file; no GPU evidence.
27. **`zip.zip/zip/GCI_qfingerprint_product_colab.ipynb`** — Notebook counterpart for q-fingerprint rank blend; same q-rate target encodings. Metrics are source claims, no embedded output. Needs competition files, LightGBM, optional champion and cache; CPU Colab.
28. **`zip.zip/zip/GCI_qfingerprint_product_colab_top_24.ipynb`** — q-fingerprint product-key comparison: one-hot, all-family one-hot and mono LightGBM. **Logged:** fold AUCs for one-hot `0.76804, 0.77217, 0.77470, 0.76703, 0.76792`; held-out `0.76992`; all-family held-out `0.77070`; mono `0.77035`; rank average `0.77158`. Also logs Drive mounted, two threads, 1047 products and 67/130 features. Cache/data required.
29. **`zip.zip/zip/GCI_qfingerprint_submission (1).py`**; alias: `zip.zip/zip/GCI_qfingerprint_submission.py` — q-fingerprint submission variant using nested smoothed encodings at q precision 3/4/5, three LightGBM models and rank-space blend. Source states **0.76713/0.76678/0.76746**, average **0.76843** vs **0.76458**, weight `W=0.7`; not logged. Needs champion and CSVs; no GPU evidence.
30. **`zip.zip/zip/GCI_q_fingerprint_colab.ipynb`** — Colab q-fingerprint pipeline; same three-model rank blend, optional NN off by default. **Markdown claims public 0.77283, held-out mean 0.76843 vs 0.76458**; no saved output. Installs LightGBM; CSVs and optional champion; CPU/Colab.
31. **`zip.zip/zip/gci_ratecurve_template_v1.py`** — Three LightGBM models over proven 67-feature representation plus 12 screened rate-curve features; contextual financial-template residual/rank averaging. No exact result found; train/test/sample and champion required; no device claim.
32. **`zip.zip/zip/gci_risk_gated_fingerprint_submission.py`** — Risk-gated financial fingerprint rank correction using repeated financial templates; source says broader model failed and fingerprint helps. Exact score absent; requires champion/fingerprint data and CV analysis from prior work; no GPU.
33. **`zip.zip/zip/gci_targeted_second_order_attack.py`** — XGBoost second-order interaction/targeted feature experiment; 3-fold stratified AUC, 12-thread cap, histogram trees. No exact result observed; depends on competition CSVs and XGBoost; CPU.
34. **`zip.zip/zip/gci_template_meta_local5_submission.py`** — LightGBM meta learner over cached cross-fitted predictions plus contextual financial-template and local rerank; final 50% champion/50% local-meta rank. No exact score; cached base OOF/test predictions are critical inputs; CPU.
35. **`zip.zip/zip/gci_testlike_specialist_submission.py`** — LightGBM adversarial train/test specialist, gated by train/test similarity; tests rank correlation and rank movement vs champion. No result saved; depends on champion input and train/test CSVs; CPU, four jobs.
36. **`zip.zip/zip/GCI_v10_candidate_blocks_colab.ipynb`** — Adds two gated candidate blocks on v6: extra learned-expectation/residual targets and related features. Reuses live v6 session state; intended OOF gate, no result output. Needs v6 objects/cache and Colab session; TabM/TabFM/denoising ideas explicitly unbuilt.
37. **`zip.zip/zip/GCI_v10_standalone_colab.ipynb`**; alias: `zip.zip/zip/GCI_v10_standalone_colab (1).ipynb` — Standalone v6 plus gated candidate blocks; LightGBM five-fold OOF, residual/product features and HGB cached arrays. No saved result; requires CSVs, HGB NPY and cache; Colab CPU.
38. **`zip.zip/zip/GCI_v11_family_bagging_colab (1).ipynb`** — E4 feature-family bagging: LightGBM members on different EXT/residual, group/residual, count/EXT subsets; blends into v6_B. Four folds choose blend weight, fifth scores; no saved result. Requires v6 caches and CSVs; CPU.
39. **`zip.zip/zip/GCI_v11_family_bagging_colab.ipynb`** — Same family-bagging experiment, slightly different notebook content from 38; same five-fold fold-held-out weight selection. No score output; depends on cache and Colab CPU; gains are not established in archive evidence.
40. **`zip.zip/zip/GCI_v12_product_aware_AB_colab.ipynb`** — A/B test of product-aware learned expectations, motivated by EXT-score residuals being key to earlier residual gain. LightGBM OOF comparison; no saved result. Requires v6-style caches, HGB/product NPY and CSVs; Colab CPU.
41. **`zip.zip/zip/GCI_v13_reweight_and_ET_colab_cc.ipynb`** — Reproduces v9 and tests reweighting/ExtraTrees; source claims ET gain around **+0.0010 to +0.0015/member** and a product-key comparison. No embedded output; five-fold OOF/cached pipeline; Colab CPU, lengthy ET training.
42. **`zip.zip/zip/GCI_v14_ET_deepening_colab.ipynb`** — Deepens ExtraTrees finding on noisy nested target encodings; explores rounds/ET blend. Source cites pooled AUC improvements around **0.0003** and prior per-member gains, not logged in this notebook. Five-fold, cache-heavy; CPU.
43. **`zip.zip/zip/GCI_v15_consistent_views_ET_colab.ipynb`** — Variant A consistent multi-view features + ExtraTrees; hypothesis ET helps newer residual/view members (~+0.0012) more than old members. Five-fold OOF/cache workflow, no saved output; CPU Colab.
44. **`zip.zip/zip/GCI_v16_everything_merged.ipynb`** — Merges v15 with community/repeat-applicant features; LightGBM/CatBoost and extensive ablations. Markdown says final score unmeasured and asks compare with v15 OOF **0.77404**; no logged output. Requires competition CSVs, cache and CPU Colab; broad selection/leakage risk.
45. **`zip.zip/zip/GCI_v16_everything_merged_o.ipynb`** — Output-bearing v16 counterpart. **Logged:** old/e1/e1et/e1Aet **0.77412/0.77512/0.77608/0.77618**, champion held-out **0.77649**, regime OOF **0.77648** vs fixed **0.77649**; adversarial **0.6856**, permutation **0.5014**. Five-fold, CPU/cache workflow; adversarial result signals train/test shift.
46. **`zip.zip/zip/GCI_v16_regulariser_screen_colab.ipynb`** — Screens noise-damping regularisers on one ExtraTrees member before retraining; compares to v15 champion blend. No saved metrics; five-fold/cache workflow, data/model dependencies, CPU.
47. **`zip.zip/zip/GCI_v17_catboost_E1_stack_colab.ipynb`** — Tests CatBoost on E1 residual + consistent-view feature stack, addressing previous CatBoost runs using base features only. Five-fold OOF candidate gate; no output or exact score; requires caches/CSV and CatBoost, Colab CPU.
48. **`zip.zip/zip/GCI_v17_optimised.ipynb`** — Optimized merged v15 research pipeline; includes integral target-encoding engine, person-pooled EXT averages/partner deltas, community and repeat-applicant features. Markdown says final score unmeasured; five-fold code and cache; CPU, no saved run.
49. **`zip.zip/zip/GCI_v17_optimised_O.ipynb`** — Output-bearing optimized v17 copy. **Logged:** e1/e1et/e1Aet **0.77509/0.77592/0.77604**, champion **0.77646**, nonnegative stacker **0.77657** (+0.00010); adversarial **0.6868**, permutation **0.4985**. Five-fold, cached CPU pipeline; tiny stacked gain subject to selection risk.
50. **`zip.zip/zip/GCI_v18_alt_direction_package.zip/GCI_v18_alt_direction/run_v18_independent_confirmation.py`**; identical-archive alias: same path beneath `GCI_v18_alt_direction_package (1).zip` — Independent confirmation of v18 alternative-direction rank blend, two LightGBM candidates against v15. Script-based CV; no output captured here, so no result asserted. Requires standalone data/cache setup; no GPU evidence.
51. **`zip.zip/zip/GCI_v18_alt_direction_package.zip/GCI_v18_alt_direction/run_v18_fast_confirmation.py`**; identical-archive alias beneath `(1).zip` — Faster confirmation/screen of same v18 alternative blend; reduced compute rather than full independent model. No saved score; requires competition data and LightGBM; CPU.
52. **`zip.zip/zip/GCI_v18_alt_direction_probe.ipynb`** — Two-model LightGBM rank blend with v15 champion; independent three-fold stratified split. Markdown claims pooled AUC **0.756907 vs 0.7563** for probe baseline; not logged and not comparable to full champion. Standalone feature pipeline, no cache output.
53. **`zip.zip/zip/GCI_v18_evidence_driven.ipynb`**; alias: `zip.zip/zip/GCI_v18_evidence_driven (1).ipynb` — Evidence-driven v18 upgrade, merged research/community/repeat-applicant feature pipeline. Markdown describes **public 0.78222 / held-out 0.77649** prior-run evidence and says current final score unmeasured; no embedded output. Five-fold, CPU Colab, caches/data required.
54. **`zip.zip/zip/GCI_v18_evidence_driven_o.ipynb`** — Output-bearing v18. **Logged:** champion held-out **0.77666**, stacker **0.77674** (+0.00008), adversarial **0.6868**, permutation **0.4985**; earlier blend values also logged. Same-fold rank/stack checks; requires caches/CSV; CPU.
55. **`zip.zip/zip/GCI_v18_ExtraTrees_diversity_colab.ipynb`** — Adds ExtraTrees as structurally different blend member; Markdown reports 3-fold HGB stand-in **0.76344 vs 0.76840** but low correlation rationale. No saved output. Requires cached champion/HGB data and CPU Colab; stand-in is explicitly not full model evidence.
56. **`zip.zip/zip/GCI_v19_NN_entity_embeddings_colab.ipynb`** — Neural network with decoded product entity embeddings; gated by standalone AUC around 0.76 and rank diversity. No logged output in this copy; prior network score claims only. PyTorch/embedding dependencies, GPU evidence not guaranteed; requires product and competition data.
57. **`zip.zip/zip/GCI_v19_NN_entity_embeddings_colab_0.ipynb`** — Same entity embedding experiment with outputs. **Logged:** fold AUC **0.76419, 0.76999, 0.77389, 0.76250, 0.76602**, pooled **0.76732**; champion held-out **0.77404**, public **0.77891** stated in output. Early self-test logged 0.736; cache HGB OOF was `None`. Runtime depends on NN stack/device.
58. **`zip.zip/zip/GCI_v19_research_candidate.ipynb`** — v19 research candidate integrating v15, community/repeat features and strict validation. **Logged** v16-family results including champion **0.77649**, adversarial **0.6856**, permutation **0.5014**; Markdown says candidate not score-improved until matched-fold promotion. CPU/Colab and caches required.
59. **`zip.zip/zip/GCI_v19_research_rebuilt.ipynb`** — New unexecuted v19 research rebuild based on v18 notes; explicitly says no claimed score and removes saved v18 outputs. Same-fold ablation intended; requires CSV plus `meta.json`, cache/dependencies, CPU Colab.
60. **`zip.zip/zip/GCI_v20_evidence_driven_candidate.ipynb`** — v20 candidate retaining evidence-driven v18 feature architecture; predeclared promotion gates around mean gain/fold wins. No saved score output; five-fold/cache-based; CSV/Colab CPU. Markdown references prior public 0.78222 and OOF 0.77649, not v20 results.
61. **`zip.zip/zip/GCI_v21_gated_research_candidate.ipynb`** — v21 pre-training evidence gate for candidate features; merged model pipeline, same v15 reference. No saved score; markdown says final unmeasured. Five-fold/cache dependencies; Colab CPU.
62. **`zip.zip/zip/GCI_v21_gated_research_candidate_o.ipynb`** — Output-bearing v21 variant; only saved output `hgb oof: None`, no validation score. Candidate gate code and five-fold setup; HGB NPY/cache/data missing; CPU/Colab.
63. **`zip.zip/zip/GCI_v22_research_expansion_gated.ipynb`** — v22 expands v21 candidate feature/evidence gate; no score run captured, Markdown says final unmeasured. Five-fold/cache workflow; depends on CSVs and HGB/Drive artifacts; CPU.
64. **`zip.zip/zip/GCI_v22_tariff_bureau_combo_research_package.zip/GCI_v22_tariff_bureau_combo/GCI_v22_tariff_bureau_combo.ipynb`** — Combines tariff/product and bureau-burst mechanisms; explicitly says not yet matched against v15 OOF and prior candidate did not improve public score. Held-out results are presented as research record, but no notebook outputs saved; CSV/cache required.
65. **`zip.zip/zip/GCI_v22_tariff_bureau_combo_research_package.zip/GCI_v22_tariff_bureau_combo/run_gci_v22_combo_screen.py`** — LightGBM three-fold stratified screen of tariff/bureau feature combo, computes OOF AUC and test predictions. No score output embedded; requires competition CSVs and package deps; CPU.
66. **`zip.zip/zip/GCI_v22_tariff_bureau_combo_research_package.zip/GCI_v22_tariff_bureau_combo/run_gci_v22_combo_screen_seed2.py`** — Seed-2 repeat of 65, same feature mechanisms/model and three-fold protocol; repeatability check, no saved numeric result. Needs same inputs/LightGBM; CPU.
67. **`zip.zip/zip/GCI_v24_evidence_lab.ipynb`** — Evidence lab for product-tier encoding, repeat applicants, views, ExtraTrees and other methods; predeclared screen/judge/ledger gates. No embedded results; source gives v18 champion held-out **0.77666**, public **0.78235** as reference only. Requires ledger/cache/data and CPU.
68. **`zip.zip/zip/GCI_v24_feature_family_results.zip/run_gci_v24_feature_family_benchmark.py`** — LightGBM stratified benchmark of feature-family blocks (including cross-source external-score agreement); no saved result. `/mnt/data` input/output paths and dependencies; CPU.
69. **`zip.zip/zip/GCI_v24_feature_family_results.zip/run_gci_v24_chunk.py`** — Chunked LightGBM feature-family benchmark; same AUC evaluation but chunk-specific results output. No saved scores in archive member; needs CLI chunk index, data and `/mnt/data`; CPU.
70. **`zip.zip/zip/GCI_v24_feature_family_results.zip/run_gci_v24_combo.py`** — LightGBM combinations of v24 feature-family candidates; stratified fold AUC code, no result recorded. Requires benchmark data/output directory; CPU.
71. **`zip.zip/zip/GCI_v25_block_tester.ipynb`** — Tests candidate feature blocks on actual LightGBM ExtraTrees member (not HGB stand-in), same-run baseline and paired folds. No saved result; five-fold/cache/CSV and LightGBM needed; CPU.
72. **`zip.zip/zip/GCI_v26_optimized_five_research.zip/run_gci_v26_optimize_five.py`**; identical-archive alias under `(1).zip` — Optimizes five research mechanisms, includes decile rank/qcut feature construction and LightGBM stratified folds. No result captured; requires input/cache paths and sklearn/LightGBM; CPU.
73. **`zip.zip/zip/GCI_v29_five_data_mechanisms_research.zip/GCI_v29_package/run_gci_v29_fresh_confirmation.py`** — Fresh-seed confirmation of five data mechanisms using LightGBM and three-fold stratified AUC. No saved score; data/cache deps; CPU.
74. **`zip.zip/zip/GCI_v29_five_data_mechanisms_research.zip/GCI_v29_package/run_gci_v29_five_data_mechanisms.py`** — Main five-mechanism experiment rebuilding v27 formula scaffold and loading `GCI_v27_optimized_five/oof.npz`; LightGBM. No score in source output here; explicit cache dependency and CPU.
75. **`zip.zip/zip/GCI_v29_five_data_mechanisms_research.zip/GCI_v29_package/run_gci_v29_combo_followup.py`** — Follow-up combo using same v27 formula baseline and OOF cache, LightGBM fold evaluation. No saved metric; stale/mismatched NPZ risk if cache provenance differs; CPU.
76. **`zip.zip/zip/GCI_v33_product_conditional_experiment.py`**; alias: `zip.zip/zip/GCI_v33_product_conditional_experiment (1).py` — Formula-aware product-conditional LightGBM submission experiment; stratified screen exists, but metadata explicitly says `model_auc_oof_not_computed=True`, exploratory/no direct champion comparison. Requires champion/cache/data; no GPU.
77. **`zip.zip/zip/GCI_v3_blend_colab.ipynb`** — Variance reduction: two shallow/regularized LightGBM members and OOF-selected blend; selects weights on held-out predictions and “pick on 4 folds, score 5th”. No saved score; cache/CSV and CPU Colab.
78. **`zip.zip/zip/GCI_v4_diversity_colab.ipynb`** — Adds CatBoost, XGBoost and all-family LightGBM variants for algorithm diversity, OOF blend selection. Code contains XGBoost `device="cuda"` option; no output. Requires multiple model dependencies, caches and Colab; selection over candidates risks optimism.
79. **`zip.zip/zip/GCI_v5_dropPF4_colab.ipynb`** — Ablation removes product×fee PF4 target-encoding key, keeps same five members/weights; source cites small prior +0.0004 hint, not logged result. Five-fold foldwise comparison; cache/CSV and CPU.
80. **`zip.zip/zip/GCI_v6_final_blend_cell.py`** — Greedy/nested blend selection among old/new OOF members; weight chosen on four folds and scored on fifth. No stored outcome; requires `gci_cache_repro`/`gci_cache_v6`; CPU.
81. **`zip.zip/zip/GCI_v6_residual_features_colab.ipynb`** — Adds learned-expectation residual features for 16 fields (EXT, income, credit, employment/date/region/family); LightGBM five-fold OOF. No saved score; nested encoding/data caches required; CPU.
82. **`zip.zip/zip/GCI_v7_extra_trees_colab.ipynb`** — Applies ExtraTrees LightGBM training procedure to v6 residual features / noisy target encodings; source states randomization helps and “per leaf” alternative worsened. No saved result; five-fold/cache; CPU.
83. **`zip.zip/zip/GCI_v8_pairwise_ranker_colab.ipynb`** — XGBoost `rank:pairwise` global-group ranker as blend member; early stopping on inner split of each outer train fold, promotion rule ≥4/5 held-out folds. No logged score; needs pairwise-group construction and XGBoost; CPU/hist.
84. **`zip.zip/zip/GCI_v9_all_experiments_colab.ipynb`** — E1 multi-view stacking (EXT/loan/person views), three candidate improvements vs v6_B; same held-out rows and predeclared decision rule. No saved scores; depends on v6 caches and CSV; CPU.
85. **`zip.zip/zip/GCI_v9_tabicl_context_colab.ipynb`** — TabICLv2 class-balanced context gated experiment; blend weight selected by nested CV, no public leaderboard use stated. No saved run; pretrained weights/organizer permission explicitly unresolved (`RULES_CONFIRMED=False`); GPU runtime/session required.
86. **`zip.zip/zip/GCI_v9_tabicl_standalone_colab.ipynb`** — Standalone TabICLv2 class-balanced-context version, GPU Colab stated. No score from this archive; AUC literature figures are not this experiment’s measurement. Requires pretrained weights/model package and CSVs; rules/dependency risk.
87. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time.ipynb`** — Company A targeted churn prevention, offers and timing; logistic baseline plus boosted models, ROC-AUC/PR-AUC and business ranking. No embedded outputs or numeric score; claims are placeholders. Requires company data, packages, figures path; ID-based order is inferred.
88. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time_v10.ipynb`**; alias: nested `zip.zip/zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip/Right_Customer_Right_Offer_Right_Time_v10.ipynb` — Churn/retention v10 with expanded feature and learning-curve analysis; LightGBM/CatBoost/SHAP dependencies, same-fold assessment. No output score; source suggests feature gains only ~0.002–0.008 AUC as claim/rationale, not logged. Data and optional packages required.
89. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time_v11.ipynb`** — v11 churn analysis/decision workflow, same family of domain and retention features plus reporting; no saved score. Same-fold ROC-AUC claims are not embedded measurements; requires company CSVs, CatBoost/SHAP optionally.
90. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time_v12.ipynb`** — v12 churn/offer iteration with revised feature/business reporting and learning curves; no embedded outputs or exact score. Dependencies and data as prior versions; beware inferred ID signup order.
91. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time_v13_Optimized.ipynb`** — Optimized churn/retention notebook; explicitly disables ID-order proxy for primary model, avoids three weak features and emphasizes honest feature validation. No saved score; source says no proven AUC improvement for promotions. Requires company data and notebook packages.
92. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time_v2.ipynb`** — Early v2 churn workflow; logistic/boosted comparison, optional LightGBM/CatBoost, feature engineering and ROC/PR metrics. No embedded scores; same-fold claims only. Requires company inputs, optional CatBoost install; no GPU evidence.
93. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time_v3.ipynb`** — v3 churn model iteration, domain features and business targeting evaluation; no output score. Cross-version same-fold comparison is described but no logs; optional boosted-model packages and input data needed.
94. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time_v4.ipynb`** — v4 continuation of churn model/feature and offer analysis; no saved result. Metrics appear as report/template claims; same data/dependency and ID-proxy caveats.
95. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time_v5.ipynb`** — v5 churn/retention workflow iteration; no embedded results. Same-fold metrics intended; requires Company A data and notebook packages; no established device.
96. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time_v6.ipynb`** — v6 churn analysis with expanded feature/reporting logic; no saved score. Source mentions small 0.002–0.008 AUC feature-gain expectation, not a logged result; same dependency caveats.
97. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time_v7.ipynb`** — v7 churn/retention notebook iteration; no output score. Same-fold evaluation described, no score evidence; company data/optional packages required.
98. **`zip.zip/zip/Right_Customer_Right_Offer_Right_Time_v9.ipynb`** — v9 churn model and business decision/report iteration; no saved score; feature-gain text is not measured output. Company data, plots and optional libraries required; no confirmed GPU.
99. **`zip.zip/zip/solution_v5_gci_research.py`** — Large Home Credit research CLI/script: LightGBM, XGBoost, CatBoost, linear/MLP, feature engineering, nested target encoding, rank/Nelder–Mead/stacking blends, CV and leakage diagnostics. Modes include repeated stratified 3×5 CV and leak checks; no single score claim to report. Depends on competition CSVs and many packages; CPU/GPU config varies by estimator.

### Nested telecom research scripts

100. **`zip.zip/zip/Telecom_Churn_Final_Research_Package.zip/Telecom_Churn_Final_Research_Experiment.py`** — Python counterpart to entry 1: corrected compact hybrid vs service-quality/lifecycle/account block; LightGBM, fixed three-fold stratified assessment, training-fold-only catalogue transforms. **Claim:** V10 champion ~0.7058 CV AUC; not logged here. Requires `Client.csv`, `Record.csv`, sklearn/LightGBM; CPU.
101. **`zip.zip/zip/telecom_churn_research_package.zip/telecom_churn_research_package/code/fast_cat.py`** — Fast CatBoost telecom churn experiment on joined Client/Record tables; train/test split metric code. No saved score; depends on `/mnt/data/Client.csv` and `Record.csv`; CPU.
102. **`zip.zip/zip/telecom_churn_research_package.zip/telecom_churn_research_package/code/feature_select_experiments.py`** — LightGBM telecom feature-selection experiments with categorical handling and AUC/accuracy/PR metrics. No numeric output in member; train/test split code, `/mnt/data` files and output directory; CPU.
103. **`zip.zip/zip/telecom_churn_research_package.zip/telecom_churn_research_package/code/optuna_lgbm.py`** — Optuna-tuned LightGBM churn model; split-based AUC/accuracy metrics and tuning trials. No score saved; needs Optuna, LightGBM, telecom CSVs under `/mnt/data`; CPU.
104. **`zip.zip/zip/telecom_churn_research_package.zip/telecom_churn_research_package/code/xgb_extended.py`** — Extended XGBoost telecom churn pipeline with preprocessing/imputation and mixed feature types; split-based ROC/PR/accuracy. No logged score; requires pandas/sklearn/XGBoost and joined telecom CSVs; CPU/hist not GPU evidenced.
105. **`zip.zip/zip/telecom_churn_research_package.zip/telecom_churn_research_package/code/lgbm_auc_earlystop.py`** — LightGBM telecom AUC model with early stopping on held-out split; no score output in archive. Requires telecom data and LightGBM/sklearn; CPU.
106. **`zip.zip/zip/telecom_churn_research_package.zip/telecom_churn_research_package/code/engineer_lgb.py`** — Telecom feature engineering and LightGBM fitting/evaluation scaffold; no persisted metric. Requires Client/Record CSVs and `/mnt/data` paths; dependency/config tied to that environment; no GPU evidence.
107. **`zip.zip/zip/telecom_churn_research_package.zip/telecom_churn_research_package/code/reproduce_3fold_cv.py`** — Reproduces telecom three-fold CV results; no logged score in source archive. Requires feature pipeline/data from package; stratified folds, CPU LightGBM ecosystem.
108. **`zip.zip/zip/telecom_churn_research_package.zip/telecom_churn_research_package/code/fast_xgb.py`** — Fast XGBoost churn model, train/test-split classification metrics; no saved score. Requires telecom CSVs, sklearn/XGBoost and `/mnt/data`; CPU.
109. **`zip.zip/zip/telecom_churn_research_round3.zip/telecom_churn_research_round3/code/final_retest_holdout_fast.py`** — Fast final telecom holdout retest; model comparison/evaluation, no saved numeric result. Fixed holdout-style validation; depends on round-3 data/features and libraries; CPU.
110. **`zip.zip/zip/telecom_churn_research_round3.zip/telecom_churn_research_round3/code/mrmr_models_fast_partial_execution.py`** — Partial-execution MRMR feature selection and model test; no complete score evidence. Requires round-3 input/result context; partial execution is an explicit reproducibility limit; CPU.
111. **`zip.zip/zip/telecom_churn_research_round3.zip/telecom_churn_research_round3/code/rotation_forest_partial_execution.py`** — Rotation-forest model trial, explicitly partial execution; no complete validation score. Requires specialized estimator/dependencies and research inputs; no GPU evidence.
112. **`zip.zip/zip/telecom_churn_research_round3.zip/telecom_churn_research_round3/code/kernel_neural_small_partial_execution.py`** — Small kernel/neural candidate, partial execution; no complete score. Requires neural/kernel stack and telecom data; hardware not established.
113. **`zip.zip/zip/telecom_churn_research_round3.zip/telecom_churn_research_round3/code/missingness_strategies.py`** — Compares missingness treatments as model features/preprocessing; no embedded or source numeric result established here. Requires telecom feature data and candidate model dependencies; CPU.
114. **`zip.zip/zip/telecom_churn_research_round3.zip/telecom_churn_research_round3/code/rotboost_only_approximation.py`** — Approximate RotBoost-only candidate; model approximation, not an exact full pipeline reproduction. No score captured; specialized dependencies and research data required; hardware unspecified.
115. **`zip.zip/zip/telecom_churn_research_round3.zip/telecom_churn_research_round3/code/audit_telecom.py`** — Telecom data/model audit rather than final submission; checks features/IDs/leakage patterns. No model score; needs telecom CSVs and audit dependencies; no GPU.
116. **`zip.zip/zip/telecom_churn_research_round3.zip/telecom_churn_research_round3/code/optuna_retest_12_trials.py`** — 12-trial Optuna model retest; no score output retained. Requires Optuna and model libraries plus round-3 feature/data files; CPU.
117. **`zip.zip/zip/telecom_churn_research_round3.zip/telecom_churn_research_round3/code/xgb_native_cat_partial_execution.py`** — XGBoost native categorical test, partial execution; no full score. Requires compatible XGBoost version/category handling and telecom CSVs; device config not evidenced.
118. **`zip.zip/zip/telecom_churn_research_round3.zip/telecom_churn_research_round3/code/id_sensitivity.py`** — ID-feature sensitivity/leakage probe; tests whether identifier structure contributes to churn prediction. No numeric result preserved; requires telecom IDs/labels and feature pipeline; ID-derived order must not be presented as confirmed signup chronology.
119. **`zip.zip/zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip/v10_1_experiment_review/v10_feature_engineering_patch.py`** — v10 churn feature-engineering patch; source is a targeted patch, not a stand-alone scored model. No validation score recorded; requires matching v10 notebook state/data and packages; portability depends on that package context.

**Cross-cutting audit caveats:** many GCI submissions consume cached predictions or prior champion files rather than rebuild all inputs; check cache provenance, row ordering, seed and fold identity before treating OOF comparisons as reproducible. Public leaderboard values are self-reported claims, not validation outputs. Repeated candidate selection can inflate apparent gains; the logged adversarial AUC around 0.686 flags distribution shift, not proof of label leakage.


# ARCHIVE RESEARCH DOCUMENT CATALOG

## Coverage

I read both top-level files in full using Python UTF-8 line iteration: **`D:\gci\truth\notepade.txt` (939 lines)** and **`D:\gci\truth\process.txt` (5,122 lines)**. The ZIP contains 374 root entries, including **39 nested ZIPs**; 30 nested archives have distinct archive hashes. Across the root ZIP and the unique nested archives, the requested document types total **149 path instances, grouped into 127 distinct content hashes**. Repeated copies are grouped below; paths shown are exact ZIP member paths. OOXML, PDF, workbook, JSON, log, and JSONL content was read without extracting or executing archive members.

**Evidence labels:** **LB** = leaderboard score reported in a file (not independently checked); **local** = saved OOF/validation result; **descriptive** = audit, configuration, or narrative without a comparable score. The GCI/Home Credit and Company A telecom-churn projects are separate. Telecom results do not evidence GCI performance.

## Top-level files

- **`D:\gci\truth\notepade.txt`** — Composite notebook/export plus report, not a single coherent run. The opening RealMLP/TabM notebook proposes shared folds, a LightGBM control, 3,000-row smoke tests, and GPU detection with CPU fallback; it has no saved model output establishing a successful full run. Embedded q-fingerprint/v9 records give local held-out blend AUC **0.77330** (old **0.77202**, new **0.77258**); the notebook text says the submitted v9 file had public **0.77865** (**LB claim**). A v12 note reports a borderline product-aware residual result (**+0.0003, 5/6 folds**) and says EXT residuals drove most of an earlier gain. The appended v15 report gives **+0.00016669 OOF** over v14G and explicitly says its candidate was unsubmitted then. The later README’s v15 public score of **0.77891** therefore conflicts with that report’s status unless it reflects a later submission.
- **`D:\gci\truth\process.txt`** — Full 5,122-line research/chat history. It mixes older Home Credit tutorial and external-solution discussion with GCI experiment updates. In its detailed older state, protected public champion **0.76858** and subsequent scores are explicitly user-reported; it describes amortization features using `ANNUITY/CREDIT = r(1+r)^n / ((1+r)^n − 1)` and rate-curve shape, then q/product encodings. Later passages report public **0.77283**, **0.77891**, and v16 **0.78222**; those remain **LB claims**, unlike the numerous OOF results. It warns that OOF-to-leaderboard mappings are unsupported and folds were reused. RTX 4050 references are user statements, not corroborating device telemetry. Claims about competitors using public training data to identify test rows are allegations, not established findings.

## ZIP document inventory

### GCI / Home Credit and experiment records

- **[000] `zip/0011 second[music].txt`** — Long experiment/chat record on conditional residual signals. It records a base OOF AUC of **0.7571105** and Spearman ≈ **0.21** between residual signal and base risk (**local**); it is exploratory, not a leaderboard result or confirmation against the later champion.
- **[032] `zip/GCI_CatBoost_Product_Experiment_Package.zip!/GCI_experiment_report_catboost_product.md`** and `zip/GCI_experiment_report_catboost_product.md` (same hash) — Controlled GCI product-CatBoost report; recommends a leaderboard test but explicitly says not to claim improvement until scored. Its local OOF evidence is in [033].
- **[033] `zip/GCI_CatBoost_Product_Experiment_Package.zip!/v9_full_oof_catboost_comparison.json`** — Saved comparison: v9 **0.77330055**, 10% CatBoost-product blend **0.77352911**, gain **+0.00022856**, positive on 5 folds (**local**). Not public performance; it does not establish a generalizable gain after extensive experiment selection.
- **[034] `zip/GCI_CatBoost_Product_Experiment_Package.zip!/catboost_product_experiment_results.json`** — Run/data and CatBoost-product experiment metadata (171,202 train, 61,500 test, target rate ≈8.07%). It documents experiment inputs/diagnostics; these figures are not a leaderboard score.
- **[035] `zip/GCI_CatBoost_Product_Experiment_Package.zip!/candidate_submission_validation.json`** — Candidate file/row validation and blend settings; establishes submission-shape checks, not predictive improvement or a public score.
- **[036] `zip/GCI_CA_EXT_promising_research.zip!/metrics.json`** — Compact-model local result: baseline AUC **0.748936**, CA×EXT AUC **0.753798**, delta **+0.004862**. It is not a full-ensemble comparison, so the apparent gain cannot be carried over to the public champion.
- **[037] `zip/GCI_Community_Research_Full_Extraction_and_Triage.docx`** — Community evidence/model-improvement triage and extracted material. It is a secondary research/handoff document, not a saved benchmark; quoted community scores or ideas require source-level verification.
- **[038] `zip/GCI_Competition_Candidates_Package.zip!/README_candidates.txt`** — Describes candidate CSVs and rank-averaging from three sources, with ROC-AUC as the stated metric. No independent score is established by this README.
- **[039] `zip/GCI_COMPLETE_CROSS_CHAT_RESEARCH_ARCHIVE_2026-10-20.md`** — Cross-chat GCI research archive/master plan. It consolidates prior findings and scores, so repeated claims are not independent replications; distinguish its historical leaderboard reports from saved local tests.
- **[040] `zip/GCI_ExtraTrees_Direct_Check_Report_2026-10-17.md`** — Direct ExtraTrees comparison report; local candidate assessment, not itself a public result. Its conclusions should be read with the v15 report’s warning that reconstructed rank-space mixing was not the original raw-branch recipe.
- **[041] `zip/GCI_FULL_CHAT_RESEARCH_LOG_2026-10-18.md`** — Chat chronology of experiments, submissions, and leaderboard feedback. Useful for history, but conversational public-score entries are user reports and runs differ in folds/artifacts.
- **[042] `zip/GCI_HomeCredit_BestSolution.zip!/README_results.txt`** — Older validation record: XGBoost **0.758296**, best blend `0.65 XGB + 0.10 LightGBM + 0.25 CatBoost` **0.758746** (**local validation**). It is an older baseline and not comparable to later full ensembles or LB claims.
- **[043] `zip/GCI_MASTER_HANDOFF_2026-10-18.md`** — Handoff with ROC-AUC and data-scope rules: supplied train/test/sample files only; no external labels, pretrained weights, or prohibited relational tables. Its public-score table is explicitly user-reported; it warns against inferring LB results from OOF.
- **[044] `zip/gci_peer_ablation_round2.zip!/PEER_ABLATION_REPORT.md`** — Controlled question about peer-relative financial features. It frames group definitions as separate ablations; any gains are model/fold-specific and do not alone prove champion improvement.
- **[045] `zip/GCI_Research_Progress_Report_2026-10-22.docx`** — Progress report, explicitly “progress, not victory.” It gives fixed-blend local OOF **0.772909** (seed 42), **0.772060** (seed 7), **0.772216** (seed 2026), mean **0.772395**. These are development results, not public scores.
- **[046] `zip/GCI_Research_Progress_Report_Copy_2026-10-24.docx`** — Later status separates user-reported v16 public **0.78222** from recorded five-fold local OOF **0.77649**. It warns against treating them as the same metric; public figure remains unverified here.
- **[047] `zip/GCI_research_report_detailed.md`** — October 12 historical GCI report: claims public **0.77746**, with rank thresholds also stated (**LB claims**); describes application-only Home Credit features. It is time-stamped and superseded by later candidate states.
- **[048] `zip/GCI_RESEARCH_STATUS_2026-10-17_evening.md`** — Status snapshot reports v14G public **0.77884** and local reference OOF **0.77393726**. Public figure is reported; local and public values must remain separate.
- **[049] `zip/GCI_research_summary.xlsx`** — Workbook with tabs for product-relative tests, v3 blend, score ledger, task split, and notes. It records fold/blend research; values are development evidence and its formulas/selected tests do not independently verify leaderboard scores.
- **[050] `zip/GCI_top1_research_report.md`** — Evidence-first top-1 investigation; incumbent **0.778302** is user-reported. Advises against spending long runs on random LightGBM tuning and stresses the gap to the top leaderboard tier; leaderboard snapshots are not independently validated.
- **[051] `zip/GCI_v14G_submission_report_2026-10-17  score 77884.md`** and `zip/GCI_v14G_submission_report_2026-10-17.md` (same hash) — v14G report: supplied competition files and validated artifacts; local comparisons/validation support the candidate construction, while public **0.77884** is a reported leaderboard score.
- **[052] `zip/GCI_v15_rankspace_candidate_report_2026-10-17 (1).md`** and `zip/GCI_v15_rankspace_candidate_report_2026-10-17.md` (same hash) — v15 candidate report: five-fold OOF **0.77410395** vs v14G **0.77393726**, gain **+0.00016669**, 5/5 folds positive, bootstrap interval **[+0.00007838,+0.00026092]**. It says candidate unsubmitted at report time; fold reuse/selection bias applies. Its rank-space formula is a substitute, not an exact reproduction of the raw original branches.
- **[053] `zip/GCI_v17_notebook_review_and_next_direction_2026-10-20.md`** — Notebook review says do not submit either v17 CSV; both fail the notebook’s acceptance rule. The report is a negative decision, not a public test.
- **[054] `zip/GCI_v18_alt_direction_package (1).zip!/GCI_v18_alt_direction/v18_fast_notes.json`** — Experiment configuration (three-fold split, LightGBM parameters); configuration alone is not evidence that the planned run completed or improved AUC.
- **[055] `zip/GCI_v20_research_report0__7789.md`** — Tariff-distance challenger/conservative probe. It describes small tariff blends and a public score around **0.7789** as reported, with no established win over v15; avoid reading the value as an independently saved metric.
- **[056] `zip/GCI_v21_research_report.md`** — Tariff-distance continuation: seed 20261026 compact baseline **0.756518**, tariff branch **0.759753**, gain **+0.003236** (**local**). Compact baseline is not v15’s ensemble, so this is not evidence of a leaderboard gain.
- **[057] `zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/metrics.json`** — Candidate metadata reports champion public score **0.77891** and test rank correlation **0.9472**; the former is a reported LB figure, the latter a saved prediction diagnostic. Neither is the candidate’s public score.
- **[058] `zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/GCI_v22_research_report.md`** — Tariff + bureau-burst + repayment-residual combination: compact seed 20261029 baseline **0.755375**, combo **0.758377**, mean fold gain **+0.003002**, 3/3 positive (**local**). Its own context says smaller tariff-only blends had not beaten the public champion; no champion-level LB win is shown.
- **[059] `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk1/metrics.json`** — Three-fold, seed 20261032 feature-family benchmark; compact baseline OOF **0.744508**. Results are screening metrics, not full-v15 comparisons or submission results.
- **[060] `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk1/run.log`** — Fold-by-fold execution log for chunk 1, including baseline fold AUCs; execution evidence only, not a public score.
- **[061] `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk2/metrics.json`** — Same baseline/seed framework for chunk 2; local screening results only.
- **[062] `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk2/run.log`** — Chunk 2 run log records fold AUCs; not a leaderboard submission.
- **[063] `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_combo/metrics.json`** — Combined-feature-family local screen against compact baseline; not a champion OOF or public result.
- **[064] `zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_combo/run.log`** — Combined screen’s fold run log; its metric class is local execution output, with no submission.
- **[065] `zip/GCI_v26_optimized_five_research (1).zip!/REPORT.md`** — Report calls itself v25 and says none of five hypotheses qualifies as an improvement; no submission generated. Repeated identical outer v26 archive copies have the same report content.
- **[066] `zip/GCI_v26_optimized_five_research (1).zip!/metrics.json`** — Three-fold compact formula baseline **0.758507** with variant metrics; local experiments only, not the v15 ensemble.
- **[067] `zip/GCI_v26_optimized_five_research (1).zip!/GCI_v26_optimize_five.log`** — Execution log shows fold-level AUCs (e.g. product-q encoding branch); it supports that a compact screen ran, not a public gain.
- **[068] `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/GCI_v29_fresh_confirmation.log`** — Fresh-split confirmation log for product-encoding mechanisms; local fold execution, not LB. Reported confirmation must be interpreted against that run’s compact baseline.
- **[069] `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/metrics_seed20261032.json`** — Three-fold local results: hierarchical Q4/Q3 **0.764664** vs compact baseline **0.758507**; risk-decile residual variant **0.754579** (negative). The report cautions this is not v15 ensemble OOF.
- **[070] `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/REPORT.md`** — Five data-mechanism experiments, using only supplied competition files and nested cross-fitting. It says Q4 product encoding replicated on a different fold seed (**+0.005812** vs its compact baseline, 3/3 positive), but explicitly not yet evidence of improvement to full v15.
- **[071] `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/metrics_seed20261027_confirmation.json`** — Fresh-split compact results: product Q4 TE **0.762855** vs formula baseline **0.757043**, gain **+0.005812** (**local**); not a leaderboard test.
- **[072] `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/GCI_v29_combo_followup.log`** — Follow-up execution log for hierarchical-plus-formula-burden blend; fold deltas are compact-screen diagnostics and can be selection-biased.
- **[073] `zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/GCI_v29_five_data_mechanisms.log`** — Main v29 fold log; hierarchical Q4/Q3 fold results are local and against a compact model, not the champion.
- **[074] `zip/GCI_v29_five_data-mechanisms_research.zip!/GCI_v29_package/metrics_combination_followup.json`** — Combination OOF **0.764028**, below hierarchical Q4/Q3 alone **0.764664**; blending diagnostics are on reused folds. *(Archive member’s exact filename is `GCI_v29_five_data_mechanisms_research.zip`; see path above.)*
- **[075] `zip/GCI_v4_results_summary.xlsx`** — Workbook tabs cover v4 held-out checks, product-key ablation, score ledger, report assessment, and next steps. It is a historical local-results workbook, not a public-score verifier.
- **[077] `zip/meta.json`** — Feature/category metadata (463 features, listed feature names and category levels); descriptive schema, not results.
- **[079] `zip/past reseach.txt`** — Literature/research synthesis on tabular ML, baselines, and limits; recommendations are hypotheses unless backed by the separate saved runs.
- **[080] `zip/Pasted markdown(20261008-171218).md`** — Pasted research transcript about rate-curve residuals, candidate files, and score inspection; a chat/documentation record, not independently reproduced metrics.
- **[081] `zip/README.md`** — Describes a reproducible five-fold pipeline and blending instructions. It explains intended use, not proof the commands were run or a winning score.
- **[082] `zip/README_FINAL (1).md`** and `zip/README_FINAL.md` (same hash) — Older final package claims public **0.76794**, rank 61, and supplied competition files (**LB claim**). Its score is stale relative to later documents and should not be called a current result.
- **[083] `zip/README_HONORS_FINAL.md`** — v9 reproduction package claims submitted file public **0.77865** (**LB claim**) and 61,500 ordered rows; the local reproduction OOF in notebook material is a separate metric.
- **[084] `zip/README_HONORS_FINAL_2.md`** — Duplicate-content v9 package README; same public **0.77865** claim and same caveat as [083].
- **[085] `zip/README_HONORS_V15.md`** — v15 final-package README claims public **0.77891** (**LB claim**). It conflicts in status with the contemporaneous v15 report [052], which says its candidate had not yet been submitted.
- **[086] `zip/REPORT.md`** — GCI follow-up on targeted optimization of weak hypotheses, dated 2026-10-20; supplied train/test scope and ROC-AUC protocol. It is a research report, not a public score record.
- **[087] `zip/reseach_15_10.txt`** — Suggests denoising-autoencoder features and explains that >0.80 Home Credit results often use richer data. It explicitly distinguishes opportunity from what the application-only files support; suggestion is not tested evidence.
- **[088] `zip/reseach_@.md`** and `zip/research2.md` (same hash) — Research notes updated after q-fingerprint public **0.77283** and prior **0.76815** (**LB claims**); distinguishes held-out from leaderboard values. Duplicate content, not independent evidence.
- **[089] `zip/research.md`** — Similar q-fingerprint research notes: public **0.77283** claimed; q/product encoding and residual direction documented. The score is a reported submission, not a saved local metric.
- **[090] `zip/research_10_15 pm.md`** — Another research-notes snapshot with q-fingerprint public **0.77283** claim and historical feature work; separate snapshot, not an independent experiment.
- **[091] `zip/RESEARCH_REPORT.md`** — Research Round 2 says do not submit a new candidate; no experiment established a robust successor to protected **0.76858**. It describes an 84-feature compact representation; score is prior LB context, not this round’s win.
- **[092] `zip/research_section13_weakness_analysis.md`** — Historical public product-key/v2 scores include **0.77715/0.77724** (reported); local held-out v2 comparisons show mixed/negative changes. It warns against promoting based only on LB or reused folds.
- **[093] `zip/Speech-and-Text-to-Indian-Sign-Language-with-a-3D-Avatar.pptx`** — Unrelated ISL literature/design presentation; warns benchmark scores do not prove signing is correct or understandable. No GCI evidence.
- **[126] `zip/WHY_RESIDUAL_SUBMISSION_DROPPED.md`** — Forensic analysis of residual-ranking submission; references user-reported v15 **0.77891** and examines why a residual submission dropped. The LB figure is reported, not independently measured.

### Company A telecom churn and related business materials (not GCI results)

- **[001] `zip/Churn_Optimisation_Decision_Package.zip!/Churn_Risk_and_Right_Offer_Time_Decision_Report.md`** — Decision report frames a churn model as a prioritization challenger, not proven replacement; OOF-based targeting cannot establish causal retention or financial impact.
- **[002] `zip/CompanyA_Final_Submission_Package.zip!/CompanyA_FinalSubmission.pdf`** and `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/CompanyA_FinalSubmission.pdf` (same hash) — Fifteen-page churn business proposal, risk ranking and pilot. Reported held-out ROC-AUC is part of telecom evidence; projected value is an assumed scenario, not measured uplift.
- **[003] `zip/CompanyA_Final_Submission_Package.zip!/CompanyA_FinalSubmission.pptx`** and `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/CompanyA_FinalSubmission.pptx` (same hash) — Presentation says model is a prioritization tool and recommends testing actions; it does not show causal campaign impact or GCI scores.
- **[004] `zip/CompanyA_Final_Submission_Package.zip!/CompanyA_References_for_Omnicampus.txt`** and corresponding Omnicampus bundle path (same hash) — References dataset documentation, leakage guidance, and churn literature; background sources, not direct empirical proof.
- **[005] `zip/CompanyA_Final_Submission_Package.zip!/README_SUBMISSION.md`** and corresponding Omnicampus bundle path (same hash) — Lists reproducible notebook, presentation, and results artifacts; package description, not a score report.
- **[006] `zip/CompanyA_Final_Submission_Package.zip!/RESEARCH_AND_LIMITATIONS.md`** and corresponding Omnicampus bundle path (same hash) — Gives same 20,000-row holdout baseline LGBM ROC-AUC **0.701895** vs nine-domain-feature **0.707243**, paired bootstrap gain **+0.005348** (95% interval **[0.002707, 0.007864]**). Explicitly says repeated dataset exploration is not adjusted for; top-decile churn rate is targeting precision, not accuracy or intervention effect.
- **[007] `zip/CompanyA_Final_Submission_Package.zip!/results/notebook_code_cell_check.json`** and corresponding Omnicampus bundle path (same hash) — Reports notebook cell check **12/12 passed**, 23.16 seconds; execution validation only, not model generalization.
- **[008] `zip/CompanyA_Final_Submission_Package.zip!/results/recreated_submission_stats.json`** and corresponding Omnicampus bundle path (same hash) — Submission/data shape and merged-row statistics; validates artifact structure, not predictive performance.
- **[009] `zip/CompanyA_Omnicampus_Submission_Bundle (1).zip!/Omnicampus_Form_Answers_and_Upload_Guide.txt`** — Submission form guidance, disclosures, and reference instructions; administrative material, not model evidence.
- **[010] `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/CompanyA_Telecom_Churn_Complete_Chat_Experiment_Log.docx`** — Chronological churn research log through V11; explicitly says scores from different splits are not comparable and associations are observational/financial effects assumed absent randomized evidence.
- **[011] `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/CompanyA_Telecom_Churn_Complete_Chat_Experiment_Log.md`** — Markdown counterpart, with aspiration of >0.80 AUC/accuracy not established. Treat historical scores as split-specific.
- **[012] `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/v10_feature_finance_continuation_report.md`** — V10.2 data quality, financial assumptions, and feature candidates; says no route to 0.80 is established and separates development evidence from financial assumptions.
- **[013] `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/v9_technical_audit_and_slide_fixes.md`** — V9 audit flags unsupported customer-ID claims and overstatement; supports risk ranking, not “high overall accuracy” or causal value.
- **[014] `zip/CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/v10_morning_to_now_report.md`** and `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/V10_MORNING_TO_NOW_RESEARCH_REPORT.md` (same hash) — V10 association/data audit; reports tenure/equipment associations and two feature bugs, but they are observational and require holdout/pilot verification.
- **[016] `zip/figures_and_results (1).zip!/results_summary.json`** and `zip/figures_and_results (2).zip!/results_summary.json` (same hash) — Churn model results summary: booster blend ROC-AUC **0.70863**; local model metric, not GCI.
- **[017] `zip/figures_and_results (1).zip!/slide_summary.txt`** and corresponding `(2)` path (same hash) — Slide summary rounds to ROC-AUC **0.709**, PR-AUC **0.701**, accuracy **0.649**, F1 **0.645**, top-decile lift **1.64×**. Telecom validation metrics; not causal or GCI.
- **[018] `zip/figures_and_results (3).zip!/results_summary.json`** — Churn model summary ROC-AUC **0.70663**, local.
- **[019] `zip/figures_and_results (3).zip!/slide_summary.txt`** — Rounded churn metrics: ROC-AUC **0.707**, PR-AUC **0.699**, accuracy **0.647**, F1 **0.644**, lift **1.63×**.
- **[020] `zip/figures_and_results (4).zip!/results_summary.json`** — XGBoost multi-seed churn result ROC-AUC **0.68209**, local.
- **[021] `zip/figures_and_results (4).zip!/slide_summary.txt`** — Rounded ROC-AUC **0.682**; metrics vary by experiment/version and are not directly interchangeable.
- **[022] `zip/figures_and_results (5).zip!/results_summary.json`** and `(6)` counterpart (same hash) — Churn blend ROC-AUC **0.70663**, local.
- **[023] `zip/figures_and_results (5).zip!/slide_summary.txt`** and `(6)` counterpart (same hash) — ROC-AUC **0.707**, PR-AUC **0.699**, accuracy **0.647**, F1 **0.644**, lift **1.63×**.
- **[024] `zip/figures_and_results.zip!/results_summary.json`** — Earlier XGBoost churn result ROC-AUC **0.69600**, local.
- **[025] `zip/figures_and_results.zip!/slide_summary.txt`** — Rounded ROC-AUC **0.696**, PR-AUC **0.684**, accuracy **0.640**, F1 **0.642**, lift **1.59×**.
- **[026] `zip/figures_and_results_v10.zip!/results_summary_v10.json`** — V10 churn summary ROC-AUC **0.70605**, local; version-specific.
- **[027] `zip/figures_and_results_v10.zip!/slide_summary_v10.txt`** — Illustrative financial case explicitly labels acceptance, effect size, margin, and cost as assumptions; not realized savings.
- **[028] `zip/figures_and_results_v11.zip!/results_summary_v11.json`** — V11 churn summary ROC-AUC **0.70555**, local.
- **[029] `zip/figures_and_results_v11.zip!/slide_summary_v11.txt`** — V11 scenario presents assumed targeting/economic effects; not a randomized treatment result.
- **[030] `zip/figures_and_results_v9.zip!/results_summary_v9.json`** — V9 churn result ROC-AUC **0.70751**, local.
- **[031] `zip/figures_and_results_v9.zip!/slide_summary_v9.txt`** — V9 slide headline projects value for positive-expected-value customers; projection depends on assumptions, not measured revenue.
- **[076] `zip/interim_feature_research_evidence.zip!/INTERIM_FINDINGS_REPORT.md`** and `zip/INTERIM_FINDINGS_REPORT.md` (same hash) — Telecom feature-engineering interim report; compares domain-feature families on supplied churn data and presents a challenger, not proven replacement of historical champion.
- **[094] `zip/Telecom_Churn_Final_Research_Package.zip!/Telecom_Churn_Final_Research_Report.md`** — Final evidence-led churn report recommends keeping corrected hybrid and targeted service/lifecycle/account features as challenger; no causal uplift or GCI claim.
- **[095] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/ML_Experiment_Report.md`** — Churn ML audit says requested **0.80 accuracy / 0.80 ROC-AUC was not achieved**; leakage-conscious runs are around 0.69–0.71.
- **[096] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/README.txt`** — Package index for churn experiment reports and metrics; no independent result.
- **[097] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/xgb_fast_results_raw.jsonl`** — Two XGBoost churn trials: AUCs about **0.6982** and **0.7024**; local validation.
- **[098] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/lgbm_fast_results_raw.jsonl`** — LightGBM churn trials about **0.7022** and **0.7012** AUC; local.
- **[099] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/diagnostic_id_ethnicity.json`** — Diagnostic model including ID/ethnicity AUC **0.70634**; explicitly diagnostic, not acceptable as proof of clean generalization.
- **[100] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/feature_select.json`** and `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/previous_feature_selection_results.json` (same hash) — Feature-selection JSON: baseline **0.70419**, top-25 **0.70216**; selection did not improve this local score.
- **[101] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/final_holdout_result.json`** — Final tuned LightGBM holdout record (stratified 70/15/15); score is a separate holdout result, not GCI.
- **[102] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/cat_fast_results_raw.jsonl`** — CatBoost depth 5/6 churn trials AUC about **0.7017/0.7024**; local validation.
- **[103] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/lgb_auc_earlystop_raw.jsonl`** — LightGBM early-stop trial around **0.70419** AUC; local.
- **[104] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/xgb_extended_raw.jsonl`** — Extended XGBoost churn trial AUC **0.70626**; local.
- **[105] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/lightgbm_3fold_oof_result.json`** and round-3 `previous_3fold_cv_result.json` (same hash) — Fixed three-fold LightGBM OOF AUC **0.69627**; local, split-specific.
- **[106] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/cat_higher_raw.jsonl`** — Higher-depth CatBoost churn trials about **0.7021/0.7010** AUC; no clear gain.
- **[107] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/engineered_lgb_result.json`** — 132-feature peer-interaction LightGBM AUC **0.69560**, below the reported baseline; local negative result.
- **[108] `zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/data_audit.json`** — Churn data merge audit: 100,000 rows each and 100 merged columns; schema/integrity metadata, not model evidence.
- **[109] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/README.txt`** — Round-3 package index, points to final findings and holdout JSON; no standalone score.
- **[110] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/FINAL_FINDINGS.md`** — Round-3 conclusion says target 0.80 accuracy/AUC was not achieved; supplied Client/Record experiments remain around 0.69–0.71.
- **[111] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/rotboost_only_result.json`** — Rotation-boost result AUC **0.62837**, local negative result.
- **[112] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/data_and_hash_audit.json`** — Dataset hash/row audit; verifies recorded inputs, not model performance.
- **[113] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/kernel_neural_partial_execution.log`** — Partial SVC runs: RBF AUC **0.61316**, polynomial **0.59955**; incomplete/local.
- **[114] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/id_sensitivity.jsonl`** — ID-excluded model AUC **0.69956** vs ID-included **0.69858**; local diagnostic suggests ID did not help.
- **[115] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/xgb_native_cat_partial_execution.log`** — Partial native-categorical XGBoost AUC **0.69724**; local and incomplete.
- **[116] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/rotation_forest_partial_execution.log`** — Rotation-tree run AUC **0.65345**; local negative result.
- **[117] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/optuna_retest_partial_3_of_12.log`** — Only 3 of 12 tuning trials logged, roughly **0.6986–0.6987** AUC; incomplete, not a final model comparison.
- **[118] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/data_audit.txt`** — Churn merge: 100,000 rows, target rate **0.49562**, ID-AUC **0.47531**; descriptive data audit.
- **[119] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/final_retest_holdout_fast.json`** — Fast LightGBM holdout record with split seed and row counts; local holdout, separate from GCI.
- **[120] `zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/references/SOURCES.md`** — Churn literature and scikit-learn leakage references; methodological background, not score evidence.
- **[121] `zip/V10_ANALYSIS_TO_DATE_REPORT.md`** — Churn domain/data audit and feature-family report; discusses controlled retention-pilot design but does not measure campaign causality.
- **[122] `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/Right_Customer_Right_Offer_Right_Time_v10_2_Financial_Feature_Pilot_Workbook.xlsx`** — Workbook includes assumptions, sensitivity, feature candidates, pilot, audits, finance and source/method tabs; scenario calculations are assumption-dependent.
- **[123] `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/revenue_semantic_reconciliation.json`** — Checks revenue/overage field reconciliation (≈99.8% exact equality where available); data semantics, not predictive lift.
- **[124] `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/v10_feature_bug_impact.json`** — Documents missing-value bugs: 357 missing plan fees assigned lowest tier and 847 missing handset prices assigned rank 1; supports correcting preprocessing, not a causal or GCI gain.
- **[125] `zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/data_overview_extract.txt`** — Company A dataset field definitions (handset price, model counts, etc.); descriptive and unrelated to GCI.
- **[015] `zip/DA_GATE2027_Syllabus (2).pdf`** — GATE syllabus covering core data/CS topics; unrelated to either competition’s model evidence.

## Duplicate archive aliases

Byte-identical nested ZIP copies were inspected once. Their contents are the same as the listed member paths under these outer aliases:

- `CompanyA_Omnicampus_Submission_Bundle.zip` and `(1)` through `(6)` are identical.
- `CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence.zip` and `(1)` are identical.
- `GCI_v18_alt_direction_package.zip` and `(1)` are identical.
- `GCI_v26_optimized_five_research.zip` and `(1)` are identical.

Identical document copies within distinct archives are explicitly grouped above. The archive reports contain historical snapshots and mixed validation protocols; reported leaderboard values, saved OOF/holdout scores, and assumed business outcomes should not be treated as interchangeable.


# ARCHIVE CSV FILE-BY-FILE CATALOG

## Scope and grouping

I read every CSV row from the archive root and the nested ZIPs, including duplicate archive copies. There are **338 CSV path occurrences** and **240 distinct CSV byte-content groups**: 150 test-submission groups, 2 OOF-prediction groups, 56 score/experiment tables, 22 feature/analysis tables, and 10 unrelated churn/business tables. Each entry below gives a SHA-256 prefix as a group identifier; paths after `|` are byte-identical copies. Paths are relative to `D:\gci\truth\zip.zip`.

All 150 submission groups have the same schema and alignment: **61,500 rows × 2 columns**, `SK_ID_CURR,TARGET`; IDs are unique, consecutive `171202–232701`, in the same order, with no nulls or malformed rows. `TARGET` is a prediction here—not an observed label. Thus, **AUC cannot be inferred** from these test-only files; names and prediction distributions do not establish a leaderboard score.

## Test-submission content groups

Ranges below are the observed `TARGET` prediction min–max; `out` counts values below 0 / above 1. These are descriptive, not scores.

```text
009c27b95b  2.43906e-05..1  RANKBLEND_5.csv
00fd95f467  2.43906e-06..0.999995  submission_10_blend_15.csv
076be21f1a  0.00294831..0.66375  FORMULA_FUSION_LAYER2_CA_RESIDUAL.csv
078a14243f  4.87813e-06..1  GCI_submission_v17 (1).csv; GCI_submission_v17.csv
0806f45615  3.25209e-05..0.999997  v4_B_greedy_old_members_only.csv
098acf6943  3.25209e-06..0.999993  submission_13_old_anchor_90_file12_10.csv
09e1e4bc5f  0.00099727..0.827248  GCI_archived_XGB_0.77308_candidate.csv; GCI_HomeCredit_BestSolution.zip!/submission_xgb.csv; submission_archived_xgb_0.77308_candidate.csv; submission_xgb(1).csv; submission_xgb.csv
0b648d4dac  0.000221789..0.999999  GCI_champion_catboost_rankblend_4pct.csv
0c5e7f17f3  3.25209e-05..0.999997  GCI_submission_reproduced.csv; v3_A_greedy_oof_weights (1).csv; v3_A_greedy_oof_weights.csv
0c9d08429f  0.00221589..0.999158  GCI_DE_SENT_CA_residual_submission.csv
189003a9c8  1.46344e-05..0.99998  GCI_submission_v15_reproduced.csv
1b217a1b30  6.55837e-05..1  GCI_ratecurve_template_v2_submission.csv; submission (1).csv
1babc45d13  1.62604e-06..0.999997  submission_10_blend_10.csv
1d07a52fb6  0.00215504..0.830364  submission_03_best_single.csv
1dfdc4b01a  0..1  GCI_v8_conditional_XGB_disagreement30_w20.csv
1e9069313b  4.06511e-05..1  RANKBLEND_15.csv
2089ae9118  2.03252e-05..0.999988  candidate_rank.csv; GCI_Competition_Candidates_Package.zip!/candidate_rank.csv
2161f54eab  0..1  GCI_adaptive_shape_top30_w08_gate93_protected.csv
23386d6a26  0.0500183..0.97008  champion_90_rc_10.csv
256ec510b0  0.0016671..1  GCI_rate_geometry_residual_gate93_w05_submission.csv
264bd37751  8.13021e-07..0.999998  submission_10_blend_5.csv
26df18e07d  0.000122114..0.999913  GCI_qfingerprint_ext2bin_micro20pct.csv
2b24eeee48  0.00262687..1  GCI_RD3_on_076858_gate95_w05.csv
2d081cd7d0  0.00284925..1  GCI_RD3_micro_75pct_of_w01_gate93 (1).csv; GCI_RD3_micro_75pct_of_w01_gate93.csv
2f5d7c8013  8.13021e-06..0.999992  submission_targeted_second_order_75_25.csv
31d5be5dea  0..1  GCI_v7_champion67_xgbrich33.csv
331ddda62a  4.87813e-06..0.99999  submission_13_old_anchor_85_file12_15.csv
34d067a696  2.84557e-05..0.999996  v3_B_plain_average_all (1).csv; v3_B_plain_average_all.csv; v3_C_average_lgb_members_only (1).csv; v3_C_average_lgb_members_only.csv
36f8cba3d2  0.000376104..1  GCI_rate_shape_residual_3pct_gate93.csv
37f22c3eed  7.78049e-05..0.999953  GCI_qfingerprint_extmean_micro10pct.csv
39a1d0a7e1  2.0163e-05..0.99999  v5_A_new_members_same_weights (1).csv; v5_A_new_members_same_weights.csv
39c682199a  0.00026911..0.99941  ROUND1_CA_EXT2_RESID_submission.csv
3a71990c40  2.1681e-05..1  GCI_submission_qmodels_only (1).csv; GCI_submission_qmodels_only.csv
3a7d90f696  0.0011274..0.8222  submission_10_xgb_onehot_no_extagg.csv
3ea86224b7  0.00095839..0.99978  GCI_harmonic_transfer_v1_7pct.csv
4048526ad3  0.0039317..0.71974  sub_lgb_rich_full.csv
410b5286cc  1.626e-05..1  gci_next_rank_tiebreak.csv
44d95183ec  3.2521e-05..1  RANKBLEND_10.csv
49680c0fbd  6.6749e-05..0.99999  GCI_financial_fingerprint_submission.csv
4a6eb7cfb6  0..1  GCI_submission_product_blend30.csv
4b50943bd2  2.5675e-05..1  v11_A_v6B_plus_family_bagged (1).csv; v11_A_v6B_plus_family_bagged (2).csv; v11_A_v6B_plus_family_bagged.csv
4c12865f2f  3.2521e-06..1  GCI_submission_v16.csv
4d530c407b  0.05164..0.91535  brf_full_candidate.csv
4d908d9eea  7.7317e-05..0.99995  GCI_qfingerprint_ext2bin_micro10pct.csv
4f98e2532a  4.0651e-05..0.99999  GCI_allfam_transfer_v1_submission.csv
50faadf3e5  6.5042e-06..0.99999  submission_13_old_anchor_80_file12_20.csv
531090e097  0.0029372..0.99908  GCI_submission_qfingerprint_cl.csv
551bb87343  3.2521e-06..0.99999  submission_13_tri_85_10_5.csv
55dce05eb0  0.000267..1  GCI_rate_shape_residual_2pct_gate93.csv
5a6821b12d  0..1  submission_FINAL_BEAT_CLAUDE_maxbin25_cat5.csv
5aadada072  5.8538e-06..0.99999  submission_NEXT_BEAT_CLAUDE_oblique18.csv
5b0674a257  9.9595e-05..1  GCI_v5_champion95_CAG5_gate97_submission.csv
5b6ff379b5  0.10002..0.94027  champion_80_rc_20.csv
5f0bc0ab60  0.025019..0.98499  champion_95_rc_5.csv
619d636c62  9.1058e-06..0.99998  v7_A_et_members_same_weights.csv
63d5956382  2.709e-05..0.99997  v8_A_v6B_plus_ranker.csv
6464b21449  2.1139e-05..0.99997  GCI_submission_v15_rankspace_consistent_view.csv
649f1eaefa  0.023794..0.48527  gbc_full_candidate.csv
6589785c8f  0.0025861..1  GCI_RD3_transfer_w02_gate93.csv
670e0fd305  0.0036807..0.75443  lgb_a.csv
68210ef7c9  −0.00044828..1; out 10/0  GCI_RD3_reverse_micro_25pct_gate93 (1).csv; GCI_RD3_reverse_micro_25pct_gate93.csv
6940c1ce94  0.0038322..1  GCI_CA_EXTQ_v3_submission.csv
6a92117ac5  7.1546e-06..0.99999  v6_A_new_members_same_weights.csv; v6_A_new_members_same_weights_modified.csv
6aa90d70f0  7.9837e-05..1  GCI_champion_catboost_rankblend_1pct.csv
6abfbf4c6c  0.00059359..1  CA_EXT1_3pct_on_0.76835_champion.csv
6b42df3c0e  4.3293e-05..1  GCI_submission_v21_tariff_distance_2p5pct_sensitivity.csv
6cc397b262  1.4634e-05..0.99998  GCI_submission_catboost_product_blend_reproduced.csv
6cdf240245  0.0018428..0.88765  xgb_scale2_seedavg.csv
6e3c64eb1c  4.3841e-05..0.99998  GCI_qfingerprint_extmean_micro025pct.csv
7056ab823d  0.00013545..0.99973  champ_lgb_domain_92_5_3.csv
74c4b46c83  4.0651e-05..1  GCI_CAG5_current_0p76837_champion_gate97_submission.csv
7511a071c7  0.012237..0.59543  BASE.csv
7a95663da3  −0.0012727..1; out 14/0  GCI_RD3_reverse_sweep_m50_of_w01_gate93.csv
7d53884847  0.00010081..0.99999  GCI_implied_rate_v3_submission.csv
7d57fc2d0e  3.3171e-05..1  GCI_CA_EXT_2pct_champion_blend_submission.csv
8143e24447  1.1382e-06..1  submission_10_blend_7.csv
83df6389ef  5.9351e-05..0.99989  GCI_CADOWN_fingerprint_submission.csv
86eabb3659  4.635e-05..1  GCI_RD3_reverse_sweep_m10_of_w01_gate93.csv
871fe745f3  4.8781e-06..1  GCI_submission_v18.csv
87a4476468  1.9107e-05..1  GCI_submission_v18_alt_direction_probe.csv
87f10ff4b2  0.0015119..0.81303  sub_xgb_rich_seedavg.csv
89930bbacd  2.6016e-05..0.99998  GCI_2ca_fingerprint_submission.csv
8caf037244  0.0010085..0.77765  submission_log_credit_term.csv
8d9afc723e  0..1  GCI_testlike_specialist_submission.csv
8fb7faef47  0..0.70938  mlp_full_candidate.csv
8fbc351fbe  2.4716e-05..0.99999  GCI_submission_reproduced_final.csv; GCI_submission_reproduced_final_oct_17.csv
9012f07a43  0.0023743..1  GCI_RD3_transfer_w01_gate93.csv
925f58728e  0.0021703..0.76311  candidate_conservative.csv; GCI_Competition_Candidates_Package.zip!/candidate_conservative.csv
928daef0ca  1.626e-05..1  v2_D_lgb50_prevrun50.csv
92fd4ad9ef  0.00045041..1  GCI_experimental_champion_catboost_rankblend_5pct (1).csv; GCI_experimental_champion_catboost_rankblend_5pct.csv
94f21664a3  4.8781e-07..1  submission_10_blend_3.csv
97a23bac04  0.0020249..1  GCI_RD3_micro_50pct_of_w01_gate93 (1).csv; GCI_RD3_micro_50pct_of_w01_gate93.csv
99d6f4554b  0.0033442..0.72914  GCI_v33_product_conditional_model_only.csv
9a4432e178  1.626e-05..1  GCI_submission_v21_candidate.csv
9d1c15a0d9  0.00010537..0.9998  champ_lgb_domain_95_3_2.csv
9fba83f468  0.035303..0.90536  GCI_HomeCredit_BestSolution.zip!/submission_rf.csv; submission_rf.csv
ac43cb2e9e  2.1139e-05..0.99997  GCI_submission_v14_G_B_plus_10pct_CatBoost.csv
ad01410dd2  4.1545e-05..0.99995  GCI_CA_fingerprint_v2_submission.csv
ad62d971dc91  0.0036007..0.71252  GCI_HomeCredit_BestSolution.zip!/submission_lgb.csv; submission_lgb.csv
ade1a2520d7c  4.7154e-05..0.99999  GCI_submission_v22_tariff_bureau_combo_10pct.csv; GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/GCI_submission_v22_tariff_bureau_combo_10pct.csv
afd5f52c2b  0.0059384..0.99585  GCI_RD3_RangeStd_15pct_candidate.csv
b008470ddd  0.00014919..1  GCI_residual_ranking_5pct_v15_leaderboard_probe.csv
b4b385c22f  5.4919e-05..0.99997  GCI_qfingerprint_ext2bin_micro05pct.csv
b57e67b8b0  1.5773e-05..0.99999  submission_BEAT_CLAUDE_cat7_lgb3.csv
b8f2f79103  4.0651e-06..1  submission__regime5.csv
b96843b24f  3.2521e-05..0.99999  v2_A_lgb_product_k.csv
bcab0d33a5  1.8618e-05..1  GCI_FINAL_CANDIDATE_0p5pct_blend_NOT_model_only.csv
bcd763c85f  0.0019285..1  v3_submission.csv
bd309d93d8  8.8619e-05..1  GCI_highres_gated_CA_submission.csv
bf2d41f17b  2.8293e-05..0.99999  v6_B_mix_old_and_new (1).csv; v6_B_mix_old_and_new (2).csv; v6_B_mix_old_and_new.csv; v6_B_mix_old_and_new_modified.csv
c0f4541f22  5.5163e-05..0.99997  GCI_qfingerprint_extmean_micro05pct.csv
c2c3dfe440  0.0017001..0.80051  submission_te_xgb_97_champ_3.csv
c384c3179f  6.7725e-05..1; out 0/1  GCI_risk_gated_fingerprint_submission.csv
c3f2ac211d  0.0044248..0.71444  submission_selected_lgb.csv
c3fc40999a  1.626e-05..0.99998  GCI_CatBoost_Product_Experiment_Package.zip!/GCI_submission_catboost_product_blend.csv; GCI_submission_catboost_product_blend.csv
c4c1a8e7bc  0.00017447..1  GCI_champion_catboost_rankblend_3pct.csv
c559fed498  1.626e-05..1  GCI_highres_lgb_submission.csv
c6491d3834  −0.00061316..1; out 10/0  GCI_RD3_reverse_sweep_m30_of_w01_gate93.csv
c7021a645f  3.7941e-05..0.99999  A_lgb_product_only (1).csv; A_lgb_product_only.csv
c7086a8159  7.8862e-05..0.99988  champ_lgb_domain_97_2_1.csv
c992fd79f7  3.5773e-05..1  GCI_goss_v1_champion90_goss10.csv
ccb0dcbffa  0.00048521..1  GCI_rate_shape_residual_4pct_gate93 (1).csv; GCI_rate_shape_residual_4pct_gate93.csv
cd807dd332  −0.00094291..1; out 14/0  GCI_RD3_reverse_sweep_m40_of_w01_gate93.csv
ce502dc663  −0.0002834..1; out 9/0  GCI_RD3_reverse_sweep_m20_of_w01_gate93.csv
d1a30fee1e  0.0022936..0.78865  submission_04_stack_LR.csv
d41c297786  0.0023141..0.75925  candidate_balanced.csv; GCI_Competition_Candidates_Package.zip!/candidate_balanced.csv
d481c039fb  2.6342e-05..1  v5_B_mix_old_and_new (1).csv; v5_B_mix_old_and_new.csv
d5e3055b40  2.1464e-05..0.99997  v7_B_mix_v6_and_et.csv
d924ca010a  0.5..0.70179  rc_xgb_beta125_prob.csv
d9fdf72fd0  0.0017517..0.79434  submission_te_xgb_seedavg.csv
e2196cbbfb  2.7968e-05..0.99998  v4_A_greedy_all_members.csv
e21ea8b353  1.5176e-06..1  submission_FINAL_76745_downsample7.csv
e41a97eb4f  0.00098376..0.99998  GCI_template_meta_local5_submission.csv
e4f128b3b6  0.0032571..0.72997  GCI_HomeCredit_BestSolution.zip!/submission_cat.csv; submission_cat.csv
e850615eb2  −0.00011853..1; out 3/0  GCI_RD3_reverse_sweep_m15_of_w01_gate93.csv
e8b33aa094  7.927e-06..1  GCI_submission_current_champion_0.76745 (1).csv; GCI_submission_current_champion_0.76745.csv
ec5a19e2a5  2.0325e-05..1  phase4_domain_view_rank15.csv
ec61f60d03  0.011254..0.60239  TOP10_COMBINED.csv
eee7753809  0.0012005..1  GCI_RD3_micro_25pct_of_w01_gate93.csv
eef8a3d9d2  1.626e-06..1  submission_13_old_anchor_95_file12_5.csv
f2aeab18d0  3.248e-05..1  GCI_submission_v21_tariff_distance_1p5pct.csv
f5afbc4e1d  1.084e-05..0.99999  submission_01_rank_equal.csv
f66ae16019  0.00022033..1  GCI_v6_champion93_CADOWN10_gate93_submission.csv
f7ebd9e956  0..1  GCI_submission_product_blend50.csv
fa4c5d07ee  0.00012309..0.99992  GCI_qfingerprint_extmean_micro20pct.csv; GCI_submission_candidate_extmean_2pct.csv
fb567544f9  0.00060096..1  GCI_ratecurve_template_v1_submission.csv
fba5ba75d7  0.0039936..1  GCI_affordability_curve_residual_8pct_gate93.csv
fe784d4694  0.0020689..0.7715  best_submission (1).csv; best_submission.csv; GCI_Competition_Candidates_Package.zip!/best_submission.csv; GCI_HomeCredit_BestSolution.zip!/best_submission.csv
fe9d605ab6  3.6857e-05..1  GCI_exact_rate_ext_affordability_submission.csv
ff62ef29af  2.1667e-05..1  GCI_submission_v20_tariff_distance_probe.csv
```

## Other distinct CSV groups

Here `rows×cols` is the full CSV size. Headers are shown in full when short; `…+N` means N additional columns. `R` gives stored numeric metric/prediction-field ranges when applicable; these are **values recorded in the table**, not independently reproduced AUCs unless stated below. Duplicate paths are listed together.

```text
1  unrelated-churn threshold table, 18×8; H=model,threshold,accuracy,precision,recall,…+3; R predicted_positive_rate 0.1009..0.8711
  zip/Churn_Optimisation_Decision_Package.zip!/Churn_Threshold_Tradeoff_Analysis.csv
2  feature/business sensitivity, 5×6; H=protected_revenue_horizon_months,gross_revenue_protected_usd_assumption,retained_contribution_at_40pct_margin,campaign_cost_held_fixed,net_contribution_fixed_base_audience
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/v10_fixed_policy_horizon_sensitivity.csv
3  score table, 6×5; H=variant,pooled_auc,delta_vs_baseline,fold_auc,fold_delta_vs_baseline; R pooled_auc .7588..7640, delta .0003184..005521
  zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/results_combination_followup.csv
4  unrelated-churn OOF blend score table, 11×4; H=quality_weight,oof_auc,oof_pr_auc,oof_log_loss; R OOF-AUC .7029..7045, PR-AUC .6935..6952
  zip/Churn_Optimisation_Decision_Package.zip!/Deep_Experiment3_Blend_Results.csv
  = zip/Telecom_Churn_Final_Research_Package.zip!/Deep_Experiment3_Blend_Results.csv
5  unrelated-churn risk-decile table, 10×6; H=risk_decile_1_highest,n,observed_churn_rate,mean_predicted_risk,lift_vs_holdout_baseline,…+1; mean risk .2158..7825
  zip/CompanyA_Final_Submission_Package.zip!/model_outputs/risk_deciles.csv
  = corresponding member in `CompanyA_Omnicampus_Submission_Bundle.zip` and its `(1)`–`(6)` copies
6  recorded V18 score table, 1×8; H=variant,n_features,pooled_auc,fold1_auc,fold2_auc,…+3; pooled .7564
  zip/GCI_v18_alt_direction_package.zip!/GCI_v18_alt_direction/v18_progress.csv
  = same member in `GCI_v18_alt_direction_package (1).zip`
7  GCI interaction screen, 18×4; H=seed,interaction_weight,auc,gain_vs_q4; AUC .6728..6973
  zip/q4_interaction_blend_screen.csv
8  unrelated-churn risk-group table, 3×7; H=top_risk_share,customers_in_group,churners_in_group,group_churn_rate,sample_baseline_churn_rate,…+2
  zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/risk_decile_business_metrics.csv
9  unrelated-churn financial sensitivity, 20×9; H=assumed_absolute_ITT_effect_pp,revenue_proxy_per_month,assumed_margin,assumed_months,implied_contribution_per_offered_customer,…+4
  zip/CompanyA_Final_Submission_Package.zip!/results/financial_sensitivity.csv
  = corresponding member in `CompanyA_Omnicampus_Submission_Bundle.zip` and its `(1)`–`(6)` copies
10 feature/analysis, 9×8; H=feature_group,n_flagged,share_flagged,churn_flagged,churn_other,…+3
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/association_risk_diff_development_only.csv
11 peer ablation score table, 20×5; H=seed,block,fold,auc,best_iteration; AUC .7400..7434
  zip/gci_peer_ablation_round2.zip!/peer_ablation_results.csv = zip/peer_ablation_results.csv
12 unrelated-churn action list, 1,000×9; H=Customer_ID,p_churn,rev_month,months,eqpdays,…+4
  zip/figures_and_results (5).zip!/action_list_top1000.csv = same in `(6).zip`
13 unrelated-churn feature experiment table, 16×7; H=experiment,roc_auc_mean,roc_auc_sd,pr_auc_mean,accuracy_threshold_0_5,…+2; ROC-AUC mean .6892..7003
  `CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/domain_feature_consolidated_results.csv`
  = same in its non-numbered ZIP; hash-identical to `interim_feature_research_evidence.zip!/consolidated_feature_experiment_results.csv`
14 CatBoost product experiment folds, 5×4; H=fold,v9_auc,v9_plus_10pct_catboost_product_auc,delta; stored AUCs .7703..7788
  zip/GCI_CatBoost_Product_Experiment_Package.zip!/v9_full_oof_catboost_comparison_folds.csv
15 unrelated-churn action list, 1,000×9; H=Customer_ID,p_churn,rev_month,months,eqpdays,…+4
  zip/figures_and_results (3).zip!/action_list_top1000.csv
16 unrelated-churn held-out comparison table, 4×11; H=model,threshold_preselected_from_validation,test_auc,test_pr_auc,test_accuracy,…+6; stored test-AUC .6961..6991
  zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/heldout_comparison_exploratory.csv
17 GCI V29 confirmation score table, 5×4; H=variant,pooled_auc,delta_vs_formula_baseline,fold_delta; AUC .7570..7629
  zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/results_seed20261027_confirmation.csv
18 unrelated-churn fold results, 6×7; H=fold,model,auc,pr_auc,log_loss,…+2; stored AUC .7019..7048
  `Churn_Optimisation_Decision_Package.zip!/Deep_Experiment3_Fold_Results.csv`
  = same member in `Telecom_Churn_Final_Research_Package.zip`
19 CA-EXT folds, 3×7; H=fold,n_train,n_valid,baseline_auc,ca_ext_auc,…+2; stored baseline .7424..7595, CA-EXT .7473..7638
  zip/GCI_CA_EXT_promising_research.zip!/folds.csv
20 V26 result table, 6×4; H=variant,pooled_auc,delta_vs_formula_baseline,fold_delta; AUC .7556..7634
  `GCI_v26_optimized_five_research (1).zip!/results.csv`
  = corresponding member in its non-numbered ZIP
21 unrelated-churn experiment scores, 33×8; H=split_or_seed,model,evaluation,roc_auc,pr_auc,…+3; ROC-AUC .5996..7082
  zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/consolidated_experiment_results.csv
22 peer ablation summary, 10×6; H=seed,block,mean_fold_auc,std_fold_auc,min_fold_auc,…+1; mean-fold AUC .7417..7422
  zip/gci_peer_ablation_round2.zip!/peer_ablation_summary.csv = zip/peer_ablation_summary.csv
23 unrelated-churn labeled predictions, 100,000×3; H=churn,oof_catalogue_traj_monet,oof_catalogue_traj_monet_quality; score ranges .0660..9544 / .0610..9515
  `Churn_Optimisation_Decision_Package.zip!/Deep_Experiment3_OOF_Predictions.csv`
  = same member in `Telecom_Churn_Final_Research_Package.zip`
24 CA-EXT experiment screen, 18×7; H=fold,exp,base_auc,best_auc,gain,…+2; stored base .7220..7443, best .7226..7466
  zip/ca_ext_ABCDE_results_fast2.csv
25 unrelated-churn action list, 1,000×9; H=Customer_ID,p_churn,rev_month,months,eqpdays,…+4
  zip/figures_and_results (4).zip!/action_list_top1000.csv
26 feature importance, 131×10; H=feature,mean_gain,gain_sd,gain_nonzero_folds,mean_splits,…+5; mean gain 3.109..27,400
  zip/interim_feature_research_evidence.zip!/feature_importance_with_definitions.csv
27 V22 baseline/combo fold results, 3×5; H=seed,fold,base_auc,combo_auc,delta; base .7532..7582, combo .7554..7614
  zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/fold_results.csv
28 unrelated-churn OOF summary, 2×7; H=model,mean_auc,sd_auc,mean_pr_auc,mean_log_loss,…+2; mean AUC .7029..7038
  zip/Telecom_Churn_Final_Research_Package.zip!/Deep_Experiment3_Summary.csv
29 GCI rate residual confirmation, 18×7; H=seed,base,w,gate,auc,…+2; AUC .7508..7560, gain .0001916..0007236
  zip/rate_residual_3holdout_confirmation.csv
30 selected-feature experiment, 3×8; H=outer_fold,n_test,selected_feature_count,auc,pr_auc,…+3; AUC .6994..6999
  zip/interim_feature_research_evidence.zip!/nested_selected_top8_results.csv
31 unrelated-churn core feature CV, 6×6; H=experiment,n_features,auc,pr_auc,…; AUC .6948..6995
  zip/interim_feature_research_evidence.zip!/core5_cv_raw.csv
32 V24 chunk2 result table, 6×8; H=variant,n_features,pooled_oof_auc,fold_auc,mean_fold_auc,…+3; pooled .7441..7507
  zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk2/results.csv
33 unrelated-churn previous held-out comparison, 5×11; H=model,threshold_prechosen_from_validation,test_auc,test_pr_auc,test_accuracy,…+6; stored test-AUC .6961..6994
  zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/previous_heldout_comparison.csv
34 feature/business break-even sensitivity, 27×7; H=acceptance_rate_assumed,protected_horizon_months_assumed,offer_subsidy_cost_multiplier_assumed,total_cost_usd,break_even_relative_risk_reduction_among_accepters,…+2
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/v10_2_break_even_sensitivity.csv
35 data quality summary, 7×3; H=metric,value,denominator
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/catalogue_and_data_quality_summary.csv
36 unrelated-churn feature rank, 97×4; H=feature,mi,discrete,mrmr_rank
  zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/mrmr_approx_rank.csv
37 unrelated-churn risk deciles, 10×6; H=decile_1_highest_risk,n,observed_churn,mean_score,lift_vs_test_base,…+1; score .2158..7825
  `CompanyA_Final_Submission_Package.zip!/results/risk_deciles.csv`
  = corresponding member in `CompanyA_Omnicampus_Submission_Bundle.zip` and its `(1)`–`(6)` copies
38 unrelated-churn targeted-feature results, 6×6; H=experiment,n_features,auc_mean,auc_sd,pr_auc_mean,…; mean AUC .6962..6992
  `CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/targeted_feature_optimization_results.csv`
  = same in its non-numbered ZIP; hash-identical to `interim_feature_research_evidence.zip!/targeted_domain_optimization_summary.csv`
39 V18 fast summary, 2×9; H=variant,n_features,pooled_auc,fold1_auc,fold2_auc,…+4; pooled .7561..7564
  zip/GCI_v18_alt_direction_package.zip!/GCI_v18_alt_direction/v18_fast_summary.csv
  = same member in `(1).zip`
40 adjusted odds-ratio table, 21×7; H=model,term,coef_log_odds,odds_ratio,or_ci95_low,…+2
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/adjusted_no_id_odds_ratios.csv
41 unrelated-churn action list, 1,000×9; H=Customer_ID,p_churn,rev_month,months,eqpdays,…+4
  zip/figures_and_results.zip!/action_list_top1000.csv
42 candidate validation, 6×10; H=file,rd3_rank_weight,catboost_rank_weight,champion_rank_weight,rows,…+5; unique_predictions 61,300..61,430
  zip/candidate_validation.csv
43 V24 chunk1 result table, 5×8; H=variant,n_features,pooled_oof_auc,fold_auc,mean_fold_auc,…+3; pooled .7440..7518
  zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_chunk1/results.csv
44 V29 seed20261032 results, 30×5; H=variant,pooled_auc,delta_vs_baseline,fold_auc,fold_delta_vs_baseline; pooled .7546..7647
  zip/GCI_v29_five_data_mechanisms_research.zip!/GCI_v29_package/results_seed20261032.csv
45 unrelated-churn feature-family scores, 13×7; H=experiment,features,auc_mean,auc_sd,pr_auc_mean,…+2; mean AUC .6892..7003
  `CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/domain_feature_family_results.csv`
  = same in its non-numbered ZIP; hash-identical to `interim_feature_research_evidence.zip!/domain_feature_resume_results.csv`
46 feature/business sensitivity, 162×13; H=acceptance_rate_assumed,relative_risk_reduction_among_accepters_assumed,protected_horizon_months_assumed,offer_subsidy_cost_multiplier_assumed,gross_revenue_protected_usd_modelled,…+8
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/v10_2_fixed_policy_sensitivity_162.csv
47 adjusted odds-ratio diagnostic, 20×8; H=model,n,term,coef_log_odds,odds_ratio,…+3
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/adjusted_plus_id_decile_diagnostic_only_odds_ratios.csv
48 unrelated-churn validation comparison, 9×8; H=model,roc_auc,pr_auc,accuracy_at_0_5,best_threshold_validation,…+3; ROC-AUC .6956..7082
  zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/validation_model_comparison.csv
49 unrelated-churn model holdout comparison, 2×9; H=model,n_features,roc_auc,pr_auc,accuracy_threshold_0_5,…+4; ROC-AUC .7019..7072
  `CompanyA_Final_Submission_Package.zip!/model_outputs/holdout_model_comparison.csv`
  = corresponding member in `CompanyA_Omnicampus_Submission_Bundle.zip` and its `(1)`–`(6)` copies
50 feature importance stability, 131×9; H=feature,mean_gain,gain_sd,gain_nonzero_folds,mean_splits,…+4
  zip/interim_feature_research_evidence.zip!/feature_importance_stability.csv
51 unrelated-churn action list v11, 1,000×9; H=Customer_ID,p_churn,rev_month,months,eqpdays,…+4
  zip/figures_and_results_v11.zip!/action_list_top1000_v11.csv
52 GCI research progress score table, 20×8; H=Group,Variant,Seed42_OOF_AUC,Seed7_OOF_AUC,Seed2026_OOF_AUC,…+3; stored AUC .7655..7729 (Seed42), .7695..7721 (Seed7)
  zip/GCI_Research_Progress_Results_2026-10-22.csv
53 fair incremental CV, 2×4; H=seed,base_q_TE_auc,expanded_q_interaction_auc,gain; base .7640..7647, expanded .7639..7646
  zip/fair_incremental_cv.csv
54 rate-shape anchor results, 15×5; H=fold,w,blend_auc,gain,spearman; AUC .7444..7629
  zip/rate_shape_strong_anchor_results.csv
55 feature/business financial sensitivity, 3×6; H=horizon_months,gross_revenue_scenario,contribution_at_40pct_margin,assumed_campaign_plus_offer_costs_inferred_from_v10,modeled_net_benefit_fixed_v10_target_group,…+1
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/financial_horizon_sensitivity_fixed_policy.csv
56 unrelated-churn risk-ranking score table, 8×8; H=model,oof_auc,oof_pr_auc,oof_log_loss,top_group_churn_rate,…+3; stored OOF-AUC .7029..7038
  zip/Churn_Optimisation_Decision_Package.zip!/Churn_Risk_Ranking_and_Segmentation.csv
57 unrelated-churn action list v9, 1,000×9; H=Customer_ID,p_churn,rev_month,months,eqpdays,…+4
  zip/figures_and_results_v9.zip!/action_list_top1000_v9.csv
58 feature association analysis, 56×15; H=feature,n_nonmissing,missing_pct,unique,spearman,…+10
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/development_only_feature_associations.csv
59 hard-error OOF score table, 25×8; H=fold,weight,base_auc,corr_auc,blend_auc,…+3; base .7419..7557, blend .7419..7557
  zip/hard_error_oof_results.csv
60 unrelated-churn missingness experiments, 2×12; H=name,n_features,n_train,n_val,train_rate,…+7; stored AUC .6944..6958
  zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/missingness_strategy_results.csv
61 V18 fast progress, 2×8; H=variant,n_features,pooled_auc,fold1_auc,fold2_auc,…+3; pooled .7561..7564
  zip/GCI_v18_alt_direction_package.zip!/GCI_v18_alt_direction/v18_fast_progress.csv
  = same member in `(1).zip`
62 unrelated-churn missingness by area, 20×4; H=area_cat,n,churn,change_missing
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/area_churn_missingness_development_only.csv
63 unrelated-churn paired bootstrap summary, 1×7; H=metric,point_estimate,bootstrap_2.5pct,bootstrap_median,bootstrap_97.5pct,…+2
  `Churn_Optimisation_Decision_Package.zip!/Deep_Experiment3_Paired_Bootstrap.csv`
  = same in `Telecom_Churn_Final_Research_Package.zip`
64 model benchmark, 7×4; H=model,validation_auc,validation_note,parameters; stored AUC .6564..7609
  zip/model_benchmark.csv
65 unrelated-churn action list, 1,000×9; H=Customer_ID,p_churn,rev_month,months,eqpdays,…+4
  zip/figures_and_results (1).zip!/action_list_top1000.csv = same in `(2).zip`
66 unrelated-churn reproduced model comparison, 2×9; H=variant,n_features,roc_auc,pr_auc,accuracy_thr_05,…+4; ROC-AUC .7019..7072
  `CompanyA_Final_Submission_Package.zip!/results/reproduced_model_comparison.csv`
  = corresponding member in `CompanyA_Omnicampus_Submission_Bundle.zip` and its `(1)`–`(6)` copies
67 unrelated-churn feature importance, 97×3; H=feature,gain,split; gain 0..35,550
  zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/lightgbm_feature_importance.csv
68 feature/analysis, 6×4; H=segment,n,share_dev,churn_rate
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/zero_voice_state_breakdown_dev.csv
69 CatBoost product experiment folds, 5×8; H=fold,lgb_auc,cb_plain_auc,cb_product_auc,product_minus_plain,…+3; stored LGB AUC .7675..7764
  zip/GCI_CatBoost_Product_Experiment_Package.zip!/catboost_product_experiment_folds.csv
70 GCI experiment ledger, 34×11; H=experiment_id,timestamp,model_architecture,feature_set,cv_folds,…+6; stored standalone OOF-AUC 0..7766, blended .7740..7767; 7 malformed-width rows
  zip/experminetbygeminioct22.csv
71 V18 fold comparison, 3×5; H=fold,no_interactions_auc,no_frequency_auc,rank_blend_auc,blend_vs_no_interactions; stored AUC .7505..7607
  zip/GCI_v18_alt_direction_package.zip!/GCI_v18_alt_direction/v18_fold_comparison.csv
  = same member in `(1).zip`
72 target-encoding screen, 64×6; H=seed,feature,oof_auc,matched_pct,mean_count,…+1; stored OOF-AUC .5210..6881
  zip/target_encoding_screen.csv
73 Home Credit model comparison, 5×3; H=Model,Validation AUC,Parameters; stored AUC .7334..7587
  zip/GCI_HomeCredit_BestSolution.zip!/model_comparison.csv
74 V22 combined folds, 6×5; H=seed,fold,base_auc,combo_auc,delta; base .7523..7582, combo .7544..7614
  zip/GCI_v22_tariff_bureau_combo_research_package.zip!/GCI_v22_tariff_bureau_combo/combined_fold_results.csv
75 duplicated V7 screening score table, 12×8; H=stage,rounds,method,baseline_auc,candidate_auc,…+3; stored baseline .7665..7684, candidate .7658..7688
  zip/GCI_v7_screening_results.csv = zip/GCI_v7_screening_results (1).csv
76 CA-EXT1 five-fold table, 5×14; H=fold,n_train,n_valid,base_auc,spearman,…+9; base .7576..7667
  zip/experiment_A_CA_EXT1_5fold_results.csv
77 V24 combined result table, 5×8; H=variant,n_features,pooled_oof_auc,fold_auc,mean_fold_auc,…+3; pooled .7445..7518
  zip/GCI_v24_feature_family_results.zip!/GCI_v24_feature_family_benchmark_combo/results.csv
78 aligned applicant OOF predictions, 171,202×3; H=oof_lgb,oof_xgb,target; prediction ranges .00653..65416 / .00404..71324
  zip/oof_lgb_xgb.csv
79 unrelated-churn V11 feature-family comparison, 3×9; H=feature_view,n_extra_features,mean_auc,std_auc,fold_auc,…+4; mean AUC .6926..7007
  `CompanyA_Telecom_Churn_Complete_Research_Log_and_Evidence (1).zip!/evidence/v11_cv_feature_family_comparison.csv`
  = same in its non-numbered ZIP
80 unrelated-churn feature-selection table, 6×5; H=experiment,features,roc_auc,accuracy_at_0_5,conclusion; ROC-AUC .6956..7042
  zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/feature_selection_and_engineering.csv
81 data quality focus, 13×10; H=column,n_rows,missing_n,missing_pct,negative_n,…+5
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/data_quality_focus.csv
82 GCI result variants, 24×5; H=variant,pooled_auc,delta_vs_baseline,fold_auc,fold_delta_vs_baseline; pooled .7572..7631
  zip/results (1).csv
83 V26 hypothesis summary, 5×5; H=hypothesis,pooled_auc,delta_vs_compact_baseline,fold_delta,decision; pooled .7306..7455
  `GCI_v26_optimized_five_research (1).zip!/five_hypothesis_summary.csv`
  = corresponding member in its non-numbered ZIP
84 marginal drift, 32×5; H=feature,type,KS_or_PSI,missing_or_TV,abs_SMD; 8 blank `abs_SMD` values
  zip/marginal_drift.csv
85 residual rate geometry screen, 35×7; H=fold,base,name,auc,gain,…+2; AUC .7503..7645
  zip/residual_rate_geometry_screen.csv
86 blend experiment, 3×7; H=fold,base_auc,aug_auc,best_blend_auc,best_w,…+2; base .7516..7667
  zip/results.csv
87 unrelated-churn ensemble validation variants, 5×7; H=name,weights,auc,prauc,acc05,…+2; AUC .7074..7082
  zip/telecom_churn_research_package.zip!/telecom_churn_research_package/results/ensemble_validation_variants.csv
  = zip/telecom_churn_research_round3.zip!/telecom_churn_research_round3/results/previous_ensemble_validation.csv
88 unrelated-churn action list v10, 1,000×9; H=Customer_ID,p_churn,rev_month,months,eqpdays,…+4
  zip/figures_and_results_v10.zip!/action_list_top1000_v10.csv
89 unrelated-churn ID-decile diagnostic, 10×4; H=id_decile,n,churn,change_missing
  zip/V10_MORNING_TO_NOW_RESEARCH_PACKAGE.zip!/v10_1_experiment_review/id_decile_churn_missingness_diagnostic_only.csv
90 unrelated-churn exploratory patterns, 93×8; H=pattern,segment,n,churners,rate,…+3
  zip/interim_feature_research_evidence.zip!/exploratory_patterns_min1000.csv
```

## Reproduced scores and interpretation

The local `D:\gci\inpuy\train.csv` has 171,202 rows. Its IDs are unique from `0–171201`, and it has 13,821 positive labels.

- `zip/oof_lgb_xgb.csv` includes the target vector, which matches the local target **exactly in row order**. Recomputed ROC-AUC / log-loss: **oof_lgb 0.757141977 / 0.246726160**; **oof_xgb 0.754067741 / 0.247549952**.
- `Deep_Experiment3_OOF_Predictions.csv` includes its own `churn` labels and two predictions; it is unrelated to the local Home Credit training target. Recomputed on its embedded 100,000-row labels: `oof_catalogue_traj_monet` **0.702937797 / 0.625571514**; `oof_catalogue_traj_monet_quality` **0.703784024 / 0.624774891**. The identical CSV copy has the same result.
- Other score tables contain reported fold/OOF/validation metrics and often model/variant or feature columns, but not aligned row-level labels and predictions. Their recorded metric ranges above are **claims/data in the CSV**, not local reproductions. File names and headers associate the tables with experiments such as CA-EXT, CatBoost product, V18, V22, V24, V26, V29, rate-residual, and peer ablation; they do not establish an unrecorded causal explanation.
- Submission names such as “champion,” “candidate,” and “0.77308” are filenames, not verification. No public leaderboard score can be inferred from prediction-only files.
- CSV integrity notes: root `experminetbygeminioct22.csv` has seven malformed-width rows; `marginal_drift.csv` has eight blank `abs_SMD` cells. Nested blanks occur in some experiment/importance tables; otherwise rows were read to EOF without malformed widths. Exact copies are grouped above.

This is the CSV-specific follow-up. The previously reported NPY/NPZ findings are unchanged; in particular, `GCI_hgb_product_test.npy` is test-only and cannot establish a score.

## Phase 52 — GPU translation of the v7 label-free feature block

Because the archived v7/v13 code uses LightGBM and HistGradientBoosting,
which cannot run on this installed GPU stack, I ran a clearly labeled GPU
translation rather than call a CPU run a GPU result. The translation compares
the same raw predictors against those predictors plus the deterministic,
label-free feature families extracted from v7. It is not a reproduction of
the archived LightGBM blend and does not include v13's learned residuals,
nested target encodings, decoded tariff keys, ExtraTrees, or caches.

- Runner: `experiments/phase52_gpu_archive_port/run_phase52_gpu_archive_port.py`
- Tests: `experiments/phase52_gpu_archive_port/test_phase52_gpu_archive_port.py`
- Run outputs: `experiments/phase52_gpu_archive_port/runs/phase52_gpu_20261108`
- Data: local 171,202-row training CSV; SHA-256
  `b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1`.
- Validation: same five stratified folds, seed `20261113`, for both feature
  sets; outer validation rows were not used to fit category encoders.
- Model: XGBoost 3.4.1, `tree_method="hist"`, `device="cuda"`, 600 fixed
  rounds, 6-depth trees. No public leaderboard score is estimated.
- Input features: raw application fields plus missingness/employment-sentinel
  features, EXT score aggregates/products/differences, financial ratios and
  log amounts, age/employment ratios, and bureau inquiry summaries.
- Intentionally excluded: constants `129.5` and `0.0132`, target encodings,
  combined train/test frequency/group statistics, external data, and
  row-specific prediction adjustment.

| Five-fold OOF model | Pooled ROC-AUC | Log-loss | Fold AUCs |
|---|---:|---:|---|
| Raw predictors, one-hot categories | 0.752413 | 0.247813 | 0.755058, 0.759881, 0.750151, 0.747978, 0.749077 |
| Raw plus v7-style deterministic features | 0.758640 | 0.246102 | 0.762967, 0.764211, 0.755609, 0.756372, 0.754151 |
| Matched difference | **+0.006227** | **-0.001711** | **+0.007909, +0.004329, +0.005458, +0.008394, +0.005074** |

The aggregate feature block improved AUC in all five matched folds and lowered
log-loss. This supports a real, reproducible signal from the added deterministic
feature bundle in this XGBoost configuration. It does **not** isolate which
feature family caused the gain; that needs separate predeclared ablations.
Fold 3 has the lowest AUC for both variants, while the feature bundle still
improves that fold, so the fold-level decline is not evidence that the new
features harmed it. The model remains well below the archived claimed
0.77–0.78 public scores; differences in learner, features, tuning, and
validation protocol preclude direct comparison.

The runner's two tests pass. A device smoke test succeeded for CatBoost GPU
and XGBoost CUDA; it failed for LightGBM because its installed build lacks the
GPU learner. Installed PyTorch is CPU-only. Therefore, the original v7/v13
pipelines were not GPU-run unchanged; the Phase 52 score is a verified GPU
translation with the limitations above, not their score.

### Phase 52 feature-family ablations

To identify which parts of the deterministic bundle account for its lift, I
ran one additional matched five-fold CUDA comparison. Each family variant
starts with the same raw one-hot baseline and adds exactly one partition of the
engineered columns; the full v7-style variant adds the complete bundle. The
data, fixed folds and seed, 600-round XGBoost CUDA setup, and OOF scoring
protocol are unchanged. These are exploratory single-seed results, not
independent confirmation or causal feature attribution.

| Variant (raw baseline plus...) | Pooled OOF ROC-AUC | Log-loss | AUC delta vs raw | Log-loss delta vs raw | Folds with AUC gain |
|---|---:|---:|---:|---:|---:|
| Raw one-hot predictors | 0.752413 | 0.247813 | +0.000000 | +0.000000 | 0/5 |
| Clean/missingness features | 0.752526 | 0.247783 | +0.000113 | -0.000030 | 3/5 |
| External-score geometry | 0.751169 | 0.248027 | -0.001244 | +0.000213 | 0/5 |
| Financial ratios and log amounts | **0.759889** | **0.245761** | **+0.007476** | **-0.002052** | **5/5** |
| Age/employment features | 0.751708 | 0.248008 | -0.000706 | +0.000195 | 1/5 |
| Bureau activity | 0.752035 | 0.247878 | -0.000378 | +0.000064 | 2/5 |
| Full v7-style deterministic bundle | 0.758640 | 0.246102 | +0.006227 | -0.001711 | 5/5 |

Within this experiment, financial ratios and log-amount features explain the
clearest and most stable standalone gain: AUC improved in every fold. Their
pooled AUC is 0.001249 higher than the full feature bundle, so adding every
other family did not improve on this financial-only addition in this model and
fold setup. The clean/missingness group has only a very small pooled gain;
external-score geometry, age/employment, and bureau groups alone do not help
the raw baseline here. This does not show that those families are universally
unhelpful: useful interactions may require the other engineered features, and
the result may depend on the learner, seed, or fold assignment.

The family groups are: missingness/sentinel indicators; external-score
aggregates and geometry; credit, annuity, goods, income-per-family and
log-amount features; age/employment/age-normalized features; and bureau inquiry
summaries. The variants retain all original predictors, so “financial only”
means raw predictors plus the financial family, not a model trained on the
financial features alone. Exact fold-level predictions and metrics are saved
in `experiments/phase52_gpu_archive_port/runs/phase52_gpu_family_ablation_20261108/phase52_oof.csv`
and `phase52_summary.json`. The test suite and Python compilation check passed
after adding the feature-family partition coverage.
