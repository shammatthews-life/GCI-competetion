# Official competition materials audit

## Scope and authority

Reviewed every competition file present under `D:\gci\rules and info`:

- `04. Competition (due Nov 20)\Competition\README.ipynb`, including assignment,
  evaluation, and rules cells.
- `04. Competition (due Nov 20)\Competition\tutorial.ipynb`.
- `04. Competition (due Nov 20)\Competition\HomeCredit_columns_description.xlsx`.
- `04. Competition (due Nov 20)\competition_tutorial.pdf` (23 pages).
- The adjacent sample submission and supplied train/test CSVs.
- `past year winner class\tutorial_model_ensembling.ipynb`, treated as
  instructional material, not a rule source.

No separate PowerPoint, organizer disclosure form, pretrained-model policy,
or additional rule document was present. The course README is the primary
rule text in the supplied package. No live organizer portal or later
organizer-issued revision was independently checked. Thus the local rule
materials were read completely; matters not addressed in them remain
unresolved rather than being inferred from another participant's report.

## Rule-by-rule findings

| Topic | Primary source and location | Rule / accurate summary | Permits | Prohibits or limits | Effect on audited code | Status |
|---|---|---|---|---|---|---|
| Permitted data | `README.ipynb`, markdown cell 4, “Prohibition of External Data Usage” | No external data at any analysis stage; use only the provided `input/train.csv` and `input/test.csv`. | The distributed training and test files and reproducible transformations of their columns. | Outside datasets, labels, or other data sources. | The UMAP notebook searches broad directories and does not pin/hash the official CSVs. The archive reproducer consumes precomputed arrays whose upstream data lineage is absent. | **UNRESOLVED for the friend pipeline**; clear rule, insufficient input provenance. |
| Hand-labeling / manual predictions | `README.ipynb`, markdown cell 4, “Prohibition of Hand-Labeling” | Predictions must be generated automatically by a reproducible model; manual decisions for any subset based on EDA-derived rules are also hand-labeling. | Automated predictions from a reproducible model. | Manual labels or EDA-driven manual prediction decisions/overrides. | The archive script algorithmically couples scores for selected test pairs using hand-specified entity conditions and a fixed coupling coefficient. Whether the organizer considers this a prohibited EDA-rule adjustment requires clarification; do not submit this method pending an answer. | **AMBIGUOUS — high risk**. |
| Reproducibility | `README.ipynb`, markdown cell 4, “Ensuring Reproducibility” | Reproduce predictions as much as possible and set random seeds. | Seeded, documented stochastic modeling. | No specific package lock or exact reproducibility threshold is stated. | UMAP notebook sets seeds in many model/transform paths, but input discovery is ambiguous and it has no executed outputs. The archive script has hard-coded machine-specific paths and missing inputs. | **FAIL for reproducible delivery as supplied**. |
| Referenced code | `README.ipynb`, markdown cell 4, “Citing Original Sources (Reference Code)” | Referencing code is allowed; explicitly record the original source. Top-ranking participants must disclose code; missing citations can jeopardize awards. | Cited references and attributed reuse. | Undisclosed copied code may jeopardize recognition. | The friend's supplied artifacts give no complete external-source/license record. Preserve attribution and obtain any needed license/author permission before reuse. | **UNRESOLVED**. |
| Private sharing / collaboration | `README.ipynb`, markdown cell 4, “Permission for Private Sharing” | The usual private-sharing prohibition is waived for this tutorial competition; advice and participant discussion are encouraged. | Private discussion/sharing under this competition's exception. | This exception does not waive the data, prediction, or citation rules. | Receipt of a friend's source is not itself a violation; its methods still require an independent compliance audit. | **ALLOWED with other controls**. |
| Feature engineering | Competition tutorial PDF, slide 22; README data and external-data rule | The deck encourages meaningful features from existing features. The README still limits data to supplied files. | Row-wise formulas and transformations from supplied fields. | No outside facts/data may be smuggled in as features. | Financial/EXT_SOURCE formulas using provided columns are not inherently disallowed. Undocumented cutoffs/weights need derivation records; full-data fitting has validation implications. | **ALLOWED when sourced; some constants UNVERIFIED**. |
| Transductive / test-covariate statistics | README cells 2 and 4 | Test data are described as evaluation data; the no-external-data rule names provided train and test files but does not state whether fitting on unlabeled test covariates is permitted. | Use of provided test rows for final inference. | No explicit rule in the supplied text specifically resolves pooling test covariates during feature fitting. | The UMAP notebook pools train/test categories and group statistics and transforms both sets jointly. This is not outside data, but permission for fitting those statistics is unstated. | **AMBIGUOUS — ORGANIZER CLARIFICATION REQUIRED**. |
| Target encoding / supervised representation | README cells 2 and 4 | Training labels belong to the provided train file; no separate target-encoding or supervised-embedding clause appears. | Model training from supplied training labels. | External labels and leakage into evaluation are not allowed by the stated data boundary; fold methodology is not otherwise specified. | Target encoding and supervised UMAP use training labels. The implementation is not nested inside the outer evaluation folds, invalidating its OOF evidence even though the rule text does not expressly ban these techniques. | **UNRESOLVED for rule scope; VALIDATION FAIL in implementation**. |
| Cross-validation / early stopping | README evaluation cell 3 and tutorial notebook | ROC-AUC is the metric. The tutorial is an instructional single-holdout example; no required CV design is specified in the rules. | Any reproducible model-selection method consistent with data and hand-label rules. | The rules do not certify OOF scores as unbiased. | Early stopping on the same outer fold used for reported AUC and global feature fitting make the friend's OOF claims unreliable. | **VALIDATION FAIL; not by itself a proven eligibility violation**. |
| Pretrained models / cached predictions | README cell 4 external-data rule; no specific model-artifact clause found | No explicit pretrained-model or prediction-cache policy is provided. If a model/cache embeds outside data, it conflicts with the external-data prohibition. | Not established by supplied text. | External data remain prohibited regardless of packaging. | The archive script requires many unprovided OOF/test arrays and a precomputed feature matrix; provenance cannot be checked. | **UNVERIFIED; do not use until provenance is supplied**. |
| Submission format and metric | `README.ipynb`, markdown cell 3, “Evaluation” | ROC-AUC; follow `sample_submission.csv` with `SK_ID_CURR,TARGET`; malformed submission scores -1. | Probability submissions in the supplied schema. | Incorrect schema is invalid. | The friend README claims the schema and row count; the bundled CSV was not opened or independently checked. The reproduction code writes the two columns but has no explicit ID/hash/schema assertions. | **UNVERIFIED for supplied CSV; source code partially conforms**. |
| Public/private leaderboard | `README.ipynb`, markdown cell 3, “Scoring” | Public score uses a subset; final private score uses all test rows; last submitted file determines ranking. | Use public feedback as competition scoring. | Public score is not a full-test estimate; repeated tuning risks leaderboard overfit. | Friend README's public score is a claim only; it is not evidence that the full-test result or OOF procedure is valid. | **CAUTION**. |
| Honors/Outstanding reproduction | `README.ipynb`, markdown cell 3 | Code capable of reproducing the submitted CSV is required for these recognitions. | Reproducible code submission. | A prediction file without reproducible source is insufficient for this recognition. | The archive's script is a synthesis/replay entry point requiring many absent prediction and feature artifacts, not a raw-data end-to-end reproduction. | **FAIL as supplied**. |
| Third-party licensing | README cell 4 allows references but does not specify licenses. | Citation is explicit; licensing terms are not detailed in the local competition rules. | Cited code subject to applicable license/permission. | This audit cannot determine licenses from the provided sources. | Archive has no license manifest or full base-model source. | **UNVERIFIED**. |
| AI assistance disclosure | No disclosure rule found in supplied README, tutorial, deck, or dictionary. | No local instruction located. | Not determined. | Not determined. | Ask organizers if the platform requires disclosure not included in this package. | **UNVERIFIED**. |

## Tutorial and dictionary implications

The beginner tutorial illustrates preprocessing, feature engineering, model
training, probability ensembling, and writing a submission. It fits some
preprocessing statistics before its holdout split, so it is not a reliable
template for unbiased OOF evaluation. The data dictionary describes the
provided columns; using a supplied `EXT_SOURCE_*` column is not the same as
obtaining an external dataset. The course deck explicitly encourages features
derived from existing features, but does not authorize external data or
manual prediction edits.

## Audit boundary

This is an audit of supplied course materials and code, not an organizer
eligibility ruling. The key clarification to obtain in writing is whether
unlabeled train/test pooling and the friend solution's feature-driven,
test-to-test score coupling are permitted. Until then, avoid those methods in
a competition submission.
