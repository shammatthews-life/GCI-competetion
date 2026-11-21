from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(r"D:\gci")
OUT = ROOT / r"truth\experiments\phase5_champion_reproduction\runs\static_audit_20261104_0501"
TRAIN = ROOT / r"inpuy\train.csv"
TEST = ROOT / r"inpuy\test.csv"
V16_EXECUTED = ROOT / r"pre info\oct22\GCI_v16_everything_merged_o_78222 score.ipynb"
V16_SOURCE_ONLY = Path(r"D:\exp colab\GCI_v16_everything_merged.ipynb")
CHAMPION_SOURCE = ROOT / r"champion\exp85_solo_runner.py"
CHAMPION_OUTPUT = ROOT / r"champion\submission_exp85_solo_champion.csv"
CHAMPION_REFERENCE = ROOT / r"champion\00_HISTORICAL_REFERENCE\EXP85_HISTORICAL_CHAMPION.csv"
V17_BUNDLE = ROOT / r"champion\EXP85_FINAL_RECONSTRUCTION\runs\recovered_v17_1793443853760\v_17_full_suite_bundle.npz"
V55_BUNDLE = ROOT / r"champion\EXP85_FINAL_RECONSTRUCTION\runs\recovered_v55_1793443036653\v55_members.npz"
ANCHOR_OOF = ROOT / r"champion\EXP85_FINAL_RECONSTRUCTION\runs\recovered_anchor74_1793444526421\anchor74_oof.npy"

DATA_HASHES = {
    r"train.csv": "b8dd5a61771d8eb2821972c1b7d3f2f747cc66471f4d34153036d2d9c458ecf1",
    r"test.csv": "dfb7aeb96e5af3d8616acaaba6b23cec8680845cfd525f7c08de15c6d2d42c65",
}

FEATURES = [
    ("raw_base", "Raw columns except SK_ID_CURR; eight categorical columns are cast to category and jointly coded across train+test.", "Cell 3", "The TARGET column is dropped before train/test concatenation. Category coding uses pooled covariates."),
    ("cleaning_and_missingness", "DAYS_EMPLOYED==365243 sentinel flag and NaN replacement; ORGANIZATION_TYPE=='XNA'; OWN_CAR_AGE>=60; row missing count; missing flags for EXT_SOURCE_1/3, OWN_CAR_AGE, AMT_REQ_CREDIT_BUREAU_YEAR.", "Cell 3", "Rowwise formulas. Missing count includes predictors only because TARGET was dropped."),
    ("external_score_moments", "EXT mean/min/max/std/range/observed count/product; weighted score (2*EXT1+3*EXT2+4*EXT3)/9; pairwise products E12/E13/E23 and differences D12/D13/D23.", "Cell 3", "Mean/min/max/std skip missing values; pandas std defaults to ddof=1 and product defaults to min_count=0 (all-missing can yield 1). EXT_W is NaN if a component is missing."),
    ("financial_ratios_and_logs", "CREDIT_INC=credit/income; ANN_INC=annuity/income; ANN_CREDIT=annuity/credit; TERM=credit/annuity; GOODS_CREDIT=goods/credit; CREDIT_MINUS_GOODS; GOODS_INC=goods/income; income/credit/annuity per family member; log1p of four amount fields.", "Cell 3", "No explicit denominator guards in these ratios; later replaces +/-infinity with NaN. Family size and amount semantics are not validated by this code."),
    ("age_employment_ratios", "AGE=-DAYS_BIRTH/365.25; cleaned employment days; EMP_AGE, EMP_YEARS, REG_AGE, ID_AGE, PHONE_AGE; car age, credit and income per age.", "Cell 3", "Negative-day convention is used. Sentinel employment is replaced by NaN before employment ratios."),
    ("bureau_summaries", "BUR_SUM=sum of all AMT_REQ* columns; BUR_RECENT=hour+month requests; BUR_MON_YEAR=month/(year+1).", "Cell 3", "The hour/month/year measures have different windows; their ratios are an unvalidated feature hypothesis."),
    ("pooled_frequency_counts", "Value counts for the eight categorical columns plus REGION_POPULATION_RELATIVE, DAYS_BIRTH, AMT_CREDIT, AMT_GOODS_PRICE, AMT_ANNUITY, AMT_INCOME_TOTAL.", "Cell 3", "Counts are computed over concatenated train+test, so validation/test covariates affect feature values."),
    ("pooled_group_deviations", "Per-row value minus pooled group mean for seven values (income, credit, EXT1/2/3, birth days, credit/income) across ORGANIZATION_TYPE, OCCUPATION_TYPE, NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, REGION_POPULATION_RELATIVE.", "Cell 3", "Group means include validation/test covariates; this is not fold-local preprocessing."),
    ("k_decoded_markup", "Cash loans: k=round(((credit/goods)-1)/0.0132) if credit>goods and distance to integer <0.1; cash with credit==goods maps to 0; also credit/goods.", "Cell 3", "0.0132 grid and tolerance are unverified assumptions. No proof this is a contractual tariff tier."),
    ("fee_adjusted_and_community_keys", "q=annuity/credit; a global frequency comparison selects FEE where rounded((annuity-129.5)/credit) occurs more often than rounded(q), for cash loans; Q0 is adjusted q or q. Rounded Q/P keys at 3/4/5 decimals, fee flag, credit/annuity keys at 1/2/3 decimals, amount combinations and smoothed target encodings.", "Cell 3", "129.5 and 0.0132 are hypotheses. Frequency selection pools train+test. Target encodings use training labels with nested encoding for train rows."),
    ("partner_matching", "Strict identity fingerprint from gender, education, region, mobile flag, and registration/birth and ID-publish/birth gaps; loose fingerprint uses gender plus the two date gaps. Features include match existence/count and nearest birth-day gap.", "Cell 3", "Heuristic entity resolution; false-positive/semantic risk. Link construction is pooled over train+test."),
    ("partner_target_encodings", "For matched rows, maximum observed TARGET among fit-set partner rows, partner count, within-180-day maximum, earlier/later-application maxima; separate loose-key maximum.", "Cell 3", "Fit labels are limited to supplied training labels and nested in the code. Cross-fitting is a positive control; entity key construction remains pooled."),
    ("learned_residual_features", "15 HistGradientBoostingRegressor targets (EXT scores, log income/credit, employment, registration, age, ID/phone dates, region population, family/car and three EXT targets from demographics/amounts); RP predictions and observed-minus-prediction RR residuals.", "Cell 5", "Five-fold KFold predictions are computed over combined train+test covariates, not inside the outer TARGET folds. Reused residual caches have no source/input hash checks."),
    ("multi_view_oof_features", "Outer-fold inner predictions for EXT, loan, person views; logit transforms and view differences; nested within training partitions.", "Cell 7", "Inner models are fitted without the outer validation labels. Feature caches can be reused by filename without validating provenance."),
    ("model_family_and_blending", "Four LightGBM sets (old/e1/e1et/e1Aet), five members per set, plus fold-averaged CatBoost product model; fixed within-set weights and fixed top-level rank blend.", "Cells 9, 11, 14", "Weights are hard-coded, but their selection history is not evidenced. OOF folds use seed 42."),
    ("optional_ablation_and_regime_blend", "HistGradientBoosting 5-fold feature-block ablation; optional cross-fitted blend weight by EXT observation count and contract type.", "Cells 16, 20", "Saved outputs are claims only until independently rerun. The regime design is hand-specified and the result is not the fixed submission blend."),
]

MODELS = [
    ("LightGBM qf_shallow", "Core view", "lr=.05; leaves=7; max_depth=3; min_child_samples=150; feature_fraction=.5; bagging_fraction=.8; lambda_l2=20", "1300", "1430", "42,7,2024,11,99"),
    ("LightGBM qf_reg", "Core view", "lr=.03; leaves=11; min_child_samples=300; feature_fraction=.4; bagging_fraction=.8; lambda_l2=40", "1939", "2133", "42,7,2024,11,99"),
    ("LightGBM v1_allfam_oh", "All-family one-hot view", "lr=.04; leaves=15; min_child_samples=150; feature_fraction=.3; bagging_fraction=.8; lambda_l2=20", "881", "969", "42,7,2024"),
    ("LightGBM v2_allfam_oh", "All-family one-hot + K_DEC", "Same as v1_allfam_oh", "881", "969", "42,7,2024"),
    ("LightGBM v2_mono", "Core + K_DEC; monotone EXT constraints", "lr=.02; leaves=15; min_child_samples=150; feature_fraction=.5; bagging_fraction=.8; lambda_l2=20; EXT_SOURCE_1/2/3 constraints=-1", "1571", "1729", "42,7,2024"),
    ("CatBoost product", "Extended numeric + native categorical product keys", "iterations=900; depth=7; lr=.05; l2_leaf_reg=8; random_strength=.5; Bernoulli subsample=.8; seed=42", "900", "900", "42"),
    ("Final V16 blend", "Within-set weights: .24/.12/.32/.16/.16; average old/e1/e1et groups; blend 75/25 with e1Aet; 90/10 rank blend with CatBoost", "Fixed formula in notebook source", "not applicable", "not applicable", "seed 42 outer folds"),
]

ALIASES = [
    ("Financial ratios", "Credit/income, annuity/income, annuity/credit, credit/annuity, goods/credit, per-family amounts", "Phase 3C/Phase 2 have the same concepts; ratio direction and missing/zero policies vary.", "EXP85 Anchor/V17/V55 feature source contains overlapping ratios and product transforms.", "Mostly represented; not novel as raw information. Denominator guards and imputation differ."),
    ("External-score summaries", "Mean/min/max/std/product, pair products/differences, weighted score", "Phase 2 includes mean/min/max/product and other ext interactions.", "EXP85 runner/Anchor74 have related summaries and pair interactions; weights and missing treatment differ. Its population standard deviation differs from V16 pandas sample std; product semantics also differ.", "High conceptual overlap; definitions are not exact aliases."),
    ("Employment/age/missingness", "365243 sentinel, age/employment/date ratios, missing flags/count", "Existing baseline uses sentinel cleaning and missing indicators.", "Anchor74/EXP85 includes cleaned employment, age/date ratios, missing count and flags.", "High overlap; formula scaling/denominators differ."),
    ("Bureau summaries", "Request totals, recent hour+month sum, month/(year+1)", "Existing raw baseline includes raw bureau columns; exact summary alias not fully established here.", "EXP85 feature builders include BUR_RECENT; exact full summary overlap not established.", "Partial/unknown; not a new source field."),
    ("Frequency counts and group deviations", "Pooled value counts and deviations from category means", "Exact Phase 3C overlap not established in this audit.", "V17/V55 recovered builders include frequency and pooled group deviation families.", "Known champion-family overlap; global pooling remains a validation concern."),
    ("K_DEC and product fee keys", "0.0132 markup grid; Q/P keys and 129.5 fee inference", "Phase 2 product model has product features; exact keys differ.", "V17/V55/EXP85 recovered sources contain K_DEC, Q/P-like product keys and fee inference.", "Strong overlap; constants remain hypotheses, not established loan terms."),
    ("Partner links and supervised partner encodings", "Strict/loose fingerprints plus partner label maxima", "No exact alias established in Phase 3C.", "EXP85 Anchor uses a different entity key and label encoding; concept overlaps, implementation differs.", "Not proven novel at concept level; high false-match and rule-review risk."),
    ("Residual and multi-view model features", "Cross-fitted learned expectations, residuals, EXT/loan/person views", "Not established as present in Phase 3C baseline.", "V17/V55 are derived from the same model lineage and have overlapping view/residual artifacts.", "Potentially distinctive, but absent cache provenance blocks measured attribution."),
]

BLOCKED_REASON = (
    "The frozen EXP85 test submission is available and byte-identical to the historical reference, "
    "but the exact final 85/15 Anchor74+eight-engine OOF vector and its fold map are not available. "
    "The source's Anchor74 contribution is a test-only transductive prediction; available V17/V55/XGB "
    "OOF arrays are upstream components and do not by themselves reproduce the final champion's OOF. "
    "Using Phase 2/3C or component OOF as a substitute would violate the requested comparator."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_shape(path: Path) -> tuple[int, int, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = sum(1 for _ in reader)
    return rows, len(header), header


def read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_source_hash(notebook: dict) -> str:
    cells = [
        {"cell_type": cell.get("cell_type"), "source": "".join(cell.get("source", []))}
        for cell in notebook["cells"]
    ]
    payload = json.dumps(cells, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def saved_output_lines(notebook: dict) -> list[tuple[int, str]]:
    rows = []
    for cell_index, cell in enumerate(notebook["cells"], start=1):
        for output in cell.get("outputs", []):
            text = output.get("text", [])
            if isinstance(text, list):
                text = "".join(text)
            if not text:
                data = output.get("data", {})
                text = data.get("text/plain", [])
                if isinstance(text, list):
                    text = "".join(text)
            if text:
                for line in str(text).splitlines():
                    if re.search(r"\b(AUC|score|fold|permutation|adversarial|held-out|delta)\b", line, re.I):
                        rows.append((cell_index, line.strip()))
    return rows


def write_csv(name: str, headers: list[str], rows: Iterable[Iterable[object]]) -> None:
    path = OUT / name
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def write_text(name: str, text: str) -> None:
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def run() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        print("PHASE 5 STATIC SOURCE / DATA / ARTIFACT AUDIT")
        print(f"UTC timestamp: {datetime.now(timezone.utc).isoformat()}")
        print(f"Python: {sys.version.split()[0]} ({platform.python_implementation()})")
        print(f"Command: python -B {Path(__file__).resolve()}")
        print("Model training: NOT RUN (pre-run comparator gate blocked)")
        print("Correction: this run preserves the prior audit outputs and uses the verified V55 bundle path.")

        for path in (TRAIN, TEST):
            rows, columns, header = csv_shape(path)
            observed_hash = sha256(path)
            expected_hash = DATA_HASHES[path.name]
            assert observed_hash == expected_hash, f"Input hash mismatch: {path}"
            print(f"INPUT {path}: rows={rows}, columns={columns}, sha256={observed_hash}")
            if path == TRAIN:
                assert rows == 171202 and columns == 34 and "TARGET" in header
            else:
                assert rows == 61500 and columns == 33 and "TARGET" not in header

        executed = read_notebook(V16_EXECUTED)
        source_only = read_notebook(V16_SOURCE_ONLY)
        executed_hash = sha256(V16_EXECUTED)
        source_only_hash = sha256(V16_SOURCE_ONLY)
        code_cells = [cell for cell in executed["cells"] if cell.get("cell_type") == "code"]
        executed_cells = sum(cell.get("execution_count") is not None for cell in code_cells)
        source_hash_exec = canonical_source_hash(executed)
        source_hash_clean = canonical_source_hash(source_only)
        assert source_hash_exec == source_hash_clean
        print(f"V16 executed notebook: {V16_EXECUTED} sha256={executed_hash}")
        print(f"V16 source-only notebook: {V16_SOURCE_ONLY} sha256={source_only_hash}")
        print(f"V16 canonical source hash (both copies): {source_hash_exec}")
        print(f"V16 code cells={len(code_cells)}; cells with execution counts={executed_cells}")

        champion_hash = sha256(CHAMPION_OUTPUT)
        reference_hash = sha256(CHAMPION_REFERENCE)
        assert champion_hash == reference_hash
        assert V17_BUNDLE.is_file()
        assert V55_BUNDLE.is_file()
        print(f"EXP85 frozen submission: {CHAMPION_OUTPUT} sha256={champion_hash}")
        print(f"Historical reference: {CHAMPION_REFERENCE} sha256={reference_hash}")
        print(f"EXP85 source runner: {CHAMPION_SOURCE} sha256={sha256(CHAMPION_SOURCE)}")
        print(f"Exact final anchor+suite OOF file exists: {any((ROOT / r'champion').rglob('exp85_solo_champion_oof.*'))}")
        print(f"Recovered Anchor74 OOF array exists at expected run path: {ANCHOR_OOF.exists()}")
        print(f"V17 component bundle exists: {V17_BUNDLE.exists()}")
        print(f"V55 component bundle exists: {V55_BUNDLE.exists()}")
        print(f"V55 component bundle path: {V55_BUNDLE}")
        print(f"Audit script exists: {Path(__file__).resolve().is_file()}")

        output_claims = saved_output_lines(executed)
        write_csv(
            "V16_SAVED_OUTPUT_EVIDENCE.csv",
            ["notebook_cell", "saved_output_line", "evidence_status"],
            [(cell, line, "SAVED_NOTEBOOK_OUTPUT_NOT_INDEPENDENTLY_REPRODUCED") for cell, line in output_claims],
        )
        print(f"Saved V16 output lines matching metric/score diagnostics: {len(output_claims)}")
        for cell, line in output_claims:
            print(f"  notebook cell {cell}: {line}")

        write_csv(
            "V16_FEATURE_EXTRACTION.csv",
            ["feature_family", "implementation_as_written", "source_cell", "audit_notes"],
            FEATURES,
        )
        write_csv(
            "V16_MODEL_CONFIGURATION.csv",
            ["model", "input_view", "parameters", "outer_training_rounds", "full_refit_rounds", "seeds"],
            MODELS,
        )
        write_csv(
            "V16_SAVED_OUTPUT_SUMMARY.csv",
            ["comparison_or_diagnostic", "saved_auc_or_value", "saved_delta", "evidence_status", "interpretation"],
            [
                ("V16 fixed final blend", "0.77649", "", "SAVED_NOTEBOOK_OUTPUT_NOT_INDEPENDENTLY_REPRODUCED", "Not champion-comparable; no cached OOF vector or input manifest."),
                ("Ablation base, no label blocks", "0.76125", "+0.00000", "SAVED_NOTEBOOK_OUTPUT_NOT_INDEPENDENTLY_REPRODUCED", "HistGradientBoosting ablation baseline."),
                ("Ablation + V15 product keys", "0.77185", "+0.01060", "SAVED_NOTEBOOK_OUTPUT_NOT_INDEPENDENTLY_REPRODUCED", "Largest reported lift in the saved block ablation."),
                ("Ablation + community CT keys", "0.77206", "+0.01081", "SAVED_NOTEBOOK_OUTPUT_NOT_INDEPENDENTLY_REPRODUCED", "Only +0.00021 beyond product keys in the saved outputs."),
                ("Ablation + community CT keys + partner", "0.77307", "+0.01182", "SAVED_NOTEBOOK_OUTPUT_NOT_INDEPENDENTLY_REPRODUCED", "Additional +0.00101 beyond the community-only block."),
                ("Target permutation diagnostic", "0.5014", "", "SAVED_NOTEBOOK_OUTPUT_NOT_INDEPENDENTLY_REPRODUCED", "Consistent with no obvious encoding leak in that saved run; not a current reproduction."),
                ("Adversarial train/test diagnostic", "0.6856", "", "SAVED_NOTEBOOK_OUTPUT_NOT_INDEPENDENTLY_REPRODUCED", "If reproduced, indicates substantial covariate separability, not target signal."),
                ("Cross-fitted regime blend", "0.77648", "-0.00001 vs fixed 0.9/0.1 blend", "SAVED_NOTEBOOK_OUTPUT_NOT_INDEPENDENTLY_REPRODUCED", "Saved result does not improve the fixed blend."),
            ],
        )
        write_csv(
            "FEATURE_ALIAS_AUDIT.csv",
            ["feature_family", "V16_formula_or_block", "Phase2_Phase3C_overlap", "EXP85_champion_source_overlap", "conclusion"],
            ALIASES,
        )
        write_csv(
            "FOLD_LEVEL_CV_RESULTS.csv",
            ["fold", "candidate", "auc", "baseline_auc", "delta_auc", "status", "reason"],
            [("", "V16 vs exact EXP85 champion", "", "", "", "BLOCKED", BLOCKED_REASON)],
        )
        write_csv(
            "BASELINE_VS_FEATURES_RESULTS.csv",
            ["comparison", "baseline_auc", "candidate_auc", "delta_auc", "folds", "status", "reason"],
            [
                ("Exact EXP85 champion vs V16", "", "", "", "", "BLOCKED", BLOCKED_REASON),
                ("Exact EXP85 champion vs V16 feature ablations", "", "", "", "", "BLOCKED", BLOCKED_REASON),
            ],
        )
        write_csv(
            "MISSING_ARTIFACTS.csv",
            ["required_artifact", "status", "searched_location_or_evidence", "effect"],
            [
                ("Final EXP85 85/15 Anchor74+suite OOF prediction vector", "NOT LOCATED", str(ROOT / r"champion\EXP85_FINAL_RECONSTRUCTION\runs") + "; runner exports only test submission", "Cannot calculate champion-matched OOF AUC or ablation delta."),
                ("Fold assignment for the exact final EXP85 OOF vector", "NOT LOCATED", str(ROOT / r"champion\EXP85_FINAL_RECONSTRUCTION\runs") + "; exact final vector absent", "Cannot align identical folds for candidate comparisons."),
                ("Final Anchor74 OOF vector corresponding to frozen transductive test Anchor74", "NOT LOCATED", str(ANCHOR_OOF), "Anchor74 test anchor has no matched OOF counterpart; available component OOF is not equivalent."),
                ("V16 cached residual/view/model arrays and cache manifests", "NOT PRESENT IN AUDITED LOCAL PATHS", r"D:\gci\gemini\gci_cache_v16; D:\gci\gemini\gci_cache_final; D:\gci\gemini\gci_cache_v6; D:\gci\pre info\oct22\gci_cache_v16", "Exact cached V16 run cannot be reproduced from the supplied notebook alone."),
                ("V16 input hashes and cache provenance manifest", "NOT RECORDED IN NOTEBOOK", str(V16_EXECUTED), "Saved output numbers cannot be bound to exact raw files/cache contents."),
            ],
        )
        write_csv(
            "V16_RULES_COMPLIANCE.csv",
            ["method", "local_rule_status", "source_audit_evidence", "action"],
            [
                ("External datasets", "PROHIBITED by local official materials; none found in V16 source", "Notebook reads competition train/test/sample by shape/schema; this audit verified local input hashes.", "Keep data lineage pinned."),
                ("Hand-labeling", "PROHIBITED by local official materials", "V16 uses models and label encodings; EXP85 upstream also applies fixed test-to-test entity score coupling.", "No manual prediction edits; obtain organizer ruling on test-to-test coupling."),
                ("Pooled unlabeled train/test feature fitting", "UNRESOLVED by supplied materials", "V16 computes frequencies, group means, product frequencies, entity groups, and residual transforms over combined covariates.", "Request written clarification before competition use."),
                ("Target encoding", "Rule scope UNRESOLVED; validation implementation is nested in inspected V16 code", "Outer validation labels are excluded from enc(a,b); train encodings are nested in outer training rows.", "Obtain organizer clarification and preserve nested folds."),
                ("ROC-AUC and submission schema", "PERMITTED/REQUIRED by supplied materials", "Frozen CSV uses SK_ID_CURR,TARGET and 61,500 unique IDs.", "Do not infer a score from correlation or filename."),
                ("Source attribution/reproducibility", "REQUIRED in supplied materials; V16 CACHE PROVENANCE INCOMPLETE", "V16 cache loads do not verify hashes/schema/folds; cache files are not present in the audited paths.", "Recover caches with source/data/fold manifests before replay."),
            ],
        )

        v16_saved_claims = "\n".join(f"- Cell {cell}: `{line}` (saved output only; not independently reproduced)." for cell, line in output_claims)
        if not v16_saved_claims:
            v16_saved_claims = "- No saved output lines matching the selected metric terms were found."
        write_text(
            "SOURCE_CODE_AUDIT.md",
            f"""# Source-code and artifact audit

## V16 source identity

- Executed notebook: `{V16_EXECUTED}`; SHA-256 `{executed_hash}`.
- Source-only copy: `{V16_SOURCE_ONLY}`; SHA-256 `{source_only_hash}`.
- Canonical code-cell source SHA-256 for both: `{source_hash_exec}`. All code-cell sources match; execution metadata/output differs.
- The executed copy has {executed_cells}/{len(code_cells)} code cells with execution counts. Saved outputs are historical notebook state, not independent evidence of a current run.
- Both copies use data discovery by filenames/row-count/schema, not pinned hashes. The local raw datasets were separately verified by this audit.

## What V16 implements

The full feature and model inventory is in `V16_FEATURE_EXTRACTION.csv` and `V16_MODEL_CONFIGURATION.csv`. In brief: rowwise missingness, external-score moments/interactions, financial and age ratios, bureau summaries, pooled category/value frequencies, pooled group deviations, K_DEC/Q/P/community key hypotheses, heuristic repeat-applicant matching, nested target encodings, learned residual/view features, four LightGBM sets, a CatBoost product model, and a fixed rank-space blend.

`TARGET` is removed before the shared train/test predictor frame is created. Target labels are intentionally used later for supervised encodings and model fitting. The V16 target encodings are nested/cross-fitted in the source; the label-free frequency/group/fee/residual transformations are nevertheless computed over pooled train+test covariates and not isolated within each outer validation fold.

## Saved output claims

{v16_saved_claims}

The notebook output is not tied to input hashes, cache hashes, or a cache manifest. Therefore its displayed AUC is not reproduced here.

## V16 cache status

The configured cache folders and V15 fallback caches were absent from the audited local paths. V16 loads model and CatBoost OOF/test cache files by filename without verifying their hashes, data identity, feature schema, fold assignment, or model configuration. This prevents a faithful replay from those notebook cells as currently supplied.
""",
        )

        write_text(
            "CHAMPION_SOURCE_AUDIT.md",
            f"""# EXP85 champion source audit

## Artifact identity

- Source runner: `{CHAMPION_SOURCE}`; SHA-256 `{sha256(CHAMPION_SOURCE)}`.
- Frozen submission: `{CHAMPION_OUTPUT}`; SHA-256 `{champion_hash}`.
- Historical reference: `{CHAMPION_REFERENCE}`; SHA-256 `{reference_hash}`.
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
""",
        )

        write_text(
            "V16_VALIDATION_AUDIT.md",
            f"""# V16 validation and leakage audit

## Positive controls

- The source removes `TARGET` before building the common feature frame.
- Outer model OOF uses `StratifiedKFold(5, shuffle=True, random_state=42)`.
- Training encodings use nested `StratifiedKFold(5, shuffle=True, random_state=7)` inside the outer training rows. Outer validation target labels are not used in `enc(a,b)`.
- The partner-label encoding for CatBoost is nested within the outer training set; validation and test encodings use only the outer training labels.
- V16 uses fixed training iteration counts, not outer-validation early stopping.
- The optional regime-aware blending cell cross-fits its blend weights across two halves rather than fitting and scoring weights on the same half.

## Validation limitations

1. `al` concatenates train and test before category coding, value counts, group means, product frequency selection, entity fingerprints, and residual fitting. Outer validation covariates therefore contribute to some V16 features. These are transductive transforms, not fold-isolated OOF feature construction.
2. The residual learner uses KFold over the concatenated predictor rows, not the outer target folds; no outside labels are used, but the outer-validation covariates affect its fitted transforms.
3. V16 cache reads accept matching filenames without validating hashes, shapes, fold maps, feature order, or configuration. The configured caches were not present in the audited local locations.
4. The displayed `0.77649` OOF result and feature ablations are saved notebook output, not independently reproduced output. The code/source hash matches the unexecuted copy, but no saved OOF arrays or cache manifest tie results to the exact code/data.
5. Repeated historical inspection of these same labels and OOF scores increases selection exposure. Any future result should be labelled exploratory unless confirmed on a genuinely untouched evaluation set.

## Competition rules

The local official-rules audit says no external data, no hand-labeling, and ROC-AUC evaluation. It does not settle pooled unlabeled train/test fitting or target encoding. Thus V16's supervised encodings have good fold isolation in their implementation, but rule permission remains unresolved; pooled covariate transforms also require organizer clarification. The EXP85 test-to-test score coupling has additional high-risk ambiguity.

## Decision

V16 does not get a champion-improvement claim. The score and saved ablations remain unverified, the cache provenance is unavailable, and the exact EXP85 final OOF comparator is missing. No model was trained and no submission or champion file was modified.
""",
        )

        write_text(
            "V16_REPRODUCTION_PROTOCOL.md",
            f"""# V16 reproduction protocol and gate

## Current status

The notebook source is recovered, but an exact replay is not currently possible. The executed notebook and source-only copy have matching code-cell source hashes, while the executed copy lacks a hashed input/cache manifest. Its configured cache files are not present in the audited local paths.

The EXP85 comparison is separately blocked because its exact final 85/15 Anchor74+suite OOF vector and fold map are absent. Do not substitute the Phase 2/3C baseline or a single upstream component.

## Required inputs before training

1. Recover `resid16.pkl`, `e1_views.pkl`, `e1A_views.pkl`, `old_*.npz`, `e1_*.npz`, `e1et_*.npz`, `e1Aet_*.npz`, and `catboost_v16_fold_*.npz`, with hashes and source/data/schema/fold provenance.
2. Recover the exact final EXP85 OOF vector and row-to-fold assignments, including a fold-safe Anchor74 equivalent for the test-only transductive anchor.
3. Pin train/test/sample hashes, dependency versions, notebook source hash, seeds, fold assignments, feature schemas, and cache hashes before running.
4. Confirm organizer permission for pooled train/test covariate transforms, target encodings, and test-to-test entity coupling.
5. Define common folds and compare candidate and baseline on the same held-out rows. Keep model parameters, preprocessing and transformations fold-local where required; do not tune on the final comparison labels.

## Static-audit command

```powershell
python -B "{Path(__file__).resolve()}"
```

No model-training command is provided until the missing comparator and provenance inputs are restored.
""",
        )

        write_text(
            "FINAL_AUDIT_REPORT.md",
            f"""# Phase 5 status: source recovery and champion-comparator gate

## Result

Static inspection is complete for the executed V16 notebook, its source-only counterpart, the EXP85 runner, the frozen champion submission, the local data, and the available component caches. The pre-run gate **blocked model training** because the exact final EXP85 OOF vector and its fold assignments are missing. This is not a partial AUC result: no candidate model ran, and no AUC improvement is claimed.

## Verified evidence

- Raw inputs: `train.csv` 171,202 x 34 and `test.csv` 61,500 x 33; hashes match the recorded local inputs.
- V16 executed and source-only notebook copies have identical canonical code-cell source hash `{source_hash_exec}`; only execution state/output differs. The executed file has {executed_cells}/{len(code_cells)} code cells with execution counts.
- EXP85 frozen submission and historical reference are byte-identical: SHA-256 `{champion_hash}`.
- V17/V55 upstream model bundles and multiple OOF component arrays exist, but the final test synthesis requires a test-only Anchor74 plus an eight-engine suite. Component OOF arrays do not constitute an exact final champion OOF.
- The V16 `0.77649` and other values, if displayed in saved outputs, are recorded in `V16_SAVED_OUTPUT_EVIDENCE.csv` as claims, not reproduced results.
- Saved V16 ablation output reports `0.76125` baseline, `0.77185` with product keys, `0.77206` with community keys, and `0.77307` with community plus partner features; all are unverified saved-output claims. The optional regime blend is reported as `0.77648` versus `0.77649` fixed, so it does not improve the saved score.
- The saved adversarial validation output is `0.6856`; if reproduced, this points to train/test covariate separability, not a hidden target pattern. It is not reproduced here.

## Why no AUC comparison was run

{BLOCKED_REASON}

The official competition log's verified `0.80041` submission hash differs from the frozen EXP85 file hash. The runner's `0.800850` comment and the user-provided V16 `0.78222` screenshot/name are not independently validated score-to-artifact links.

## Next required inputs

1. Exact final EXP85 champion OOF predictions from the same 85/15 synthesis, with the OOF fold assignment and member-level lineage.
2. The exact Anchor74 OOF representation corresponding to the frozen test anchor, including how its test-to-test coupling is represented out of fold.
3. V16 cache files (`resid16.pkl`, view caches, model-member NPZs, CatBoost fold NPZs) with a provenance manifest binding them to raw data hashes, source, feature schema, and folds.
4. Written organizer clarification for pooled test-covariate fitting and test-to-test score coupling before any competition submission using those methods.

## Deliverables

See `SOURCE_CODE_AUDIT.md`, `CHAMPION_SOURCE_AUDIT.md`, `V16_VALIDATION_AUDIT.md`, `V16_REPRODUCTION_PROTOCOL.md`, `V16_FEATURE_EXTRACTION.csv`, `V16_MODEL_CONFIGURATION.csv`, `V16_SAVED_OUTPUT_SUMMARY.csv`, `FEATURE_ALIAS_AUDIT.csv`, `V16_RULES_COMPLIANCE.csv`, `MISSING_ARTIFACTS.csv`, `FOLD_LEVEL_CV_RESULTS.csv`, and `BASELINE_VS_FEATURES_RESULTS.csv`. Blocked metric fields are intentionally empty rather than fabricated.
""",
        )

        write_text(
            "ENVIRONMENT.txt",
            f"""Audit runtime: Python {platform.python_version()} ({platform.python_implementation()})
Operating system: {platform.platform()}
Dependencies: Python standard library only
Modeling libraries: not imported
Model training: not run
""",
        )
        write_text("requirements.txt", "# Standard-library-only static audit; no pip dependencies.\n")

        required = [
            "SOURCE_CODE_AUDIT.md",
            "CHAMPION_SOURCE_AUDIT.md",
            "V16_VALIDATION_AUDIT.md",
            "V16_REPRODUCTION_PROTOCOL.md",
            "V16_FEATURE_EXTRACTION.csv",
            "V16_MODEL_CONFIGURATION.csv",
            "V16_SAVED_OUTPUT_EVIDENCE.csv",
            "V16_SAVED_OUTPUT_SUMMARY.csv",
            "FEATURE_ALIAS_AUDIT.csv",
            "MISSING_ARTIFACTS.csv",
            "V16_RULES_COMPLIANCE.csv",
            "FOLD_LEVEL_CV_RESULTS.csv",
            "BASELINE_VS_FEATURES_RESULTS.csv",
            "FINAL_AUDIT_REPORT.md",
            "ENVIRONMENT.txt",
            "requirements.txt",
        ]
        for name in required:
            path = OUT / name
            assert path.is_file() and path.stat().st_size > 0, f"Expected output missing/empty: {path}"
        print("Output files created and non-empty:")
        for name in required:
            print(f"  {OUT / name}")
        print("STATUS: STATIC AUDIT PASS; MODEL EXPERIMENT BLOCKED; NO AUC CLAIM.")

    log = stdout.getvalue()
    print(log, end="")
    log_path = OUT / "EXECUTION_LOG.txt"
    log_path.write_text(log + f"\nExecution status: PASS (static audit); timestamp {datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")
    assert log_path.is_file() and log_path.stat().st_size > 0
    return log


if __name__ == "__main__":
    run()
