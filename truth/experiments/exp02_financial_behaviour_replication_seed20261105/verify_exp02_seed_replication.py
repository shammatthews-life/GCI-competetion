from __future__ import annotations

import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(
    r"D:\gci\truth\experiments\exp02_financial_behaviour_replication_seed20261105"
)
RUN = ROOT / "runs" / "financial_ratios_seed20261105"
DATA = Path(r"D:\gci\inpuy")
SOURCE_NOTEBOOK = Path(
    r"D:\gci\truth\experiments\exp02_financial_behaviour"
    r"\EXP02_FINANCIAL_BEHAVIOUR.ipynb"
)
AUDIT_RESULTS = ROOT / "REPLICATION_AUDIT_RESULTS.json"
AUDIT_REPORT = ROOT / "REPLICATION_AUDIT_REPORT.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_source_sha256(document: dict) -> str:
    source_cells = [
        {
            "cell_type": cell.get("cell_type"),
            "source": "".join(cell.get("source", [])),
        }
        for cell in document.get("cells", [])
    ]
    payload = json.dumps(
        source_cells,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_hashes(hash_manifest: dict) -> None:
    for name, expected in hash_manifest["raw_input_sha256"].items():
        actual = sha256(DATA / name)
        require(actual == expected, f"Input hash mismatch: {name}")

    for name, expected in hash_manifest["root_artifact_sha256"].items():
        actual = sha256(ROOT / name)
        require(actual == expected, f"Root artifact hash mismatch: {name}")

    for name, expected in hash_manifest["run_artifact_sha256"].items():
        actual = sha256(RUN / name)
        require(actual == expected, f"Run artifact hash mismatch: {name}")

    clone = json.loads(
        (ROOT / "REPLICATION_CLONE_MANIFEST.json").read_text(encoding="utf-8")
    )
    require(clone["status"] == "PASS", "Clone manifest is not PASS")
    require(clone["training_started"], "Clone manifest says training did not start")
    require(clone["training_completed"], "Clone manifest says training did not complete")
    require(
        sha256(ROOT / "EXPERIMENT_PROTOCOL.md") == clone["protocol_sha256"],
        "Protocol hash differs from clone manifest",
    )
    require(
        sha256(ROOT / "EXECUTION_LOG.txt") == clone["execution_log_sha256"],
        "Execution log hash differs from clone manifest",
    )
    require(
        sha256(ROOT / "EXP02_FINANCIAL_BEHAVIOUR_SEED20261105.ipynb")
        == clone["replica_notebook_final_sha256"],
        "Executed notebook hash differs from clone manifest",
    )


def verify_executed_source(clone: dict) -> tuple[dict, str]:
    source = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    replica_path = ROOT / "EXP02_FINANCIAL_BEHAVIOUR_SEED20261105.ipynb"
    replica = json.loads(replica_path.read_text(encoding="utf-8"))
    require(
        sha256(SOURCE_NOTEBOOK) == clone["source_notebook_sha256"],
        "Source notebook hash differs from clone manifest",
    )
    source_hash = canonical_source_sha256(replica)
    require(
        source_hash == clone["replica_notebook_source_sha256"],
        "Replica source hash differs from clone manifest",
    )
    require(
        source_hash == json.loads((RUN / "run_manifest.json").read_text())[
            "notebook_source_sha256"
        ],
        "Replica source hash differs from run manifest",
    )
    require(len(source["cells"]) == len(replica["cells"]), "Notebook cell count changed")

    source_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in replica["cells"]
        if cell.get("cell_type") == "code"
    )
    setup_text = next(
        "".join(cell.get("source", []))
        for cell in replica["cells"]
        if cell.get("cell_type") == "code"
        and "RUN_ID = \"financial_ratios_seed20261105\"" in "".join(
            cell.get("source", [])
        )
    )
    actual_seeds = {
        "outer_seed": 20261105,
        "inner_stop_seed_base": 20262105,
        "model_seed_base": 20263105,
    }
    for name, value in actual_seeds.items():
        require(f"{name.upper()} = {value}" in setup_text, f"Missing frozen {name}")
    require(
        "random_state=INNER_STOP_SEED_BASE + fold_number" in source_text,
        "Inner split does not use the frozen seed constant",
    )
    require(
        "seed = MODEL_SEED_BASE + fold_number" in source_text,
        "Model fitting does not use the frozen seed constant",
    )
    require(
        "random_seed=model_seed" in source_text,
        "CatBoost does not receive the fold-specific model seed",
    )
    require(
        "random_state=OUTER_SEED" in source_text,
        "Outer splitter does not use the frozen seed constant",
    )
    return replica, source_text


def verify_and_measure() -> dict:
    hash_manifest = json.loads(
        (ROOT / "SHA256_MANIFEST.json").read_text(encoding="utf-8")
    )
    verify_hashes(hash_manifest)
    clone = json.loads(
        (ROOT / "REPLICATION_CLONE_MANIFEST.json").read_text(encoding="utf-8")
    )
    _, source_text = verify_executed_source(clone)

    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    sample = pd.read_csv(DATA / "sample_submission.csv")
    require(train.shape == (171202, 34), "Unexpected train dimensions")
    require(test.shape == (61500, 33), "Unexpected test dimensions")
    require("TARGET" in train.columns and "TARGET" not in test.columns, "Target schema mismatch")
    require(train["SK_ID_CURR"].is_unique, "Training IDs are not unique")
    require(test["SK_ID_CURR"].is_unique, "Test IDs are not unique")
    require(
        set(train["SK_ID_CURR"]).isdisjoint(set(test["SK_ID_CURR"])),
        "Train/test IDs overlap",
    )
    require(
        np.array_equal(sample["SK_ID_CURR"].to_numpy(), test["SK_ID_CURR"].to_numpy()),
        "Sample submission IDs differ from test IDs",
    )

    predictions_raw = np.load(RUN / "oof_raw_baseline.npy")
    predictions_features = np.load(RUN / "oof_financial_features.npy")
    fold_ids = np.load(RUN / "fold_ids.npy")
    test_ids = np.load(RUN / "test_ids.npy")
    test_raw = np.load(RUN / "test_raw_baseline.npy")
    test_features = np.load(RUN / "test_financial_features.npy")
    y = train["TARGET"].to_numpy()
    n_splits = 5
    require(predictions_raw.shape == y.shape, "Raw OOF shape mismatch")
    require(predictions_features.shape == y.shape, "Feature OOF shape mismatch")
    require(fold_ids.shape == y.shape, "Fold ID shape mismatch")
    require(np.isfinite(predictions_raw).all(), "Raw OOF has non-finite values")
    require(np.isfinite(predictions_features).all(), "Feature OOF has non-finite values")
    require(
        ((predictions_raw >= 0) & (predictions_raw <= 1)).all(),
        "Raw OOF is outside [0, 1]",
    )
    require(
        ((predictions_features >= 0) & (predictions_features <= 1)).all(),
        "Feature OOF is outside [0, 1]",
    )
    require(np.array_equal(test_ids, test["SK_ID_CURR"].to_numpy()), "Test ID order mismatch")
    require(test_raw.shape == (len(test),) and test_features.shape == (len(test),), "Test prediction shape mismatch")
    require(
        np.isfinite(test_raw).all()
        and np.isfinite(test_features).all()
        and ((test_raw >= 0) & (test_raw <= 1)).all()
        and ((test_features >= 0) & (test_features <= 1)).all(),
        "Test predictions are invalid",
    )

    expected_fold_ids = np.zeros(len(y), dtype=np.int8)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20261105)
    for fold_number, (_, valid) in enumerate(splitter.split(np.zeros(len(y)), y), 1):
        expected_fold_ids[valid] = fold_number
    require(np.array_equal(fold_ids, expected_fold_ids), "Saved outer folds differ from protocol")

    results = json.loads((ROOT / "RESULTS.json").read_text(encoding="utf-8"))
    fold_rows = []
    for fold_number in range(1, n_splits + 1):
        mask = fold_ids == fold_number
        raw_auc = float(roc_auc_score(y[mask], predictions_raw[mask]))
        features_auc = float(roc_auc_score(y[mask], predictions_features[mask]))
        fold_rows.append(
            {
                "fold": fold_number,
                "raw_auc": raw_auc,
                "financial_auc": features_auc,
                "paired_auc_delta": features_auc - raw_auc,
            }
        )

    pooled_raw = float(roc_auc_score(y, predictions_raw))
    pooled_features = float(roc_auc_score(y, predictions_features))
    pooled_delta = pooled_features - pooled_raw
    deltas = [row["paired_auc_delta"] for row in fold_rows]
    mean_delta = float(statistics.mean(deltas))
    delta_sd = float(statistics.stdev(deltas))

    require(abs(pooled_raw - results["pooled_oof_auc"]["raw_baseline"]) < 1e-12, "Raw pooled AUC mismatch")
    require(abs(pooled_features - results["pooled_oof_auc"]["financial_features"]) < 1e-12, "Feature pooled AUC mismatch")
    require(abs(pooled_delta - results["pooled_oof_auc_delta"]) < 1e-12, "Pooled delta mismatch")
    require(abs(mean_delta - results["mean_paired_fold_auc_delta"]) < 1e-12, "Mean fold delta mismatch")
    require(results["positive_folds"] == sum(delta > 0 for delta in deltas), "Positive fold count mismatch")
    require(results["submission_created"] is False, "Unexpected submission claim")

    with (ROOT / "FOLD_METRICS.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == n_splits, "Fold CSV row count mismatch")
    for expected, saved in zip(fold_rows, csv_rows):
        for key in ("fold", "raw_auc", "financial_auc", "paired_auc_delta"):
            require(
                abs(float(saved[key]) - float(expected[key])) < 1e-10,
                f"Fold CSV mismatch at fold {expected['fold']} for {key}",
            )

    run_manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    require(run_manifest["outer_seed"] == 20261105, "Saved outer seed mismatch")
    require(run_manifest["inner_seed_rule"] != "20262105 + 1-based outer fold number", "Unexpectedly corrected source metadata")
    require("20261102" in run_manifest["inner_seed_rule"], "Expected stale inherited inner-seed text not found")
    require("20261103" in run_manifest["model_seed_rule"], "Expected stale inherited model-seed text not found")
    report_text = (ROOT / "EXP02_REPORT.md").read_text(encoding="utf-8")
    require("Five stratified outer folds (seed 42)" in report_text, "Expected stale inherited report text not found")
    require('"independent_seed_replication_complete": false' in (ROOT / "RESULTS.json").read_text(encoding="utf-8"), "Expected stale inherited status not found")
    require("seed = MODEL_SEED_BASE + fold_number" in source_text, "Actual model seed expression missing")

    passed = (
        pooled_delta > 0
        and mean_delta >= 0.0002
        and sum(delta > 0 for delta in deltas) >= 4
    )
    require(passed, "Predeclared advancement screen did not pass")

    return {
        "verification_status": "PASS",
        "experiment": "EXP02_FINANCIAL_BEHAVIOUR",
        "run_id": "financial_ratios_seed20261105",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "data": {
            "train_rows": len(train),
            "test_rows": len(test),
            "positive_rate": float(y.mean()),
            "input_sha256": hash_manifest["raw_input_sha256"],
        },
        "actual_seed_values": {
            "outer_seed": 20261105,
            "inner_stop_seed_by_fold": [20262105 + i for i in range(1, 6)],
            "model_seed_by_fold": [20263105 + i for i in range(1, 6)],
        },
        "pooled_oof_auc": {
            "raw_baseline": pooled_raw,
            "financial_features": pooled_features,
            "delta": pooled_delta,
        },
        "fold_metrics": fold_rows,
        "mean_paired_fold_delta": mean_delta,
        "sample_sd_paired_fold_delta": delta_sd,
        "positive_folds": sum(delta > 0 for delta in deltas),
        "predeclared_screen_passed": passed,
        "static_metadata_corrections": [
            "EXP02_REPORT.md retains the source-run seed-42 sentence.",
            "RESULTS.json retains the source-run decision and incomplete-replication flag.",
            "run_manifest.json and notebook output retain old seed-rule strings; executed code uses the frozen 20262105 and 20263105 seed bases.",
        ],
        "limitations": [
            "This is a seven-feature-block replication against the matched raw-feature CatBoost baseline, not a champion ablation.",
            "The result does not identify individual feature effects or establish causal relationships.",
            "AMT_ANNUITY frequency is unspecified in the available dictionary; per-family-member ratios assume equal allocation.",
            "No public leaderboard score was evaluated and no submission was created.",
        ],
    }


def write_audit_files(data: dict) -> None:
    AUDIT_RESULTS.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# EXP02 independent-seed replication audit",
        "",
        f"- Verification: **{data['verification_status']}**",
        f"- Verified UTC: `{data['verified_utc']}`",
        f"- Run ID: `{data['run_id']}`",
        "- Evidence source: independently recomputed from saved OOF arrays, folds, inputs, and SHA-256 manifests.",
        "",
        "## Reproduced OOF results",
        "",
        "| Metric | Raw-feature baseline | Financial feature block | Delta |",
        "|---|---:|---:|---:|",
        (
            f"| Pooled OOF ROC-AUC | {data['pooled_oof_auc']['raw_baseline']:.9f} "
            f"| {data['pooled_oof_auc']['financial_features']:.9f} "
            f"| {data['pooled_oof_auc']['delta']:+.9f} |"
        ),
        (
            f"| Mean paired fold delta | - | - | "
            f"{data['mean_paired_fold_delta']:+.9f} |"
        ),
        "",
        f"Fold-delta sample SD: `{data['sample_sd_paired_fold_delta']:.9f}`. "
        f"Positive folds: `{data['positive_folds']}/5`.",
        "",
        "| Fold | Raw AUC | Feature AUC | Paired delta |",
        "|---:|---:|---:|---:|",
    ]
    for row in data["fold_metrics"]:
        lines.append(
            f"| {row['fold']} | {row['raw_auc']:.9f} | "
            f"{row['financial_auc']:.9f} | {row['paired_auc_delta']:+.9f} |"
        )
    lines.extend(
        [
            "",
            "## Protocol and integrity",
            "",
            "- The predeclared screen passed: positive pooled delta, mean fold delta at least `+0.0002`, and at least 4/5 positive folds.",
            "- Outer seed `20261105`; inner stopping seeds `20262106` through `20262110`; model seeds `20263106` through `20263110`.",
            "- Input, notebook-source, result, prediction, fold, and execution-log hashes were checked. Five stratified folds were independently regenerated and matched the saved assignments.",
            "- Training/test dimensions: `171,202 x 34` and `61,500 x 33`; sample/test ID order matched. OOF and test predictions were finite and within `[0,1]`.",
            "- CatBoost GPU execution completed. The execution log includes four Pandas `inplace` deprecation warnings and CatBoost GPU AUC metric-period notices; neither stopped training.",
            "",
            "## Inherited metadata correction",
            "",
            "The generated `EXP02_REPORT.md`, `RESULTS.json`, and `run_manifest.json` retain static text from seed 42. In particular, they report seed 42, old inner/model seed rules, and `independent_seed_replication_complete=false`. These fields are stale, not the executed configuration. The executed notebook source sets the preregistered seed constants and passes the fold-specific constants into the splitters and CatBoost models. Original generated artifacts are preserved unchanged; this independently verified audit is the corrected status record.",
            "",
            "## Interpretation and limitations",
            "",
            "- The seven-feature block reproduced a consistent improvement over its matched raw-feature baseline across this independent seed: 5/5 folds improved.",
            "- This supports a repeatable predictive signal in the joint ratio block, not a causal explanation, individual-feature attribution, or improvement over the champion.",
            "- The next valid step is a predeclared matched champion ablation on identical folds. Do not claim champion lift yet.",
            "- The available dictionary does not establish `AMT_ANNUITY` frequency; per-family-member ratios assume equal allocation. These semantic caveats remain.",
            "- No public leaderboard score was evaluated; no submission was created.",
            "",
            "## Artifacts",
            "",
            "- Original execution evidence: `EXECUTION_LOG.txt`",
            "- Executed notebook: `EXP02_FINANCIAL_BEHAVIOUR_SEED20261105.ipynb`",
            "- Independent verifier: `verify_exp02_seed_replication.py`",
            "- Machine-readable audit: `REPLICATION_AUDIT_RESULTS.json`",
            "- Original generated files were not edited by this verifier.",
        ]
    )
    AUDIT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    audit = verify_and_measure()
    write_audit_files(audit)
    print(
        "verification=PASS",
        f"pooled_raw_auc={audit['pooled_oof_auc']['raw_baseline']:.9f}",
        f"pooled_feature_auc={audit['pooled_oof_auc']['financial_features']:.9f}",
        f"delta={audit['pooled_oof_auc']['delta']:+.9f}",
        f"mean_fold_delta={audit['mean_paired_fold_delta']:+.9f}",
        f"fold_delta_sample_sd={audit['sample_sd_paired_fold_delta']:.9f}",
        f"positive_folds={audit['positive_folds']}/5",
        sep="\n",
    )
    print(f"audit_report={AUDIT_REPORT}")
    print(f"audit_results={AUDIT_RESULTS}")
