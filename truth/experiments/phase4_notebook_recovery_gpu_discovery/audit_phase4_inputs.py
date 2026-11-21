from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook


DATA_DIR = Path(r"D:\gci\inpuy")
OFFICIAL_INPUT_DIR = Path(
    r"D:\gci\rules and info\04. Competition (due Nov 20)\Competition\input"
)
DICTIONARY = Path(
    r"D:\gci\rules and info\04. Competition (due Nov 20)\Competition"
    r"\HomeCredit_columns_description.xlsx"
)
OUT = Path(r"D:\gci\truth\experiments\phase4_notebook_recovery_gpu_discovery")
FILES = ("train.csv", "test.csv", "sample_submission.csv")
FIELDS = (
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    hash_rows = []
    for root_name, root in (("project", DATA_DIR), ("official_package", OFFICIAL_INPUT_DIR)):
        for filename in FILES:
            path = root / filename
            if not path.is_file():
                raise FileNotFoundError(f"Required input is missing: {path}")
            hash_rows.append(
                {
                    "copy": root_name,
                    "file": filename,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "path": str(path),
                }
            )

    hashes = pd.DataFrame(hash_rows)
    for filename in FILES:
        values = hashes.loc[hashes["file"] == filename, "sha256"].unique()
        if len(values) != 1:
            raise ValueError(f"Project and official package copies differ for {filename}")

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    if "TARGET" not in train or "TARGET" in test:
        raise ValueError("TARGET must exist only in train.csv.")
    if train["TARGET"].isna().any() or not set(train["TARGET"].unique()).issubset({0, 1}):
        raise ValueError("Training TARGET must contain non-missing binary labels only.")
    train_features = [column for column in train.columns if column != "TARGET"]
    if train_features != list(test.columns):
        raise ValueError("Train predictors and test columns do not match in name/order.")
    if train["SK_ID_CURR"].duplicated().any() or test["SK_ID_CURR"].duplicated().any():
        raise ValueError("SK_ID_CURR must be unique within train and test.")
    if list(sample.columns) != ["SK_ID_CURR", "TARGET"]:
        raise ValueError("Unexpected sample_submission.csv schema.")
    if not sample["SK_ID_CURR"].equals(test["SK_ID_CURR"]):
        raise ValueError("Sample submission IDs/order differ from test.csv.")

    workbook = load_workbook(DICTIONARY, read_only=True, data_only=True)
    if "Sheet1" not in workbook.sheetnames:
        raise ValueError("Expected dictionary worksheet 'Sheet1' was not found.")
    descriptions = {}
    for row in workbook["Sheet1"].iter_rows(min_row=2, values_only=True):
        if len(row) >= 2 and row[0]:
            descriptions[str(row[0])] = str(row[1])
    required = set(FIELDS) | {
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    }
    missing = sorted(required - descriptions.keys())
    if missing:
        raise ValueError(f"Dictionary fields are missing: {missing}")

    dict_rows = [
        f"- `{field}`: {descriptions[field]}"
        for field in sorted(required)
    ]
    bureau_rows = [
        f"- `{field}`: {descriptions[field]}"
        for field in sorted(descriptions)
        if field.startswith("AMT_REQ_CREDIT_BUREAU_")
    ]
    train_rate = float(train["TARGET"].mean())
    report = f"""# Phase 4 input and dictionary audit

## Execution record

- Timestamp (UTC): {timestamp}
- Command: `python D:\\gci\\truth\\experiments\\phase4_notebook_recovery_gpu_discovery\\audit_phase4_inputs.py`
- Source SHA-256: `{sha256(Path(__file__))}`
- No model or feature experiment was run.

## Dataset validation

- `train.csv`: {train.shape[0]} rows x {train.shape[1]} columns; positive rate {train_rate:.10f}.
- `test.csv`: {test.shape[0]} rows x {test.shape[1]} columns; TARGET is absent.
- `sample_submission.csv`: {sample.shape[0]} rows x {sample.shape[1]} columns; IDs/order exactly match test.
- Train IDs and test IDs are unique within each file; predictor column names/order match after excluding train TARGET.
- Project CSV copies and the competition-package copies have matching SHA-256 for all three files. Full paths, byte counts, and hashes are in `INPUT_HASH_MANIFEST.csv`.
- Input rows are read only to validate schema, IDs, target integrity, and dictionary semantics. No labels were generated or edited.

## Dictionary definitions used in formula triage

{chr(10).join(dict_rows)}

### Bureau inquiry intervals (verbatim local dictionary text)

{chr(10).join(bureau_rows)}

The local dictionary does not state a payment frequency for `AMT_ANNUITY`. Its `AMT_GOODS_PRICE` definition is scoped to consumer loans; do not interpret credit/goods as universal collateral LTV without checking contract coverage. The YEAR inquiry description literally says “one day year”; this wording is ambiguous and is not silently corrected here. Exact interval lengths must be confirmed before constructing normalized acceleration rates.

## Result

All structural checks passed. This establishes that the local modeling CSVs match the supplied competition-package copies; it does not resolve organizer permission for transductive unlabeled transforms or certify any model score.
"""
    (OUT / "INPUT_HASH_MANIFEST.csv").write_text(
        hashes.to_csv(index=False), encoding="utf-8"
    )
    (OUT / "DATA_DICTIONARY_AND_INPUT_AUDIT.md").write_text(report, encoding="utf-8")
    log = (
        f"timestamp_utc={timestamp}\n"
        f"command=python {Path(__file__)}\n"
        f"status=PASS\n"
        f"train_shape={train.shape}\n"
        f"test_shape={test.shape}\n"
        f"sample_shape={sample.shape}\n"
        f"target_mean={train_rate:.10f}\n"
        "model_training=NOT_RUN\n"
        f"outputs={OUT / 'INPUT_HASH_MANIFEST.csv'}; "
        f"{OUT / 'DATA_DICTIONARY_AND_INPUT_AUDIT.md'}; "
        f"{OUT / 'INPUT_AUDIT_LOG.txt'}\n"
    )
    (OUT / "INPUT_AUDIT_LOG.txt").write_text(log, encoding="utf-8")
    print(log, end="")


if __name__ == "__main__":
    main()
