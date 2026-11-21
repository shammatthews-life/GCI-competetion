from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path


SOURCE_NOTEBOOK = Path(
    r"D:\gci\truth\experiments\exp02_financial_behaviour"
    r"\EXP02_FINANCIAL_BEHAVIOUR.ipynb"
)
EXPERIMENT_DIR = Path(
    r"D:\gci\truth\experiments\exp02_financial_behaviour_replication_seed20261105"
)
REPLICA_NOTEBOOK = EXPERIMENT_DIR / "EXP02_FINANCIAL_BEHAVIOUR_SEED20261105.ipynb"
PROTOCOL = EXPERIMENT_DIR / "EXPERIMENT_PROTOCOL.md"
RUN_LOG = EXPERIMENT_DIR / "EXECUTION_LOG.txt"
CLONE_MANIFEST = EXPERIMENT_DIR / "REPLICATION_CLONE_MANIFEST.json"
SEEDS = {
    "outer_seed": 20261105,
    "inner_stop_seed_base": 20262105,
    "model_seed_base": 20263105,
}


def sha256_file(path: Path) -> str:
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


def save_notebook(document: dict) -> None:
    REPLICA_NOTEBOOK.write_text(
        json.dumps(document, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Expected exactly one occurrence of {old!r}.")
    return source.replace(old, new, 1)


class Tee:
    def __init__(self, *targets):
        self.targets = targets

    def write(self, text: str) -> int:
        for target in self.targets:
            target.write(text)
        return len(text)

    def flush(self) -> None:
        for target in self.targets:
            target.flush()


def prepare_replica() -> dict:
    if not SOURCE_NOTEBOOK.is_file():
        raise FileNotFoundError(f"Source notebook missing: {SOURCE_NOTEBOOK}")
    if not PROTOCOL.is_file():
        raise FileNotFoundError(f"Predeclared protocol missing: {PROTOCOL}")
    for path in (REPLICA_NOTEBOOK, RUN_LOG, CLONE_MANIFEST):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite replication artifact: {path}")

    document = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    setup_cells = [
        cell
        for cell in document.get("cells", [])
        if cell.get("cell_type") == "code"
        and "RUN_ID = \"financial_ratios_seed42\"" in "".join(cell.get("source", []))
    ]
    if len(setup_cells) != 1:
        raise ValueError("Could not uniquely locate the frozen EXP02 setup cell.")

    setup = setup_cells[0]
    source = "".join(setup.get("source", []))
    replacements = (
        (
            'EXPERIMENT_DIR = TRUTH_ROOT / "experiments" / "exp02_financial_behaviour"',
            'EXPERIMENT_DIR = TRUTH_ROOT / "experiments" / "exp02_financial_behaviour_replication_seed20261105"',
        ),
        (
            'NOTEBOOK_PATH = EXPERIMENT_DIR / "EXP02_FINANCIAL_BEHAVIOUR.ipynb"',
            'NOTEBOOK_PATH = EXPERIMENT_DIR / "EXP02_FINANCIAL_BEHAVIOUR_SEED20261105.ipynb"',
        ),
        ('RUN_ID = "financial_ratios_seed42"', 'RUN_ID = "financial_ratios_seed20261105"'),
        ("OUTER_SEED = 42", "OUTER_SEED = 20261105"),
        ("INNER_STOP_SEED_BASE = 20261102", "INNER_STOP_SEED_BASE = 20262105"),
        ("MODEL_SEED_BASE = 20261103", "MODEL_SEED_BASE = 20263105"),
    )
    for old, new in replacements:
        source = replace_once(source, old, new)
    setup["source"] = source.splitlines(keepends=True)

    for cell in document.get("cells", []):
        if cell.get("cell_type") == "code":
            code = "".join(cell.get("source", []))
            if any(line.lstrip().startswith(("%", "!")) for line in code.splitlines()):
                raise ValueError("Notebook contains a Jupyter-only magic/shell cell.")
            if "get_ipython(" in code:
                raise ValueError("Notebook depends on an IPython-only runtime call.")
            cell["execution_count"] = None
            cell["outputs"] = []

    save_notebook(document)
    return document


def build_clone_manifest(document: dict) -> dict:
    return {
        "status": "PREPARED",
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source_notebook": str(SOURCE_NOTEBOOK),
        "source_notebook_sha256": sha256_file(SOURCE_NOTEBOOK),
        "source_canonical_cell_sha256": canonical_source_sha256(
            json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
        ),
        "replica_notebook": str(REPLICA_NOTEBOOK),
        "replica_notebook_source_sha256": canonical_source_sha256(document),
        "replica_notebook_pre_execution_sha256": sha256_file(REPLICA_NOTEBOOK),
        "protocol_sha256": sha256_file(PROTOCOL),
        "data_directory": r"D:\gci\inpuy",
        "run_id": "financial_ratios_seed20261105",
        "seeds": SEEDS,
        "training_started": False,
        "training_completed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create and verify the isolated notebook copy without model training.",
    )
    args = parser.parse_args()
    if args.prepare_only:
        document = prepare_replica()
        manifest = build_clone_manifest(document)
        CLONE_MANIFEST.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print("status=PREPARED; training=NOT_RUN")
        print(f"replica_notebook={REPLICA_NOTEBOOK}")
        print(f"replica_source_sha256={manifest['replica_notebook_source_sha256']}")
        print(f"clone_manifest={CLONE_MANIFEST}")
        return

    if not REPLICA_NOTEBOOK.is_file() or not CLONE_MANIFEST.is_file():
        raise FileNotFoundError(
            "Run with --prepare-only and inspect the clone before model execution."
        )
    if RUN_LOG.exists():
        raise FileExistsError(f"Refusing to overwrite replication log: {RUN_LOG}")
    document = json.loads(REPLICA_NOTEBOOK.read_text(encoding="utf-8"))
    manifest = json.loads(CLONE_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("status") != "PREPARED":
        raise ValueError("Replication clone is not in PREPARED state.")
    if sha256_file(REPLICA_NOTEBOOK) != manifest["replica_notebook_pre_execution_sha256"]:
        raise ValueError("Prepared notebook changed after pre-run inspection.")
    if sha256_file(SOURCE_NOTEBOOK) != manifest["source_notebook_sha256"]:
        raise ValueError("Source notebook changed after the replication was prepared.")
    if sha256_file(PROTOCOL) != manifest["protocol_sha256"]:
        raise ValueError("Predeclared protocol changed after preparation.")

    source_hash = canonical_source_sha256(document)
    if source_hash != manifest["replica_notebook_source_sha256"]:
        raise ValueError("Prepared replica source hash mismatch.")
    started = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "RUNNING"
    manifest["run_started_utc"] = started
    CLONE_MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    namespace = {"__name__": "__main__", "__file__": str(REPLICA_NOTEBOOK)}
    original_stdout = sys.stdout
    status = "PASS"
    failure = None
    execution_error = None
    training_started = False
    training_completed = False
    with RUN_LOG.open("x", encoding="utf-8", newline="") as log:
        log.write(f"timestamp_utc={started}\n")
        log.write(f"command=python {Path(__file__)}\n")
        log.write(f"source_notebook_sha256={manifest['source_notebook_sha256']}\n")
        log.write(f"replica_notebook_source_sha256={source_hash}\n")
        log.write(f"protocol_sha256={manifest['protocol_sha256']}\n")
        log.write(f"seeds={json.dumps(SEEDS, sort_keys=True)}\n")
        log.write("execution_mode=sequential Python execution of notebook code cells\n")
        log.write("notebook_execution=STARTED\n")
        log.write("training_status=NOT_STARTED\n")
        log.flush()
        execution_count = 0
        for cell_index, cell in enumerate(document.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            if cell_index == 10:
                training_started = True
                manifest["training_started"] = True
                CLONE_MANIFEST.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )
                log.write("training_status=STARTED\n")
                log.flush()
            execution_count += 1
            source = "".join(cell.get("source", []))
            buffer = io.StringIO()
            tee = Tee(original_stdout, log, buffer)
            try:
                with redirect_stdout(tee), redirect_stderr(tee):
                    print(f"\n--- code cell {cell_index} start ---")
                    exec(
                        compile(source, f"{REPLICA_NOTEBOOK}#cell-{cell_index}", "exec"),
                        namespace,
                    )
                output = buffer.getvalue()
                cell["execution_count"] = execution_count
                cell["outputs"] = (
                    [
                        {
                            "output_type": "stream",
                            "name": "stdout",
                            "text": output.splitlines(keepends=True),
                        }
                    ]
                    if output
                    else []
                )
                save_notebook(document)
            except Exception as exc:
                status = "FAIL"
                execution_error = exc
                failure = traceback.format_exc()
                cell["execution_count"] = execution_count
                cell["outputs"] = [
                    {
                        "output_type": "error",
                        "ename": "CellExecutionError",
                        "evalue": f"Cell {cell_index} failed.",
                        "traceback": failure.splitlines(),
                    }
                ]
                save_notebook(document)
                log.write(f"\nexecution_status=FAIL\n{failure}\n")
                log.flush()
                break
            if cell_index == 10:
                training_completed = True
                manifest["training_completed"] = True
                CLONE_MANIFEST.write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )

        log.write(f"\nexecution_status={status}\n")
        log.flush()

    manifest["status"] = status
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["training_started"] = training_started
    manifest["training_completed"] = training_completed
    manifest["replica_notebook_final_sha256"] = sha256_file(REPLICA_NOTEBOOK)
    manifest["execution_log_sha256"] = sha256_file(RUN_LOG)
    if failure:
        manifest["failure"] = failure
    CLONE_MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"status={status}")
    print(f"replica_notebook={REPLICA_NOTEBOOK}")
    print(f"execution_log={RUN_LOG}")
    print(f"clone_manifest={CLONE_MANIFEST}")
    if execution_error:
        raise RuntimeError("Seed replication execution failed; see execution log.") from execution_error


if __name__ == "__main__":
    main()
