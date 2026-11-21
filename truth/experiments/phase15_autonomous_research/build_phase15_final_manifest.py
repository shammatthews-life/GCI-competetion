from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
TRAIN = Path(r"D:\gci\inpuy\train.csv")
FILES = (
    "PHASE15_EXECUTION_REPORT.md",
    "PHASE15_RESEARCH_MAP.md",
    "PHASE15_ARTIFACT_REGISTRY.csv",
    "PHASE15_INDEPENDENT_AUC_VERIFICATION.csv",
    "PHASE15_MANIFEST_CORRECTIONS.csv",
    "PHASE15_VERIFICATION_SUMMARY.json",
    "PHASE15_POSTHOC_DIAGNOSTICS.json",
    "PHASE15_FIXED_BLEND_DIAGNOSTICS.csv",
    "PHASE15_OOF_CORRELATIONS.csv",
    "PHASE15_ID_ORDER_DIAGNOSTICS.csv",
    "PHASE15_TIME_FEATURE_PAIRED_BOOTSTRAP.json",
    "phase15_features.py",
    "phase15_interactions.py",
    "generate_phase15_diagnostics.py",
    "build_phase15_final_manifest.py",
    "test_phase15_features.py",
    "verify_phase15_artifacts.py",
    "run_phase15_catboost_preflight.py",
    "run_phase15_catboost_oof.py",
    "run_phase15_catboost_multiseed.py",
    "run_phase15_xgboost_category_preflight.py",
    "run_phase15_ctr2_preflight.py",
    "run_phase15_org_ext_cross_preflight.py",
    "run_phase15_org_ext_cross_oof.py",
)
RUN_FILES = (
    (
        "runs/phase15_native_catboost_oof_20261105T124238Z/"
        "OOF_PREDICTIONS.csv"
    ),
    (
        "runs/phase15_native_catboost_multiseed_20261105T125826Z/"
        "MULTISEED_OOF_PREDICTIONS.csv"
    ),
    (
        "runs/phase15_org_ext_cross_oof_20261105T130802Z/"
        "OOF_PREDICTIONS.csv"
    ),
    (
        "runs/phase15_native_catboost_oof_20261105T124238Z/"
        "SHA256_MANIFEST.csv"
    ),
    (
        "runs/phase15_native_catboost_multiseed_20261105T125826Z/"
        "SHA256_MANIFEST.csv"
    ),
)
HISTORICAL_INPUTS = (
    Path(
        r"D:\gci\truth\experiments\phase9_product_mechanism_and_catboost"
        r"\runs\phase9_20261104T181500LOCAL\OOF_PREDICTIONS.csv"
    ),
    Path(
        r"D:\gci\truth\experiments\phase13_matched_baseline"
        r"\runs\phase13_20261105T075946Z\OOF_PREDICTIONS.csv"
    ),
    Path(
        r"D:\gci\truth\experiments\exp85_phase2_inner_early_stopping"
        r"\runs\phase2_inner_stop_seed42\oof_phase2_blend.npy"
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main() -> None:
    paths = [ROOT / name for name in FILES]
    paths.extend(ROOT / Path(name) for name in RUN_FILES)
    paths.extend((TRAIN, *HISTORICAL_INPUTS))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required manifest inputs are missing: " + ", ".join(missing))

    manifest = ROOT / "PHASE15_FINAL_SHA256_MANIFEST.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("file", "bytes", "sha256"), lineterminator="\n"
        )
        writer.writeheader()
        for path in paths:
            writer.writerow(
                {
                    "file": display_path(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )


if __name__ == "__main__":
    main()
