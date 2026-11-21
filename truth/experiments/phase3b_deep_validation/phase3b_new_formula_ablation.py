from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(r"D:\gci\truth")
DATA = Path(r"D:\gci\inpuy")
PHASE3 = ROOT / "experiments" / "phase3_hidden_pattern_discovery"
BASELINE_DIR = ROOT / "experiments" / "phase2_advanced_feature_engineering"
OUT = ROOT / "experiments" / "phase3b_deep_validation"
BASELINE_SOURCE = (
    ROOT
    / "research_archive"
    / "exp85_phase2_source_snapshot"
    / "project"
    / "source"
    / "run_independent_baseline.py"
)
ABLATION_SOURCE = OUT / "phase3b_ablation.py"
ID = "SK_ID_CURR"
TARGET = "TARGET"
SELECTED = [
    "NEW_CREDIT_GOODS_GAP_TO_INCOME",
    "NEW_GOODS_TO_INCOME",
    "NEW_ANNUITY_TO_GOODS",
    "NEW_CREDIT_GOODS_LOG_RATIO",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.time()
    started_utc = datetime.now(timezone.utc)
    baseline = load_module("phase3b_formula_baseline", BASELINE_SOURCE)
    ablation = load_module("phase3b_formula_builder", ABLATION_SOURCE)
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].to_numpy(dtype=np.int8)
    ids = train[ID].to_numpy()
    raw = train.drop(columns=[TARGET]).reset_index(drop=True)
    folds = pd.read_csv(BASELINE_DIR / "FOLD_ASSIGNMENTS.csv").set_index(ID).loc[ids, "fold"].to_numpy()
    rounds_by_fold = {
        int(k): int(v)
        for k, v in json.loads((OUT / "FOLD_SPECIFIC_BOOSTING_ROUNDS.json").read_text(encoding="utf-8")).items()
    }
    main_oof = pd.read_csv(OUT / "PHASE3B_ABLATION_OOF.csv").set_index(ID).loc[ids]
    baseline_refit = main_oof["OOF_BASELINE_REFIT_FIXED_ROUNDS"].to_numpy(dtype=np.float64)
    saved_baseline = main_oof["OOF_BASELINE_REFERENCE"].to_numpy(dtype=np.float64)
    formula_frame = ablation.new_formula_features(raw)

    config = baseline.MODEL_CONFIG["lightgbm"]
    model_params = {
        key: value
        for key, value in config.items()
        if key not in ("device", "num_boost_round", "early_stopping_rounds")
    }
    predictions = {name: np.full(len(train), np.nan, dtype=np.float64) for name in SELECTED}
    fold_rows = []
    for fold in sorted(np.unique(folds)):
        fit_idx, valid_idx = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        fit_base = baseline.build_rowwise_features(raw.iloc[fit_idx])
        valid_base = baseline.build_rowwise_features(raw.iloc[valid_idx])
        rounds = rounds_by_fold[int(fold)]
        for name in SELECTED:
            fit_extra = formula_frame.iloc[fit_idx][[name]].reset_index(drop=True)
            valid_extra = formula_frame.iloc[valid_idx][[name]].reset_index(drop=True)
            fit_matrix, valid_matrix = baseline.make_lightgbm_frames(
                pd.concat([fit_base, fit_extra], axis=1),
                pd.concat([valid_base, valid_extra], axis=1),
            )
            model = lgb.train(
                {**model_params, "seed": baseline.FOLD_SEED + int(fold)},
                lgb.Dataset(fit_matrix, label=y[fit_idx]),
                num_boost_round=rounds,
                callbacks=[lgb.log_evaluation(0)],
            )
            pred = model.predict(valid_matrix, num_iteration=rounds)
            if not np.isfinite(pred).all():
                raise ValueError(f"Non-finite predictions for {name}, fold {fold}")
            predictions[name][valid_idx] = pred
            auc = roc_auc_score(y[valid_idx], pred)
            baseline_auc = roc_auc_score(y[valid_idx], baseline_refit[valid_idx])
            fold_rows.append(
                {
                    "fold": int(fold),
                    "feature_name": name,
                    "boosting_rounds_from_inner_raw_baseline_only": rounds,
                    "fold_auc": auc,
                    "refit_baseline_fold_auc": baseline_auc,
                    "fold_delta_vs_refit_baseline": auc - baseline_auc,
                    "saved_baseline_fold_auc": roc_auc_score(y[valid_idx], saved_baseline[valid_idx]),
                    "outer_validation_used_for_early_stopping": False,
                }
            )
            print(f"fold={fold} feature={name} auc={auc:.6f} delta={auc-baseline_auc:+.6f}", flush=True)

    folds_frame = pd.DataFrame(fold_rows)
    folds_frame.to_csv(OUT / "NEW_FORMULA_INDIVIDUAL_FOLD_RESULTS.csv", index=False)
    summary = []
    for name in SELECTED:
        pred = predictions[name]
        rows = folds_frame.loc[folds_frame.feature_name == name].sort_values("fold")
        deltas = rows["fold_delta_vs_refit_baseline"].to_numpy()
        summary.append(
            {
                "feature_name": name,
                "pooled_oof_auc": float(roc_auc_score(y, pred)),
                "pooled_delta_vs_refit_baseline": float(roc_auc_score(y, pred) - roc_auc_score(y, baseline_refit)),
                "pooled_delta_vs_saved_baseline": float(roc_auc_score(y, pred) - roc_auc_score(y, saved_baseline)),
                "mean_fold_auc": float(rows.fold_auc.mean()),
                "std_fold_auc": float(rows.fold_auc.std(ddof=1)),
                "mean_paired_fold_delta": float(deltas.mean()),
                "std_paired_fold_delta": float(deltas.std(ddof=1)),
                "folds_improved": int((deltas > 0).sum()),
                "fold_deltas": json.dumps({int(f): float(d) for f, d in zip(rows.fold, deltas)}, sort_keys=True),
                "validation_status": "OOF discovery on reused folds; individual feature ablation",
            }
        )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUT / "NEW_FORMULA_INDIVIDUAL_ABLATIONS.csv", index=False)

    prediction_frame = pd.DataFrame(
        {
            ID: ids,
            TARGET: y,
            "fold": folds,
            "OOF_BASELINE_REFIT_FIXED_ROUNDS": baseline_refit,
            **{f"OOF_{name}": pred for name, pred in predictions.items()},
        }
    )
    prediction_frame.to_csv(OUT / "NEW_FORMULA_INDIVIDUAL_OOF.csv", index=False)

    validation_path = OUT / "NEW_FORMULA_VALIDATION.csv"
    validation = pd.read_csv(validation_path)
    metric_map = summary_frame.set_index("feature_name").to_dict(orient="index")
    for column, default in [
        ("individual_incremental_pooled_auc", np.nan),
        ("individual_delta_vs_refit_baseline", np.nan),
        ("individual_folds_improved", np.nan),
        ("individual_fold_deltas", ""),
    ]:
        if column not in validation:
            validation[column] = default
    validation["incremental_cv_status"] = "NOT_INDIVIDUALLY_MODELLED; see formula catalog decision"
    for idx, row in validation.iterrows():
        name = row["feature_name"]
        if name in metric_map:
            metric = metric_map[name]
            validation.loc[idx, "individual_incremental_pooled_auc"] = metric["pooled_oof_auc"]
            validation.loc[idx, "individual_delta_vs_refit_baseline"] = metric["pooled_delta_vs_refit_baseline"]
            validation.loc[idx, "individual_folds_improved"] = metric["folds_improved"]
            validation.loc[idx, "individual_fold_deltas"] = metric["fold_deltas"]
            validation.loc[idx, "incremental_cv_status"] = "INDIVIDUAL_FIXED_ROUND_OOF; folds previously exposed"
    validation.to_csv(validation_path, index=False)

    finished = datetime.now(timezone.utc)
    log = [
        f"started_utc={started_utc.isoformat()}",
        f"finished_utc={finished.isoformat()}",
        f"runtime_seconds={time.time()-started:.3f}",
        f"source_sha256={sha256(Path(__file__).resolve())}",
        f"baseline_source_sha256={sha256(BASELINE_SOURCE)}",
        f"ablation_builder_sha256={sha256(ABLATION_SOURCE)}",
        f"train_sha256={sha256(DATA/'train.csv')}",
        f"individual_formula_count={len(SELECTED)}",
        f"outer_folds={len(np.unique(folds))}",
        "status=SUCCESS",
        summary_frame.to_string(index=False),
    ]
    with (OUT / "EXECUTION_LOG.txt").open("a", encoding="utf-8") as stream:
        stream.write("\n" + "\n".join(log) + "\n")
    (OUT / "PHASE3B_NEW_FORMULA_LOG.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log), flush=True)


if __name__ == "__main__":
    main()
