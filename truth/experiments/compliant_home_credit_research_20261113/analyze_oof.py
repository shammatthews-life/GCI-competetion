from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


def _fold_metrics(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
    seed: int,
    folds: int,
) -> dict[str, list[float]]:
    split_iter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=seed
    ).split(np.zeros(len(y)), y)
    fold_auc: dict[str, list[float]] = {name: [] for name in predictions}
    for _, valid_idx in split_iter:
        for name, score in predictions.items():
            fold_auc[name].append(float(roc_auc_score(y[valid_idx], score[valid_idx])))
    return fold_auc


def compare_runs(
    data_dir: Path,
    run_dirs: list[Path],
    output_path: Path,
    seed: int = 314159,
    folds: int = 5,
) -> None:
    train = pd.read_csv(data_dir / "train.csv", usecols=[ID_COLUMN, TARGET_COLUMN])
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    predictions: dict[str, np.ndarray] = {}
    source_runs: dict[str, str] = {}
    for run_dir in run_dirs:
        frame = pd.read_csv(run_dir / "oof_predictions.csv")
        if not np.array_equal(frame[ID_COLUMN].to_numpy(), train[ID_COLUMN].to_numpy()):
            raise ValueError(f"OOF IDs/order mismatch for {run_dir}")
        for column in frame.columns:
            if not column.startswith("OOF_"):
                continue
            name = f"{run_dir.name}:{column.removeprefix('OOF_')}"
            values = frame[column].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite OOF predictions in {name}")
            predictions[name] = values
            source_runs[name] = run_dir.name

    if not predictions:
        raise ValueError("No OOF prediction columns found.")
    fold_auc = _fold_metrics(y, predictions, seed, folds)

    equal_probability = np.mean(np.column_stack(list(predictions.values())), axis=1)
    ensemble_predictions = {
        "equal_probability_all_models": equal_probability,
    }
    ensemble_auc = _fold_metrics(y, ensemble_predictions, seed, folds)

    rows = []
    for name, score in predictions.items():
        fold_values = fold_auc[name]
        rows.append(
            {
                "candidate_model": name,
                "source_run": source_runs[name],
                "pooled_oof_auc": float(roc_auc_score(y, score)),
                "mean_fold_auc": float(np.mean(fold_values)),
                "std_fold_auc": float(np.std(fold_values, ddof=1)),
                "fold_auc": json.dumps(fold_values),
            }
        )
    for name, score in ensemble_predictions.items():
        fold_values = ensemble_auc[name]
        rows.append(
            {
                "candidate_model": name,
                "source_run": "fixed equal-weight diagnostic",
                "pooled_oof_auc": float(roc_auc_score(y, score)),
                "mean_fold_auc": float(np.mean(fold_values)),
                "std_fold_auc": float(np.std(fold_values, ddof=1)),
                "fold_auc": json.dumps(fold_values),
            }
        )
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path / "model_metrics.csv", index=False)

    fold_deltas: dict[str, dict[str, float]] = {}
    run_names = list(dict.fromkeys(source_runs.values()))
    if len(run_names) >= 2:
        baseline = run_names[0]
        for candidate in run_names[1:]:
            common_models = sorted(
                {
                    name.split(":", 1)[1]
                    for name in predictions
                    if source_runs[name] == baseline
                }
                & {
                    name.split(":", 1)[1]
                    for name in predictions
                    if source_runs[name] == candidate
                }
            )
            for model in common_models:
                base_name = f"{baseline}:{model}"
                candidate_name = f"{candidate}:{model}"
                delta = np.asarray(fold_auc[candidate_name]) - np.asarray(
                    fold_auc[base_name]
                )
                fold_deltas[f"{candidate}:{model}-vs-{baseline}:{model}"] = {
                    "fold_delta_auc": delta.tolist(),
                    "mean_fold_delta_auc": float(delta.mean()),
                    "folds_improved": int((delta > 0).sum()),
                    "folds_tied": int((delta == 0).sum()),
                }

    correlations = {}
    names = list(predictions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            values = []
            _, splits = zip(
                *list(
                    StratifiedKFold(
                        n_splits=folds, shuffle=True, random_state=seed
                    ).split(np.zeros(len(y)), y)
                )
            )
            for valid_idx in splits:
                values.append(
                    float(
                        spearmanr(
                            predictions[left][valid_idx],
                            predictions[right][valid_idx],
                        ).statistic
                    )
                )
            correlations[f"{left}|{right}"] = {
                "mean_within_fold_spearman": float(np.mean(values)),
                "fold_spearman": values,
            }

    summary = {
        "seed": seed,
        "folds": folds,
        "n_train": len(y),
        "candidate_runs": [run.name for run in run_dirs],
        "paired_fold_deltas": fold_deltas,
        "within_fold_spearman": correlations,
        "interpretation": (
            "Fold means and paired fold deltas are primary. Pooled OOF AUC is "
            "also reported but can reflect score-scale differences across folds. "
            "The ensemble is a fixed equal-probability diagnostic, not a tuned blend."
        ),
    }
    (output_path / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(r"D:\gci\inpuy"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    compare_runs(
        args.data_dir, args.run_dirs, args.output_dir, args.seed, args.folds
    )


if __name__ == "__main__":
    main()
