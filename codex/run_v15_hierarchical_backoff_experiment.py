"""Matched v15 experiment: explicit hierarchical product target encoding.

This runner executes the feature/model setup from the original v15 notebook,
loads its cached residual and view features, and reruns only the E1Aet
v1_allfam_oh member as a control and with one predeclared hierarchical TE block.
It then replaces only that member in the cached full v15 formula.

No submission file is written. The experiment is a development comparison on
the reused seed-42 folds; any apparent winner needs a fresh full-v15 confirm.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
CODEX = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "pre info" / "GCI_FINAL_v15_end_to_end_colab.ipynb"
V15_CACHE = CODEX / "v15_cache" / "gci_cache_final"
RESID_CACHE = CODEX / "v_3_cache" / "resid16.pkl"
OUT = CODEX / "v15_hierarchical_backoff_experiment"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT = 500


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S] ") + message, flush=True)


def read_cell(nb: dict, i: int) -> str:
    return "".join(nb["cells"][i].get("source", []))


def bootstrap_delta(y: np.ndarray, base: np.ndarray, cand: np.ndarray,
                    seed: int = 7319) -> list[float]:
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        idx = np.concatenate((
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ))
        vals[i] = roc_auc_score(y[idx], cand[idx]) - roc_auc_score(y[idx], base[idx])
    return [float(x) for x in np.quantile(vals, [0.025, 0.5, 0.975])]


def main() -> None:
    t0 = time.time()
    if not NOTEBOOK.exists() or not V15_CACHE.exists() or not RESID_CACHE.exists():
        raise FileNotFoundError(
            f"Missing notebook/cache input: notebook={NOTEBOOK.exists()}, "
            f"v15_cache={V15_CACHE.exists()}, residual_cache={RESID_CACHE.exists()}"
        )

    import json as _json
    nb = _json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__v15_experiment__"}

    log("Executing v15 setup and feature-engineering cells")
    setup_cell = read_cell(nb, 1)
    setup_cell = setup_cell.replace(
        'base_dir = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else "/content"\n'
        'CACHE_DIR = os.path.join(base_dir, "gci_cache_final") if not SMOKE else "/tmp/gci_cache_smoke"; OUT_DIR = os.path.join(base_dir, "gci_output_final") if not SMOKE else "/tmp/gci_out_smoke"\n'
        'os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)',
        f'base_dir = {str(CODEX)!r}\nCACHE_DIR = {str(V15_CACHE)!r}; OUT_DIR = {str(OUT)!r}\n'
        'os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)',
    )
    ns.update({"CODEX": CODEX, "V15_CACHE": V15_CACHE, "OUT": OUT})
    exec(compile(setup_cell, str(NOTEBOOK) + "#cell1", "exec"), ns)
    ns["CACHE_DIR"] = str(V15_CACHE)
    ns["OUT_DIR"] = str(OUT)
    ns["THREADS"] = 3
    exec(compile(read_cell(nb, 2), str(NOTEBOOK) + "#cell2", "exec"), ns)

    # The recovered residual pickle is the same 16-target v6 recipe referenced
    # by the v15 notebook; use it rather than rebuilding 16 HGB models.
    ns["get_resid"] = lambda seeds, name: pd.read_pickle(RESID_CACHE)
    exec(compile(read_cell(nb, 3), str(NOTEBOOK) + "#cell3", "exec"), ns)
    exec(compile(read_cell(nb, 4), str(NOTEBOOK) + "#cell4", "exec"), ns)

    y = np.asarray(ns["y"], dtype=np.int8)
    n = len(y)
    weights = {"qf_shallow": 0.24, "qf_reg": 0.12,
               "v1_allfam_oh": 0.32, "v2_allfam_oh": 0.16,
               "v2_mono": 0.16}
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    rk = lambda x: (rankdata(x, method="average") - 1) / (len(x) - 1)

    # Explicit hierarchical posterior rates, with fixed shrinkage values
    # aligned to v15's Q3/Q4 target-encoding priors (100 and 30).
    q4 = ns["KS"]["Q4"].astype(str).to_numpy(dtype=object)
    q3 = ns["KS"]["Q3"].astype(str).to_numpy(dtype=object)
    contract = ns["al"]["NAME_CONTRACT_TYPE"].fillna("__MISSING__").astype(str).to_numpy(dtype=object)
    q4_contract = np.char.add(np.char.add(q4.astype(str), "|"), contract.astype(str)).astype(object)

    def stats(keys: np.ndarray, fit_idx: np.ndarray):
        frame = pd.DataFrame({"key": keys[fit_idx], "target": y[fit_idx]})
        grouped = frame.groupby("key", sort=False)["target"].agg(["sum", "count"])
        return grouped["sum"], grouped["count"]

    def apply_map(keys: np.ndarray, query_idx: np.ndarray, sums: pd.Series,
                  counts: pd.Series, prior: np.ndarray | float,
                  alpha: float) -> tuple[np.ndarray, np.ndarray]:
        q = pd.Series(keys[query_idx])
        s = q.map(sums).fillna(0).to_numpy(dtype=np.float64)
        c = q.map(counts).fillna(0).to_numpy(dtype=np.float64)
        prior_arr = np.broadcast_to(np.asarray(prior, dtype=np.float64), c.shape)
        te = (s + alpha * prior_arr) / (c + alpha)
        return te.astype(np.float32), np.log1p(c).astype(np.float32)

    def hierarchical_enc(fit_idx: np.ndarray, query_idx: np.ndarray) -> pd.DataFrame:
        global_rate = float(np.mean(y[fit_idx]))
        s3, c3 = stats(q3, fit_idx)
        te3, cnt3 = apply_map(q3, query_idx, s3, c3, global_rate, 100.0)

        s4, c4 = stats(q4, fit_idx)
        te4, cnt4 = apply_map(q4, query_idx, s4, c4, te3, 30.0)

        s4c, c4c = stats(q4_contract, fit_idx)
        te4c, cnt4c = apply_map(q4_contract, query_idx, s4c, c4c, te4, 30.0)
        return pd.DataFrame({
            "HIER_Q3_TE": te3,
            "HIER_Q4_TE": te4,
            "HIER_Q4_CONTRACT_TE": te4c,
            "HIER_Q3_LOGCOUNT": cnt3,
            "HIER_Q4_LOGCOUNT": cnt4,
            "HIER_Q4_CONTRACT_LOGCOUNT": cnt4c,
            "HIER_Q4_MINUS_Q3": te4 - te3,
            "HIER_Q4_CONTRACT_MINUS_Q4": te4c - te4,
        })

    original_enc = ns["enc"]
    original_train_enc = ns["train_enc"]

    def enc_hier(fit_idx, query_idx, keys=None):
        return pd.concat(
            [original_enc(fit_idx, query_idx, keys), hierarchical_enc(fit_idx, query_idx)],
            axis=1,
        )

    def train_enc_hier(fit_idx):
        base_cols = original_enc(fit_idx[:10], fit_idx[:5]).columns
        base_values = np.zeros((len(fit_idx), len(base_cols)), dtype=np.float32)
        extra = np.zeros((len(fit_idx), 8), dtype=np.float32)
        inner = StratifiedKFold(5, shuffle=True, random_state=7)
        for i_fit, i_query in inner.split(np.zeros(len(fit_idx)), y[fit_idx]):
            base_values[i_query] = original_enc(fit_idx[i_fit], fit_idx[i_query]).to_numpy(dtype=np.float32)
            extra[i_query] = hierarchical_enc(fit_idx[i_fit], fit_idx[i_query]).to_numpy(dtype=np.float32)
        base = pd.DataFrame(base_values, columns=base_cols)
        extra_df = pd.DataFrame(extra, columns=hierarchical_enc(fit_idx[:10], fit_idx[:5]).columns)
        return pd.concat([base.reset_index(drop=True), extra_df], axis=1)

    # Build the exact E1ET member definition used inside E1Aet.
    spec = {}
    for name, (X, Xt, params, rounds, full_rounds, seeds) in ns["SPEC1"].items():
        spec[name] = (X, Xt, {**params, "extra_trees": True}, rounds, full_rounds, seeds)

    # First rerun an unchanged control so local library/CPU differences are
    # shared by both arms. Then add the hierarchical encodings and rerun it.
    log("Rerunning unchanged E1Aet v1_allfam_oh control (5 OOF folds + full seeds)")
    ns["enc"] = original_enc
    ns["train_enc"] = original_train_enc
    control_test, control_oof = ns["run_member"](
        "v1_allfam_oh", spec, "hier_backoff_control", ns["VIEW_A"]
    )

    log("Training matched E1Aet v1_allfam_oh with hierarchical Q4-to-Q3-to-contract encodings")
    ns["enc"] = enc_hier
    ns["train_enc"] = train_enc_hier
    candidate_test, candidate_oof = ns["run_member"](
        "v1_allfam_oh", spec, "hier_backoff_candidate", ns["VIEW_A"]
    )
    ns["enc"] = original_enc
    ns["train_enc"] = original_train_enc

    # A second predeclared direction: household adult-equivalent affordability.
    # v15 has per-family ratios, but not child-adjusted adult-equivalent burden.
    al = ns["al"]
    family = al["CNT_FAM_MEMBERS"].clip(lower=1)
    children = al["CNT_CHILDREN"].clip(lower=0).fillna(0)
    adults_eq = (family - 0.5 * children).clip(lower=1)
    household_block = pd.DataFrame({
        "INCOME_PER_ADULT_EQ": al["AMT_INCOME_TOTAL"] / adults_eq,
        "CREDIT_PER_ADULT_EQ": al["AMT_CREDIT"] / adults_eq,
        "ANNUITY_PER_ADULT_EQ": al["AMT_ANNUITY"] / adults_eq,
        "DISPOSABLE_INCOME_PER_ADULT_EQ": (al["AMT_INCOME_TOTAL"] - al["AMT_ANNUITY"]) / adults_eq,
        "CHILDREN_PER_ADULT_EQ": children / adults_eq,
        "ADULT_EQ_FRACTION": adults_eq / family,
    }).replace([np.inf, -np.inf], np.nan)
    X0, Xt0, p0, rounds0, full_rounds0, seeds0 = spec["v1_allfam_oh"]
    Xh = pd.concat([X0.reset_index(drop=True), household_block.iloc[:n].reset_index(drop=True)], axis=1)
    Xth = pd.concat([Xt0.reset_index(drop=True), household_block.iloc[n:].reset_index(drop=True)], axis=1)
    household_spec = dict(spec)
    household_spec["v1_allfam_oh"] = (Xh, Xth, p0, rounds0, full_rounds0, seeds0)
    log("Training matched E1Aet v1_allfam_oh with six household adult-equivalent features")
    household_test, household_oof = ns["run_member"](
        "v1_allfam_oh", household_spec, "household_feature_candidate", ns["VIEW_A"]
    )

    # Rebuild cached champion from its component arrays, replacing only this
    # member. CatBoost OOF pieces are aligned with the same seed-42 folds.
    def cached_member(tag: str, member: str, key: str) -> np.ndarray:
        with np.load(V15_CACHE / f"{tag}_{member}.npz") as z:
            return z[key].copy()

    def full_formula(replacement: np.ndarray) -> np.ndarray:
        branches = {}
        for tag in ("old", "e1", "e1et", "e1Aet"):
            vals = {}
            for member in weights:
                if tag == "e1Aet" and member == "v1_allfam_oh":
                    raw = replacement
                else:
                    raw = cached_member(tag, member, "oof")
                vals[member] = rk(raw)
            branches[tag] = sum(weights[k] * vals[k] for k in weights)
        B = (branches["old"] + branches["e1"] + branches["e1et"]) / 3
        cb_parts = [np.load(V15_CACHE / f"catboost_product_fold_{i}.npz")["oof"] for i in range(1, 6)]
        cb_oof = np.zeros(n, dtype=np.float64)
        for (_, valid), pred in zip(folds, cb_parts):
            if len(valid) != len(pred):
                raise ValueError("CatBoost cached OOF rows do not match seed-42 folds")
            cb_oof[valid] = pred
        return 0.90 * rk(0.75 * rk(B) + 0.25 * branches["e1Aet"]) + 0.10 * rk(cb_oof)

    original_cache = np.load(CODEX / "claude_7_matched_v15_specialist_oof.npz")
    y_cache = original_cache["y"].copy()
    if not np.array_equal(y, y_cache):
        raise ValueError("Labels differ from the previously reconstructed v15 cache")
    baseline_oof = full_formula(control_oof)
    candidate_full_oof = full_formula(candidate_oof)
    household_full_oof = full_formula(household_oof)
    cached_champion_oof = original_cache["v15_oof"].copy()

    base_auc = float(roc_auc_score(y, baseline_oof))
    cached_auc = float(roc_auc_score(y, cached_champion_oof))
    fold_rows = []
    comparison_metrics = {}
    for label, candidate_pred in (("hierarchical_backoff", candidate_full_oof),
                                  ("household_adult_equivalent", household_full_oof)):
        cand_auc = float(roc_auc_score(y, candidate_pred))
        delta = cand_auc - base_auc
        fold_deltas = []
        for fi, (_, valid) in enumerate(folds, 1):
            a0 = roc_auc_score(y[valid], baseline_oof[valid])
            a1 = roc_auc_score(y[valid], candidate_pred[valid])
            fd = float(a1-a0)
            fold_deltas.append(fd)
            fold_rows.append({"variant": label, "fold": fi,
                              "control_auc": float(a0), "candidate_auc": float(a1),
                              "delta_auc": fd})
        ci = bootstrap_delta(y, baseline_oof, candidate_pred, seed=7319 if label.startswith("hierarchical") else 7320)
        comparison_metrics[label] = {
            "candidate_auc": cand_auc,
            "delta_auc": float(delta),
            "fold_deltas": fold_deltas,
            "positive_folds": int(sum(x > 0 for x in fold_deltas)),
            "paired_stratified_bootstrap_95_ci": ci,
            "passes_development_gate": bool(delta >= 0.00015 and sum(x > 0 for x in fold_deltas) >= 4 and ci[0] > 0),
        }
    agreement = float(spearmanr(control_oof, cached_member("e1Aet", "v1_allfam_oh", "oof")).statistic)

    metrics = {
        "experiment": "matched feature additions to one E1Aet v1_allfam_oh member",
        "candidate_scope": "replace only E1Aet v1_allfam_oh member in full v15 rank-space formula",
        "outer_folds": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
        "hierarchical_features": ["HIER_Q3_TE", "HIER_Q4_TE", "HIER_Q4_CONTRACT_TE",
                                  "HIER_Q3_LOGCOUNT", "HIER_Q4_LOGCOUNT",
                                  "HIER_Q4_CONTRACT_LOGCOUNT", "HIER_Q4_MINUS_Q3",
                                  "HIER_Q4_CONTRACT_MINUS_Q4"],
        "shrinkage": {"q3_to_global": 100, "q4_to_q3": 30, "q4_contract_to_q4": 30},
        "control_auc": base_auc,
        "cached_v15_auc": cached_auc,
        "control_vs_cached_v15_auc_delta": float(base_auc-cached_auc),
        "comparisons": comparison_metrics,
        "folds": fold_rows,
        "control_member_spearman_vs_cached_member": agreement,
        "promotion_gate": {"min_delta": 0.00015, "min_positive_folds": 4,
                           "ci_lower_above_zero": True},
        "fresh_confirmation_required": True,
        "submission_written": False,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    pd.DataFrame(fold_rows).to_csv(OUT / "fold_metrics.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez_compressed(OUT / "oof.npz", y=y, control_oof=baseline_oof,
                        hierarchical_backoff_oof=candidate_full_oof,
                        household_adult_equivalent_oof=household_full_oof,
                        cached_v15_oof=cached_champion_oof,
                        control_member_oof=control_oof, candidate_member_oof=candidate_oof,
                        household_member_oof=household_oof,
                        control_member_test=control_test, candidate_member_test=candidate_test,
                        household_member_test=household_test)
    log(json.dumps(metrics, indent=2))
    log(f"Artifacts saved under {OUT}")


if __name__ == "__main__":
    main()
