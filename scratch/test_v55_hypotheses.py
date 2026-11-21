import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb

# Set up paths
DATA_DIR = r"D:\gci\inpuy"
tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
te = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
y = tr["TARGET"].to_numpy().astype(int)

CATS = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
EXT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
FREQ_COLS = CATS + ["REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY", "AMT_INCOME_TOTAL"]
GROUP_KEYS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "REGION_POPULATION_RELATIVE"]
GROUP_VARS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "CREDIT_INC"]
TE_M0 = {"Q3": 100., "Q4": 30., "Q5": 30., "P3": 100., "P4": 30., "P5": 30., "PF4": 30.}

def rowwise(df, add_v55_features=True):
    inc, cr, an, gp, fam = df.AMT_INCOME_TOTAL, df.AMT_CREDIT, df.AMT_ANNUITY, df.AMT_GOODS_PRICE, df.CNT_FAM_MEMBERS
    de = df.DAYS_EMPLOYED.where(df.DAYS_EMPLOYED != 365243)
    age = -df.DAYS_BIRTH
    d = {
        "DE_SENT": (df.DAYS_EMPLOYED == 365243).astype(int),
        "DAYS_EMPLOYED_CLEAN": de,
        "ORG_XNA": (df.ORGANIZATION_TYPE == "XNA").astype(int),
        "CAR_AGE_OUT": (df.OWN_CAR_AGE >= 60).astype(int),
        "MISS_CNT": df.isna().sum(axis=1)
    }
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "AMT_REQ_CREDIT_BUREAU_YEAR"]:
        d[c + "_NA"] = df[c].isna().astype(int)

    d.update(
        CREDIT_INC=cr / inc, ANN_INC=an / inc, ANN_CREDIT=an / cr, TERM=cr / an,
        GOODS_CREDIT=gp / cr, CREDIT_MINUS_GOODS=cr - gp, GOODS_INC=gp / inc,
        INC_PER_FAM=inc / fam, CREDIT_PER_FAM=cr / fam, ANN_PER_FAM=an / fam
    )
    for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]:
        d["LOG_" + c] = np.log1p(df[c])

    d.update(
        AGE=age / 365.25, EMP_AGE=de / df.DAYS_BIRTH, EMP_YEARS=-de / 365.25,
        REG_AGE=df.DAYS_REGISTRATION / df.DAYS_BIRTH, ID_AGE=df.DAYS_ID_PUBLISH / df.DAYS_BIRTH,
        PHONE_AGE=df.DAYS_LAST_PHONE_CHANGE / df.DAYS_BIRTH, CAR_AGE_RATIO=df.OWN_CAR_AGE / (age / 365.25),
        CREDIT_PER_AGE=cr / (age / 365.25), INC_PER_AGE=inc / (age / 365.25)
    )
    b = [c for c in df.columns if c.startswith("AMT_REQ")]
    d.update(
        BUR_SUM=df[b].sum(axis=1, min_count=1),
        BUR_RECENT=df[["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON"]].sum(axis=1, min_count=1),
        BUR_MON_YEAR=df.AMT_REQ_CREDIT_BUREAU_MON / (df.AMT_REQ_CREDIT_BUREAU_YEAR + 1.0)
    )

    e = df[EXT]
    ext_m = e.mean(axis=1)
    x = {
        "EXT_MEAN": ext_m, "EXT_MIN": e.min(axis=1), "EXT_MAX": e.max(axis=1),
        "EXT_STD": e.std(axis=1), "EXT_RANGE": e.max(axis=1) - e.min(axis=1),
        "EXT_NOBS": e.notna().sum(axis=1), "EXT_PROD": e.prod(axis=1),
        "EXT_W": (df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4) / 9.0,
        "E12": df.EXT_SOURCE_1 * df.EXT_SOURCE_2, "E13": df.EXT_SOURCE_1 * df.EXT_SOURCE_3,
        "E23": df.EXT_SOURCE_2 * df.EXT_SOURCE_3, "D12": df.EXT_SOURCE_1 - df.EXT_SOURCE_2,
        "D13": df.EXT_SOURCE_1 - df.EXT_SOURCE_3, "D23": df.EXT_SOURCE_2 - df.EXT_SOURCE_3
    }

    if add_v55_features:
        # High-signal non-collinear solvency and bureau leverage extensions
        d.update(
            MONTHLY_CASH_MARGIN=(inc / 12.0) - an,
            MONTHLY_BURDEN=an / ((inc / 12.0) + 1.0),
            CREDIT_PER_EXT=cr / (ext_m + 0.001),
            ANN_PER_EXT=an / (ext_m + 0.001),
            EXT_OVER_AGE=ext_m / (age / 365.25),
            DOWNPAY_RATIO=(cr - gp) / (gp + 1.0)
        )

    clean = lambda z: pd.DataFrame(z, index=df.index).replace([np.inf, -np.inf], np.nan).astype("float32")
    return clean(d), clean(x)

class RefStats:
    def __init__(self, df_ref):
        self.cat_levels = {c: sorted(df_ref[c].dropna().unique().tolist()) for c in CATS}
        self.freq = {c: df_ref[c].value_counts() for c in FREQ_COLS}
        tmp = self._grp_frame(df_ref)
        self.grp = {(k, v): tmp.groupby(k)[v].agg(["sum", "count"]) for k in GROUP_KEYS for v in GROUP_VARS}
        cash = (df_ref.NAME_CONTRACT_TYPE == "Cash loans")
        A, C = df_ref.AMT_ANNUITY, df_ref.AMT_CREDIT
        grid = np.arange(0.0, 400.25, 0.5)
        sc = []
        for F in grid:
            v = ((A - F) / C)[cash].round(5)
            sc.append(float((v.map(v.value_counts()) >= 20).mean()))
        sc = np.array(sc)
        self.fee = float(np.median(grid[sc >= sc.max() - 0.001]))
        self.fee_scan_best = float(sc.max())
        self.fee_scan_zero = float(sc[0])
        self.q5c = (A / C).round(5).value_counts()
        self.qb5c = ((A - self.fee) / C).round(5).value_counts()

    @staticmethod
    def _grp_frame(df):
        t = df[list(set(GROUP_KEYS + GROUP_VARS) - {"CREDIT_INC"})].copy()
        t["CREDIT_INC"] = df.AMT_CREDIT / df.AMT_INCOME_TOTAL
        return t

    def codes(self, s, c):
        return pd.Series(pd.Categorical(s, categories=self.cat_levels[c]).codes, index=s.index).astype(float).replace(-1.0, np.nan)

    def onehot(self, df):
        cols = {}
        for c in CATS:
            code = pd.Categorical(df[c], categories=self.cat_levels[c]).codes
            code = np.where(code < 0, len(self.cat_levels[c]), code)
            for j in range(len(self.cat_levels[c]) + 1):
                cols[f"OH_{c}_{j}"] = (code == j).astype(np.float32)
        return pd.DataFrame(cols, index=df.index)

    def cross(self, df, is_ref):
        out = {}
        for c in FREQ_COLS:
            cnt = df[c].map(self.freq[c]).fillna(0).astype(float)
            out["CNT_" + c] = (cnt - df[c].notna().astype(float)).clip(lower=0) if is_ref else cnt
        tmp = self._grp_frame(df)
        for k in GROUP_KEYS:
            for v in GROUP_VARS:
                st = self.grp[(k, v)]
                S = tmp[k].map(st["sum"]).astype(float)
                N = tmp[k].map(st["count"]).astype(float)
                x = tmp[v]
                if is_ref:
                    S = S - x.fillna(0)
                    N = N - x.notna().astype(float)
                out[f"G_{k[:6]}_{v[-6:]}_DIFF"] = x - (S / N.where(N > 0))
        A, C = df.AMT_ANNUITY, df.AMT_CREDIT
        q = A / C
        qb = (A - self.fee) / C
        cq = q.round(5).map(self.q5c).fillna(0)
        cb = qb.round(5).map(self.qb5c).fillna(0)
        if is_ref:
            cq = cq - q.notna()
            cb = cb - qb.notna()
        fee_flag = ((cb > cq).to_numpy() & (df.NAME_CONTRACT_TYPE == "Cash loans").to_numpy()).astype(int)
        Q0 = pd.Series(np.where(fee_flag == 1, qb, q), index=df.index)
        s = lambda z, r: z.round(r).astype(str)
        keys = {
            "Q3": s(q, 3), "Q4": s(q, 4), "Q5": s(q, 5),
            "P3": s(Q0, 3), "P4": s(Q0, 4), "P5": s(Q0, 5),
            "PF4": s(Q0, 4) + "|" + pd.Series(fee_flag, index=df.index).astype(str)
        }
        return pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan).astype("float32"), keys, Q0, fee_flag

def te_fit(keys, yv):
    prior = float(np.mean(yv))
    return {k: pd.DataFrame({"g": s.values, "y": yv}).groupby("g")["y"].agg(["sum", "count"]) for k, s in keys.items()}, prior

def te_apply(st, prior, keys):
    out = {}
    for k, s in keys.items():
        g = st[k]
        cnt = s.map(g["count"]).fillna(0).to_numpy(float)
        sm = s.map(g["sum"]).fillna(0).to_numpy(float)
        out["TE_" + k] = (sm + TE_M0[k] * prior) / (cnt + TE_M0[k])
        out["TN_" + k] = cnt
    return pd.DataFrame(out, index=next(iter(keys.values())).index).astype("float32")

def te_train_oof(keys, yv, K=5, seed=7):
    out = None
    for ia, ib in StratifiedKFold(K, shuffle=True, random_state=seed).split(np.zeros(len(yv)), yv):
        st, prior = te_fit({k: s.iloc[ia] for k, s in keys.items()}, yv[ia])
        part = te_apply(st, prior, {k: s.iloc[ib] for k, s in keys.items()})
        if out is None:
            out = pd.DataFrame(index=next(iter(keys.values())).index, columns=part.columns, dtype="float32")
        out.iloc[ib] = part.values
    return out.astype("float32")

RES_T = {
    "EXT1": (lambda r: r.EXT_SOURCE_1, ["EXT_SOURCE_1"]),
    "EXT2": (lambda r: r.EXT_SOURCE_2, ["EXT_SOURCE_2"]),
    "EXT3": (lambda r: r.EXT_SOURCE_3, ["EXT_SOURCE_3"]),
    "LOGINC": (lambda r: np.log(r.AMT_INCOME_TOTAL), ["AMT_INCOME_TOTAL"]),
    "LOGCR": (lambda r: np.log(r.AMT_CREDIT), ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]),
    "LOGAN": (lambda r: np.log(r.AMT_ANNUITY), ["AMT_ANNUITY"]),
    "LOGGP": (lambda r: np.log(r.AMT_GOODS_PRICE), ["AMT_GOODS_PRICE"]),
    "EMP": (lambda r: r.DAYS_EMPLOYED, ["DAYS_EMPLOYED"]),
    "REG": (lambda r: r.DAYS_REGISTRATION, ["DAYS_REGISTRATION"]),
    "AGE": (lambda r: r.DAYS_BIRTH, ["DAYS_BIRTH"]),
    "IDP": (lambda r: r.DAYS_ID_PUBLISH, ["DAYS_ID_PUBLISH"]),
    "PHONE": (lambda r: r.DAYS_LAST_PHONE_CHANGE, ["DAYS_LAST_PHONE_CHANGE"]),
    "REGPOP": (lambda r: r.REGION_POPULATION_RELATIVE, ["REGION_POPULATION_RELATIVE"]),
    "FAM": (lambda r: r.CNT_FAM_MEMBERS, ["CNT_FAM_MEMBERS"]),
    "CAR": (lambda r: r.OWN_CAR_AGE, ["OWN_CAR_AGE"]),
    "EXT1D": (lambda r: r.EXT_SOURCE_1, "DEMO"),
    "EXT2D": (lambda r: r.EXT_SOURCE_2, "DEMO"),
    "EXT3D": (lambda r: r.EXT_SOURCE_3, "DEMO")
}
LGB_BASE = dict(verbose=-1, num_threads=4, seed=42, deterministic=True, force_row_wise=True)
RES_PARAMS = dict(objective="regression", learning_rate=0.1, num_leaves=31, min_child_samples=100, **LGB_BASE)

def residual_block(raw_ref, raw_others):
    out_ref, out_oth = {}, {k: {} for k in raw_others}
    for nm in list(RES_T):
        fn, drop = RES_T[nm]
        cols = [c for c in raw_ref.columns if (not (c.startswith("EXT_SOURCE") or c.startswith("AMT_")) if drop == "DEMO" else c not in drop)]
        yt = fn(raw_ref).astype(float)
        obs = yt.notna().to_numpy()
        pred = np.zeros(len(raw_ref))
        for tri, tei in KFold(3, shuffle=True, random_state=0).split(raw_ref):
            fit = tri[obs[tri]]
            if len(fit) < 50:
                pred[tei] = float(yt.iloc[fit].mean()) if len(fit) else 0.0
                continue
            pred[tei] = lgb.train(RES_PARAMS, lgb.Dataset(raw_ref.iloc[fit][cols], yt.iloc[fit]), 100).predict(raw_ref.iloc[tei][cols])
        m = lgb.train(RES_PARAMS, lgb.Dataset(raw_ref[cols][obs], yt[obs]), 100)
        out_ref["RP_" + nm] = pred
        out_ref["RR_" + nm] = np.where(obs, yt.to_numpy() - pred, np.nan)
        for k, r in raw_others.items():
            p = m.predict(r[cols])
            v = fn(r).astype(float).to_numpy()
            out_oth[k]["RP_" + nm] = p
            out_oth[k]["RR_" + nm] = np.where(~np.isnan(v), v - p, np.nan)
    return pd.DataFrame(out_ref).astype("float32"), {k: pd.DataFrame(v).astype("float32") for k, v in out_oth.items()}

VIEW_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=15, min_child_samples=150, lambda_l2=20, **LGB_BASE)

def view_columns(cols):
    ext = [c for c in cols if "EXT" in c or c in ("E12", "E13", "E23", "D12", "D13", "D23")]
    loan = [c for c in cols if c.startswith(("AMT_", "LOG_", "CREDIT_", "ANN_", "GOODS_", "TERM", "Q0num", "MONTHLY_")) or c in ("CREDIT_MINUS_GOODS", "RP_LOGINC", "RR_LOGINC", "RP_LOGCR", "RR_LOGCR", "DOWNPAY_RATIO")]
    person = [c for c in cols if c not in set(ext) | set(loan) and not c.startswith(("G_", "CNT_"))]
    agec = [c for c in cols if any(t in c for t in ("AGE", "BIRTH", "EMPLOY", "EMP_", "ID_", "REG_", "PHONE", "DAYS"))]
    grpc = [c for c in cols if c.startswith(("G_", "CNT_")) or c in CATS]
    return {
        "EXT": ext,
        "LOAN": loan,
        "PERSON": person,
        "EXTAGE": sorted(set(ext + agec)),
        "NOEXT": sorted(set(loan + person)),
        "GROUPS": sorted(set(grpc + loan[:6]))
    }

STREAM_VIEWS = [
    "VIEW_EXT", "VIEW_LOAN", "VIEW_PERSON", "VIEW_EXTAGE", "VIEW_NOEXT", "VIEW_GROUPS",
    "VD_EL", "VD_EP", "VD_LP", "VD_EN", "VD_EG", "VD_LG", "VD_EA"
]

def view_block(X_ref, y_ref, X_others):
    logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    ref_c, oth_c = {}, {k: {} for k in X_others}
    for vn, vc in view_columns(list(X_ref.columns)).items():
        inner = np.zeros(len(X_ref))
        acc = {k: np.zeros(len(v)) for k, v in X_others.items()}
        for ia, ib in StratifiedKFold(3, shuffle=True, random_state=11).split(np.zeros(len(X_ref)), y_ref):
            m = lgb.train(VIEW_PARAMS, lgb.Dataset(X_ref.iloc[ia][vc], y_ref[ia]), 100)
            inner[ib] = m.predict(X_ref.iloc[ib][vc])
            for k, v in X_others.items():
                acc[k] += m.predict(v[vc]) / 3.0
        ref_c["VIEW_" + vn] = logit(inner)
        for k in X_others:
            oth_c[k]["VIEW_" + vn] = logit(acc[k])

    def vf(d):
        df = pd.DataFrame(d)
        df["VD_EL"] = df.VIEW_EXT - df.VIEW_LOAN
        df["VD_EP"] = df.VIEW_EXT - df.VIEW_PERSON
        df["VD_LP"] = df.VIEW_LOAN - df.VIEW_PERSON
        df["VD_EN"] = df.VIEW_EXT - df.VIEW_NOEXT
        df["VD_EG"] = df.VIEW_EXT - df.VIEW_GROUPS
        df["VD_LG"] = df.VIEW_LOAN - df.VIEW_GROUPS
        df["VD_EA"] = df.VIEW_EXT - df.VIEW_EXTAGE
        return df.astype("float32")

    return vf(ref_c), {k: vf(v) for k, v in oth_c.items()}

def build_fold(df_a, y_a, others, add_v55=True):
    rs = RefStats(df_a)
    dfs = {"ref": df_a.reset_index(drop=True), **{k: v.reset_index(drop=True) for k, v in others.items()}}
    F, raws = {}, {}
    for nm, d in dfs.items():
        base = d.drop(columns=["SK_ID_CURR"]).copy()
        for c in CATS:
            base[c] = rs.codes(d[c], c)
        core_rw, ext_rw = rowwise(d, add_v55_features=add_v55)
        cross, keys, Q0, fee_flag = rs.cross(d, is_ref=(nm == "ref"))
        raw = base.copy()
        raw["DAYS_EMPLOYED"] = raw.DAYS_EMPLOYED.where(raw.DAYS_EMPLOYED != 365243)
        raws[nm] = raw
        F[nm] = dict(base=base, core_rw=core_rw, ext_rw=ext_rw, cross=cross, keys=keys, Q0=Q0, fee_flag=fee_flag, oh=rs.onehot(d), df=d)

    res_ref, res_oth = residual_block(raws["ref"], {k: raws[k] for k in others})
    st, prior = te_fit(F["ref"]["keys"], y_a)
    te_ref = te_train_oof(F["ref"]["keys"], y_a)

    for nm in dfs:
        f = F[nm]
        res = res_ref if nm == "ref" else res_oth[nm]
        f["res"] = res
        f["te"] = te_ref if nm == "ref" else te_apply(st, prior, f["keys"])
        f["core"] = pd.concat([f["base"], f["core_rw"], res], axis=1)
        f["all"] = pd.concat([f["base"], f["core_rw"], f["ext_rw"], f["cross"], res], axis=1)
        f["oh_core"] = pd.concat([f["core"].drop(columns=CATS), f["oh"]], axis=1)
        f["oh_all"] = pd.concat([f["all"].drop(columns=CATS), f["oh"]], axis=1)
        f["viewin"] = f["all"].assign(Q0num=f["Q0"].astype("float32").values)

    vref, voth = view_block(F["ref"]["viewin"], y_a, {k: F[k]["viewin"] for k in others})
    for nm in dfs:
        F[nm]["views"] = vref if nm == "ref" else voth[nm]

    for nm in dfs:
        f = F[nm]
        cb = pd.DataFrame({c: f["df"][c].astype("string").fillna("__NA__").astype(str) for c in CATS}, index=f["df"].index)
        for k, s in f["keys"].items():
            cb["PROD_" + k] = s
        cb["PROD_FEE"] = f["fee_flag"].astype(str)
        f["cb"] = cb

    return F

rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)

SPECS = {
    "qf_shallow": dict(frame="oh_core", rounds=1300, params=dict(learning_rate=0.05, num_leaves=7, max_depth=3, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20)),
    "qf_reg":     dict(frame="oh_core", rounds=1939, params=dict(learning_rate=0.03, num_leaves=11, min_child_samples=300, feature_fraction=0.4, bagging_fraction=0.8, lambda_l2=40)),
    "allfam":     dict(frame="oh_all",  rounds=950,  params=dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20)),
    "mono":       dict(frame="core",    rounds=1571, params=dict(learning_rate=0.02, num_leaves=15, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20), mono=True)
}
LGB_MEMBER = dict(objective="binary", bagging_freq=1, extra_trees=True, **LGB_BASE)

def member_frame(F, nm, spec):
    return pd.concat([F[nm][spec["frame"]], F[nm]["te"], F[nm]["views"][STREAM_VIEWS]], axis=1)

if __name__ == "__main__":
    print("[*] Starting Fold 1 Empirical Evaluation of V55 Enhancements...")
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(tr)), y))
    a, b = folds[0]
    df_a = tr.iloc[a].drop(columns=["TARGET"])
    df_b = tr.iloc[b].drop(columns=["TARGET"])

    t0 = time.time()
    F = build_fold(df_a, y[a], {"val": df_b, "test": te}, add_v55=True)
    print(f"Features built in {time.time()-t0:.1f}s")

    preds = {}
    for m in ["qf_shallow", "qf_reg", "allfam", "mono"]:
        spec = SPECS[m]
        fa = member_frame(F, "ref", spec)
        fv = member_frame(F, "val", spec)
        p = {**LGB_MEMBER, **spec["params"]}
        if spec.get("mono"):
            p["monotone_constraints"] = [-1 if c in EXT else 0 for c in fa.columns]
        mod = lgb.train(p, lgb.Dataset(fa, y[a]), spec["rounds"])
        preds[m] = mod.predict(fv)
        print(f"  [LGBM {m:10s}] Fold 1 AUC: {roc_auc_score(y[b], preds[m]):.5f}")

    # CatBoost GPU with 600 iterations
    cbf = lambda nm: pd.concat([F[nm]["all"].drop(columns=CATS), F[nm]["cb"]], axis=1)
    cb_mod = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.040, l2_leaf_reg=8, loss_function="Logloss", eval_metric="AUC", random_seed=42, task_type="GPU", verbose=False)
    cb_mod.fit(cbf("ref"), y[a], cat_features=list(F["ref"]["cb"].columns))
    cb_pred = cb_mod.predict_proba(cbf("val"))[:, 1]
    print(f"  [CatBoost GPU 600it] Fold 1 AUC: {roc_auc_score(y[b], cb_pred):.5f}")

    # XGBoost GPU member
    X_train_xgb = member_frame(F, "ref", SPECS["qf_reg"])
    X_val_xgb = member_frame(F, "val", SPECS["qf_reg"])
    xgb_mod = xgb.XGBClassifier(n_estimators=1200, learning_rate=0.03, max_depth=4, min_child_weight=50, subsample=0.8, colsample_bytree=0.5, reg_lambda=20.0, reg_alpha=5.0, tree_method="hist", device="cuda", random_state=42, eval_metric="auc")
    xgb_mod.fit(X_train_xgb, y[a], verbose=False)
    xgb_pred = xgb_mod.predict_proba(X_val_xgb)[:, 1]
    print(f"  [XGBoost GPU 1200it] Fold 1 AUC: {roc_auc_score(y[b], xgb_pred):.5f}")

    # Ensembles
    # 1. 4 LGBM + CB (V54 style weights)
    mem_blend = 0.25 * rk(preds["qf_shallow"]) + 0.20 * rk(preds["qf_reg"]) + 0.25 * rk(preds["allfam"]) + 0.30 * rk(preds["mono"])
    v54_style = 0.84 * mem_blend + 0.16 * rk(cb_pred)
    print(f"\n==> Enhanced 4-LGBM + CB600 Fold 1 AUC: {roc_auc_score(y[b], v54_style):.5f} (vs V54 baseline 0.77004)")

    # 2. Tri-Engine Zoo: 4 LGBM + CB GPU + XGB GPU
    for w_xgb in [0.05, 0.08, 0.10, 0.12]:
        tri_style = 0.76 * mem_blend + 0.16 * rk(cb_pred) + w_xgb * rk(xgb_pred)
        # renormalize
        total_w = 0.76 + 0.16 + w_xgb
        tri_auc = roc_auc_score(y[b], tri_style)
        print(f"==> Tri-Engine Zoo (XGB_W={w_xgb:.2f}) Fold 1 AUC: {tri_auc:.5f}")
