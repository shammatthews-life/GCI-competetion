"""
Scratch Experiment: Testing Fold 1 Lift for Actuarial Implied APR, Merton Solvency, and XGBoost GPU
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import ndtr
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

DATA_DIR = r"D:\gci\inpuy"
tr = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
y = tr["TARGET"].to_numpy().astype(int)

CATS = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
EXT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
FREQ_COLS = CATS + ["REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY", "AMT_INCOME_TOTAL"]
GROUP_KEYS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "REGION_POPULATION_RELATIVE"]
GROUP_VARS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "CREDIT_INC"]
TE_M0 = {"Q3": 100., "Q4": 30., "Q5": 30., "P3": 100., "P4": 30., "P5": 30., "PF4": 30.}

def rowwise_enhanced(df):
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

    # 1. Actuarial Implied Interest Rate Solver (vectorized Newton-Raphson across candidate terms)
    q_ratio = (an / np.maximum(cr, 1.0)).to_numpy()
    best_res = np.full(len(df), 1e9, dtype=np.float32)
    best_term = np.full(len(df), 36.0, dtype=np.float32)
    for n in [12, 24, 36, 48, 60]:
        mu = np.maximum(q_ratio - 1.0 / n, 1e-4)
        for _ in range(5):
            den = np.maximum(1.0 - (1.0 + mu) ** (-n), 1e-6)
            f_val = mu / den - q_ratio
            df_val = (den - mu * n * (1.0 + mu) ** (-n - 1)) / (den ** 2)
            df_val = np.where(np.abs(df_val) < 1e-6, 1e-6, df_val)
            mu = np.clip(mu - f_val / df_val, 1e-5, 2.0)
        res_n = np.abs(mu / np.maximum(1.0 - (1.0 + mu) ** (-n), 1e-6) - q_ratio)
        d[f"IMPLIED_APR_{n}"] = 12.0 * mu
        d[f"IMPLIED_RES_{n}"] = res_n
        better = res_n < best_res
        best_res = np.where(better, res_n, best_res)
        best_term = np.where(better, float(n), best_term)
    
    d["ACTUARIAL_BEST_TERM"] = best_term
    tot_pay = an * best_term
    d["ACTUARIAL_TOTAL_INTEREST"] = tot_pay - cr
    d["ACTUARIAL_INTEREST_RATIO"] = (tot_pay - cr) / np.maximum(cr, 1.0)
    d["ACTUARIAL_TOTAL_INC_RATIO"] = tot_pay / np.maximum(inc, 1.0)

    # 2. Merton Structural Solvency Distance-to-Default
    inc_m = inc.to_numpy() / 12.0
    an_v = np.maximum(an.to_numpy(), 1.0)
    cr_v = np.maximum(cr.to_numpy(), 1.0)
    T = np.clip(cr_v / (12.0 * an_v), 0.25, 10.0)
    cf_ratio = np.maximum(inc_m / an_v, 1e-4)
    for sigma in [0.15, 0.25]:
        dd = (np.log(cf_ratio) + (0.03 - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d[f"MERTON_DD_{int(sigma*100)}"] = dd
        d[f"MERTON_PD_{int(sigma*100)}"] = ndtr(-dd)

    # 3. External bureau manifold & Clayton Copula
    e = df[EXT]
    x = {
        "EXT_MEAN": e.mean(axis=1), "EXT_MIN": e.min(axis=1), "EXT_MAX": e.max(axis=1),
        "EXT_STD": e.std(axis=1), "EXT_RANGE": e.max(axis=1) - e.min(axis=1),
        "EXT_NOBS": e.notna().sum(axis=1), "EXT_PROD": e.prod(axis=1),
        "EXT_W": (df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4) / 9.0,
        "E12": df.EXT_SOURCE_1 * df.EXT_SOURCE_2, "E13": df.EXT_SOURCE_1 * df.EXT_SOURCE_3,
        "E23": df.EXT_SOURCE_2 * df.EXT_SOURCE_3, "D12": df.EXT_SOURCE_1 - df.EXT_SOURCE_2,
        "D13": df.EXT_SOURCE_1 - df.EXT_SOURCE_3, "D23": df.EXT_SOURCE_2 - df.EXT_SOURCE_3
    }

    # Clayton Copula on external scores (joint lower tail collapse)
    u1 = np.clip(df.EXT_SOURCE_1.to_numpy(), 1e-4, 1.0 - 1e-4)
    u2 = np.clip(df.EXT_SOURCE_2.to_numpy(), 1e-4, 1.0 - 1e-4)
    u3 = np.clip(df.EXT_SOURCE_3.to_numpy(), 1e-4, 1.0 - 1e-4)
    for theta in [1.5, 2.5]:
        s = u1**(-theta) + u2**(-theta) + u3**(-theta) - 2.0
        c_val = np.where(s > 0, s**(-1.0 / theta), 0.0)
        nan_m = df.EXT_SOURCE_1.isna() | df.EXT_SOURCE_2.isna() | df.EXT_SOURCE_3.isna()
        x[f"CLAYTON_COPULA_{int(theta*10)}"] = np.where(nan_m, np.nan, c_val)

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
        grid = np.arange(0.0, 400.5, 1.0)
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
    ext = [c for c in cols if "EXT" in c or c in ("E12", "E13", "E23", "D12", "D13", "D23", "CLAYTON_COPULA_15", "CLAYTON_COPULA_25")]
    loan = [c for c in cols if c.startswith(("AMT_", "LOG_", "CREDIT_", "ANN_", "GOODS_", "TERM", "Q0num", "IMPLIED_", "ACTUARIAL_", "MERTON_")) or c in ("CREDIT_MINUS_GOODS", "RP_LOGINC", "RR_LOGINC", "RP_LOGCR", "RR_LOGCR")]
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
        return df.astype("float32")
    return vf(ref_c), {k: vf(v) for k, v in oth_c.items()}

def build_fold(df_a, y_a, others):
    rs = RefStats(df_a)
    dfs = {"ref": df_a.reset_index(drop=True), **{k: v.reset_index(drop=True) for k, v in others.items()}}
    F, raws = {}, {}
    for nm, d in dfs.items():
        base = d.drop(columns=["SK_ID_CURR"]).copy()
        for c in CATS:
            base[c] = rs.codes(d[c], c)
        core_rw, ext_rw = rowwise_enhanced(d)
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

    F["_meta"] = dict(fee=rs.fee, fee_scan_best=rs.fee_scan_best, fee_scan_zero=rs.fee_scan_zero,
                      fee_share=float(F["ref"]["fee_flag"][(df_a.NAME_CONTRACT_TYPE == "Cash loans").to_numpy()].mean()))
    return F

STREAM_VIEWS = ["VIEW_EXT", "VIEW_LOAN", "VIEW_PERSON", "VIEW_EXTAGE", "VIEW_NOEXT", "VIEW_GROUPS", "VD_EL", "VD_EP", "VD_LP"]
SPECS = {
    "qf_shallow": dict(frame="oh_core", rounds=1300, params=dict(learning_rate=0.05, num_leaves=7, max_depth=3, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20)),
    "qf_reg":     dict(frame="oh_core", rounds=1939, params=dict(learning_rate=0.03, num_leaves=11, min_child_samples=300, feature_fraction=0.4, bagging_fraction=0.8, lambda_l2=40)),
    "allfam":     dict(frame="oh_all",  rounds=881,  params=dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20)),
    "mono":       dict(frame="core",    rounds=1571, params=dict(learning_rate=0.02, num_leaves=15, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20), mono=True)
}
LGB_MEMBER = dict(objective="binary", bagging_freq=1, extra_trees=True, **LGB_BASE)

def member_frame(F, nm, spec):
    return pd.concat([F[nm][spec["frame"]], F[nm]["te"], F[nm]["views"][STREAM_VIEWS]], axis=1)

print("[*] Commencing Fold 1 Rigorous Experiment...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold1_train_idx, fold1_val_idx = next(skf.split(tr, y))

df_tr_f1 = tr.iloc[fold1_train_idx].drop(columns=["TARGET"])
df_val_f1 = tr.iloc[fold1_val_idx].drop(columns=["TARGET"])
y_tr_f1 = y[fold1_train_idx]
y_val_f1 = y[fold1_val_idx]

t0 = time.time()
F1 = build_fold(df_tr_f1, y_tr_f1, {"val": df_val_f1})
print(f"[+] Fold 1 Features Built in {time.time()-t0:.1f}s | Fee: {F1['_meta']['fee']} | Fee Share: {F1['_meta']['fee_share']:.3f}")

preds_f1 = {}
rk = lambda v: (rankdata(v) - 1.0) / (len(v) - 1.0)

for m_name, spec in SPECS.items():
    fa = member_frame(F1, "ref", spec)
    fv = member_frame(F1, "val", spec)
    p = {**LGB_MEMBER, **spec["params"]}
    if spec.get("mono"):
        p["monotone_constraints"] = [-1 if c in EXT else 0 for c in fa.columns]
    mod = lgb.train(p, lgb.Dataset(fa, y_tr_f1), spec["rounds"])
    pred = mod.predict(fv)
    preds_f1[m_name] = pred
    print(f"  * [LGBM {m_name:10s}] Fold 1 AUC: {roc_auc_score(y_val_f1, pred):.5f}")

# CatBoost GPU
cbf = lambda nm: pd.concat([F1[nm]["all"].drop(columns=CATS), F1[nm]["cb"]], axis=1)
cb_train = cbf("ref")
cb_val = cbf("val")
cat_cols = list(F1["ref"]["cb"].columns)

cb_mod = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=8, loss_function="Logloss", eval_metric="AUC", random_seed=42, task_type="GPU", verbose=False)
cb_mod.fit(cb_train, y_tr_f1, cat_features=cat_cols)
cb_pred = cb_mod.predict_proba(cb_val)[:, 1]
preds_f1["catboost"] = cb_pred
print(f"  * [CatBoost GPU   ] Fold 1 AUC: {roc_auc_score(y_val_f1, cb_pred):.5f}")

# XGBoost GPU Hist on oh_core
spec_xgb = SPECS["qf_shallow"]
xgb_train_x = member_frame(F1, "ref", spec_xgb)
xgb_val_x = member_frame(F1, "val", spec_xgb)

dtrain_xgb = xgb.DMatrix(xgb_train_x, label=y_tr_f1)
dval_xgb = xgb.DMatrix(xgb_val_x, label=y_val_f1)
xgb_params = {
    "tree_method": "hist",
    "device": "cuda",
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.03,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "reg_lambda": 20.0,
    "seed": 42
}
xgb_mod = xgb.train(xgb_params, dtrain_xgb, num_boost_round=1000)
xgb_pred = xgb_mod.predict(dval_xgb)
preds_f1["xgboost"] = xgb_pred
print(f"  * [XGBoost GPU    ] Fold 1 AUC: {roc_auc_score(y_val_f1, xgb_pred):.5f}")

# Baseline V52 blend (4 LGBM + CB)
lgb_blend_v52 = 0.25 * rk(preds_f1["qf_shallow"]) + 0.15 * rk(preds_f1["qf_reg"]) + 0.35 * rk(preds_f1["allfam"]) + 0.25 * rk(preds_f1["mono"])
final_v52_f1 = (1.0 - 0.15) * lgb_blend_v52 + 0.15 * rk(preds_f1["catboost"])
auc_v52_f1 = roc_auc_score(y_val_f1, final_v52_f1)

# V53 Triad blend (4 LGBM + CB + XGB)
triad_blend = 0.70 * lgb_blend_v52 + 0.15 * rk(preds_f1["catboost"]) + 0.15 * rk(preds_f1["xgboost"])
auc_triad_f1 = roc_auc_score(y_val_f1, triad_blend)

print("\n" + "=" * 60)
print(f"FOLD 1 AUC COMPARISON:")
print(f"  V52 Actual Fold 1 AUC (from log) : 0.77049")
print(f"  V53 Enhanced (New Features) AUC : {auc_v52_f1:.5f} (Delta vs V52: {auc_v52_f1 - 0.77049:+.5f})")
print(f"  V53 Triad (+ XGBoost GPU)   AUC : {auc_triad_f1:.5f} (Delta vs V52: {auc_triad_f1 - 0.77049:+.5f})")
print("=" * 60)
