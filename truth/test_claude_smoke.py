"""
Smoke test of the Claude pipeline logic
"""
import os, sys, glob, json, time, hashlib, warnings, random, platform, inspect
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore")

SMOKE = True
PRESET = "full"
CB_TASK = "GPU"
SEED = 42; random.seed(SEED); np.random.seed(SEED)
THREADS = os.cpu_count() or 4
DATA_DIR = r"D:\gci\inpuy"

OFFICIAL_SHA256 = {
    "train.csv": "B8DD5A61771D8EB2821972C1B7D3F2F747CC66471F4D34153036D2D9C458ECF1",
    "test.csv": "DFB7AEB96E5AF3D8616ACAABA6B23CEC8680845CFD525F7C08DE15C6D2D42C65",
    "sample_submission.csv": "6E5AD587B7C0B3D9FA39C90CE03E161FF5EF292193E117A6A02B11075DFFE7DB"}

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""): h.update(blk)
    return h.hexdigest().upper()

PATHS = {}
for fname in ("train.csv", "test.csv", "sample_submission.csv"):
    p = os.path.join(DATA_DIR, fname)
    assert sha256_file(p) == OFFICIAL_SHA256[fname]
    PATHS[fname] = p

tr = pd.read_csv(PATHS["train.csv"]); te = pd.read_csv(PATHS["test.csv"])
sample_sub = pd.read_csv(PATHS["sample_submission.csv"])
assert sample_sub["SK_ID_CURR"].equals(te["SK_ID_CURR"])

# Smoke sample
tr = tr.sample(7000, random_state=0).reset_index(drop=True)
te = te.iloc[:2500].reset_index(drop=True)
sample_sub = sample_sub.iloc[:2500].reset_index(drop=True)
y = tr["TARGET"].to_numpy().astype(int)
print("train", tr.shape, "| test", te.shape, "| default rate %.4f" % y.mean())

STREAMS, MEMBERS, USE_CATBOOST, RES_TARGETS = ["v18"], ["qf_shallow", "qf_reg", "allfam", "mono"], True, None
MEMBER_W_ALL = {"qf_shallow": 0.25, "qf_reg": 0.15, "allfam": 0.35, "mono": 0.25}
STREAM_W_ALL = {"v17": 0.55, "v18": 0.45}; CB_W = 0.15
RES_ROUNDS, VIEW_ROUNDS, CB_ITERS, ROUND_SCALE = (25, 25, 40, 0.03)

CATS = ["NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"]
EXT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
FREQ_COLS = CATS + ["REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY", "AMT_INCOME_TOTAL"]
GROUP_KEYS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "REGION_POPULATION_RELATIVE"]
GROUP_VARS = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "CREDIT_INC"]
TE_M0 = {"Q3": 100., "Q4": 30., "Q5": 30., "P3": 100., "P4": 30., "P5": 30., "PF4": 30.}

def rowwise(df):
    inc, cr, an, gp, fam = df.AMT_INCOME_TOTAL, df.AMT_CREDIT, df.AMT_ANNUITY, df.AMT_GOODS_PRICE, df.CNT_FAM_MEMBERS
    de = df.DAYS_EMPLOYED.where(df.DAYS_EMPLOYED != 365243); age = -df.DAYS_BIRTH
    d = {"DE_SENT": (df.DAYS_EMPLOYED == 365243).astype(int), "DAYS_EMPLOYED_CLEAN": de, "ORG_XNA": (df.ORGANIZATION_TYPE == "XNA").astype(int),
         "CAR_AGE_OUT": (df.OWN_CAR_AGE >= 60).astype(int), "MISS_CNT": df.isna().sum(axis=1)}
    for c in ["EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE", "AMT_REQ_CREDIT_BUREAU_YEAR"]: d[c + "_NA"] = df[c].isna().astype(int)
    d.update(CREDIT_INC=cr / inc, ANN_INC=an / inc, ANN_CREDIT=an / cr, TERM=cr / an, GOODS_CREDIT=gp / cr, CREDIT_MINUS_GOODS=cr - gp, GOODS_INC=gp / inc,
             INC_PER_FAM=inc / fam, CREDIT_PER_FAM=cr / fam, ANN_PER_FAM=an / fam)
    for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]: d["LOG_" + c] = np.log1p(df[c])
    d.update(AGE=age / 365.25, EMP_AGE=de / df.DAYS_BIRTH, EMP_YEARS=-de / 365.25, REG_AGE=df.DAYS_REGISTRATION / df.DAYS_BIRTH, ID_AGE=df.DAYS_ID_PUBLISH / df.DAYS_BIRTH,
             PHONE_AGE=df.DAYS_LAST_PHONE_CHANGE / df.DAYS_BIRTH, CAR_AGE_RATIO=df.OWN_CAR_AGE / (age / 365.25), CREDIT_PER_AGE=cr / (age / 365.25), INC_PER_AGE=inc / (age / 365.25))
    b = [c for c in df.columns if c.startswith("AMT_REQ")]
    d.update(BUR_SUM=df[b].sum(axis=1, min_count=1), BUR_RECENT=df[["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_MON"]].sum(axis=1, min_count=1),
             BUR_MON_YEAR=df.AMT_REQ_CREDIT_BUREAU_MON / (df.AMT_REQ_CREDIT_BUREAU_YEAR + 1.0))
    e = df[EXT]
    x = {"EXT_MEAN": e.mean(axis=1), "EXT_MIN": e.min(axis=1), "EXT_MAX": e.max(axis=1), "EXT_STD": e.std(axis=1), "EXT_RANGE": e.max(axis=1) - e.min(axis=1), "EXT_NOBS": e.notna().sum(axis=1),
         "EXT_PROD": e.prod(axis=1), "EXT_W": (df.EXT_SOURCE_1 * 2 + df.EXT_SOURCE_2 * 3 + df.EXT_SOURCE_3 * 4) / 9.0, "E12": df.EXT_SOURCE_1 * df.EXT_SOURCE_2, "E13": df.EXT_SOURCE_1 * df.EXT_SOURCE_3,
         "E23": df.EXT_SOURCE_2 * df.EXT_SOURCE_3, "D12": df.EXT_SOURCE_1 - df.EXT_SOURCE_2, "D13": df.EXT_SOURCE_1 - df.EXT_SOURCE_3, "D23": df.EXT_SOURCE_2 - df.EXT_SOURCE_3}
    clean = lambda z: pd.DataFrame(z, index=df.index).replace([np.inf, -np.inf], np.nan).astype("float32")
    return clean(d), clean(x)

class RefStats:
    def __init__(self, df_ref):
        self.cat_levels = {c: sorted(df_ref[c].dropna().unique().tolist()) for c in CATS}
        self.freq = {c: df_ref[c].value_counts() for c in FREQ_COLS}
        tmp = self._grp_frame(df_ref); self.grp = {(k, v): tmp.groupby(k)[v].agg(["sum", "count"]) for k in GROUP_KEYS for v in GROUP_VARS}
        cash = (df_ref.NAME_CONTRACT_TYPE == "Cash loans"); A, C = df_ref.AMT_ANNUITY, df_ref.AMT_CREDIT
        grid = np.arange(0.0, 400.5, 1.0); sc = []
        for F in grid:
            v = ((A - F) / C)[cash].round(5); sc.append(float((v.map(v.value_counts()) >= 20).mean()))
        sc = np.array(sc); self.fee = float(np.median(grid[sc >= sc.max() - 0.001])); self.fee_scan_best = float(sc.max()); self.fee_scan_zero = float(sc[0])
        self.q5c = (A / C).round(5).value_counts(); self.qb5c = ((A - self.fee) / C).round(5).value_counts()
    @staticmethod
    def _grp_frame(df):
        t = df[list(set(GROUP_KEYS + GROUP_VARS) - {"CREDIT_INC"})].copy(); t["CREDIT_INC"] = df.AMT_CREDIT / df.AMT_INCOME_TOTAL; return t
    def codes(self, s, c): return pd.Series(pd.Categorical(s, categories=self.cat_levels[c]).codes, index=s.index).astype(float).replace(-1.0, np.nan)
    def onehot(self, df):
        cols = {}
        for c in CATS:
            code = pd.Categorical(df[c], categories=self.cat_levels[c]).codes; code = np.where(code < 0, len(self.cat_levels[c]), code)
            for j in range(len(self.cat_levels[c]) + 1): cols[f"OH_{c}_{j}"] = (code == j).astype(np.float32)
        return pd.DataFrame(cols, index=df.index)
    def cross(self, df, is_ref):
        out = {}
        for c in FREQ_COLS:
            cnt = df[c].map(self.freq[c]).fillna(0).astype(float)
            out["CNT_" + c] = (cnt - df[c].notna().astype(float)).clip(lower=0) if is_ref else cnt
        tmp = self._grp_frame(df)
        for k in GROUP_KEYS:
            for v in GROUP_VARS:
                st = self.grp[(k, v)]; S = tmp[k].map(st["sum"]).astype(float); N = tmp[k].map(st["count"]).astype(float); x = tmp[v]
                if is_ref: S = S - x.fillna(0); N = N - x.notna().astype(float)
                out[f"G_{k[:6]}_{v[-6:]}_DIFF"] = x - (S / N.where(N > 0))
        A, C = df.AMT_ANNUITY, df.AMT_CREDIT; q = A / C; qb = (A - self.fee) / C
        cq = q.round(5).map(self.q5c).fillna(0); cb = qb.round(5).map(self.qb5c).fillna(0)
        if is_ref: cq = cq - q.notna(); cb = cb - qb.notna()
        fee_flag = ((cb > cq).to_numpy() & (df.NAME_CONTRACT_TYPE == "Cash loans").to_numpy()).astype(int); Q0 = pd.Series(np.where(fee_flag == 1, qb, q), index=df.index)
        s = lambda z, r: z.round(r).astype(str)
        keys = {"Q3": s(q, 3), "Q4": s(q, 4), "Q5": s(q, 5), "P3": s(Q0, 3), "P4": s(Q0, 4), "P5": s(Q0, 5), "PF4": s(Q0, 4) + "|" + pd.Series(fee_flag, index=df.index).astype(str)}
        return pd.DataFrame(out, index=df.index).replace([np.inf, -np.inf], np.nan).astype("float32"), keys, Q0, fee_flag

def te_fit(keys, yv):
    prior = float(np.mean(yv)); return {k: pd.DataFrame({"g": s.values, "y": yv}).groupby("g")["y"].agg(["sum", "count"]) for k, s in keys.items()}, prior
def te_apply(st, prior, keys):
    out = {}
    for k, s in keys.items():
        g = st[k]; cnt = s.map(g["count"]).fillna(0).to_numpy(float); sm = s.map(g["sum"]).fillna(0).to_numpy(float)
        out["TE_" + k] = (sm + TE_M0[k] * prior) / (cnt + TE_M0[k]); out["TN_" + k] = cnt
    return pd.DataFrame(out, index=next(iter(keys.values())).index).astype("float32")
def te_train_oof(keys, yv, K=5, seed=7):
    out = None
    for ia, ib in StratifiedKFold(K, shuffle=True, random_state=seed).split(np.zeros(len(yv)), yv):
        st, prior = te_fit({k: s.iloc[ia] for k, s in keys.items()}, yv[ia]); part = te_apply(st, prior, {k: s.iloc[ib] for k, s in keys.items()})
        if out is None: out = pd.DataFrame(index=next(iter(keys.values())).index, columns=part.columns, dtype="float32")
        out.iloc[ib] = part.values
    return out.astype("float32")

RES_T = {"EXT1": (lambda r: r.EXT_SOURCE_1, ["EXT_SOURCE_1"]), "EXT2": (lambda r: r.EXT_SOURCE_2, ["EXT_SOURCE_2"]), "EXT3": (lambda r: r.EXT_SOURCE_3, ["EXT_SOURCE_3"]),
         "LOGINC": (lambda r: np.log(r.AMT_INCOME_TOTAL), ["AMT_INCOME_TOTAL"]), "LOGCR": (lambda r: np.log(r.AMT_CREDIT), ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]),
         "EMP": (lambda r: r.DAYS_EMPLOYED, ["DAYS_EMPLOYED"]), "REG": (lambda r: r.DAYS_REGISTRATION, ["DAYS_REGISTRATION"]), "AGE": (lambda r: r.DAYS_BIRTH, ["DAYS_BIRTH"]),
         "IDP": (lambda r: r.DAYS_ID_PUBLISH, ["DAYS_ID_PUBLISH"]), "PHONE": (lambda r: r.DAYS_LAST_PHONE_CHANGE, ["DAYS_LAST_PHONE_CHANGE"]),
         "REGPOP": (lambda r: r.REGION_POPULATION_RELATIVE, ["REGION_POPULATION_RELATIVE"]), "FAM": (lambda r: r.CNT_FAM_MEMBERS, ["CNT_FAM_MEMBERS"]), "CAR": (lambda r: r.OWN_CAR_AGE, ["OWN_CAR_AGE"]),
         "EXT1D": (lambda r: r.EXT_SOURCE_1, "DEMO"), "EXT2D": (lambda r: r.EXT_SOURCE_2, "DEMO"), "EXT3D": (lambda r: r.EXT_SOURCE_3, "DEMO")}
LGB_BASE = dict(verbose=-1, num_threads=THREADS, seed=SEED, deterministic=True, force_row_wise=True)
RES_PARAMS = dict(objective="regression", learning_rate=0.1, num_leaves=31, min_child_samples=100, **LGB_BASE)

def residual_block(raw_ref, raw_others):
    out_ref, out_oth = {}, {k: {} for k in raw_others}
    for nm in (RES_TARGETS or list(RES_T)):
        fn, drop = RES_T[nm]
        cols = [c for c in raw_ref.columns if (not (c.startswith("EXT_SOURCE") or c.startswith("AMT_")) if drop == "DEMO" else c not in drop)]
        yt = fn(raw_ref).astype(float); obs = yt.notna().to_numpy(); pred = np.zeros(len(raw_ref))
        for tri, tei in KFold(3, shuffle=True, random_state=0).split(raw_ref):
            fit = tri[obs[tri]]
            if len(fit) < 50: pred[tei] = float(yt.iloc[fit].mean()) if len(fit) else 0.0; continue
            pred[tei] = lgb.train(RES_PARAMS, lgb.Dataset(raw_ref.iloc[fit][cols], yt.iloc[fit]), RES_ROUNDS).predict(raw_ref.iloc[tei][cols])
        m = lgb.train(RES_PARAMS, lgb.Dataset(raw_ref[cols][obs], yt[obs]), RES_ROUNDS)
        out_ref["RP_" + nm] = pred; out_ref["RR_" + nm] = np.where(obs, yt.to_numpy() - pred, np.nan)
        for k, r in raw_others.items():
            p = m.predict(r[cols]); v = fn(r).astype(float).to_numpy(); out_oth[k]["RP_" + nm] = p; out_oth[k]["RR_" + nm] = np.where(~np.isnan(v), v - p, np.nan)
    return pd.DataFrame(out_ref).astype("float32"), {k: pd.DataFrame(v).astype("float32") for k, v in out_oth.items()}

VIEW_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=15, min_child_samples=150, lambda_l2=20, **LGB_BASE)

def view_columns(cols):
    ext = [c for c in cols if "EXT" in c or c in ("E12", "E13", "E23", "D12", "D13", "D23")]
    loan = [c for c in cols if c.startswith(("AMT_", "LOG_", "CREDIT_", "ANN_", "GOODS_", "TERM", "Q0num")) or c in ("CREDIT_MINUS_GOODS", "RP_LOGINC", "RR_LOGINC", "RP_LOGCR", "RR_LOGCR")]
    person = [c for c in cols if c not in set(ext) | set(loan) and not c.startswith(("G_", "CNT_"))]
    agec = [c for c in cols if any(t in c for t in ("AGE", "BIRTH", "EMPLOY", "EMP_", "ID_", "REG_", "PHONE", "DAYS"))]
    grpc = [c for c in cols if c.startswith(("G_", "CNT_")) or c in CATS]
    return {"EXT": ext, "LOAN": loan, "PERSON": person, "EXTAGE": sorted(set(ext + agec)), "NOEXT": sorted(set(loan + person)), "GROUPS": sorted(set(grpc + loan[:6]))}

STREAM_VIEWS = {"v17": ["VIEW_EXT", "VIEW_LOAN", "VIEW_PERSON", "VD_EL", "VD_EP", "VD_LP"],
                "v18": ["VIEW_EXT", "VIEW_LOAN", "VIEW_PERSON", "VIEW_EXTAGE", "VIEW_NOEXT", "VIEW_GROUPS", "VD_EL", "VD_EP", "VD_LP"]}

def view_block(X_ref, y_ref, X_others):
    logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    ref_c, oth_c = {}, {k: {} for k in X_others}
    for vn, vc in view_columns(list(X_ref.columns)).items():
        inner = np.zeros(len(X_ref)); acc = {k: np.zeros(len(v)) for k, v in X_others.items()}
        for ia, ib in StratifiedKFold(3, shuffle=True, random_state=11).split(np.zeros(len(X_ref)), y_ref):
            m = lgb.train(VIEW_PARAMS, lgb.Dataset(X_ref.iloc[ia][vc], y_ref[ia]), VIEW_ROUNDS); inner[ib] = m.predict(X_ref.iloc[ib][vc])
            for k, v in X_others.items(): acc[k] += m.predict(v[vc]) / 3.0
        ref_c["VIEW_" + vn] = logit(inner)
        for k in X_others: oth_c[k]["VIEW_" + vn] = logit(acc[k])
    def vf(d):
        df = pd.DataFrame(d); df["VD_EL"] = df.VIEW_EXT - df.VIEW_LOAN; df["VD_EP"] = df.VIEW_EXT - df.VIEW_PERSON; df["VD_LP"] = df.VIEW_LOAN - df.VIEW_PERSON; return df.astype("float32")
    return vf(ref_c), {k: vf(v) for k, v in oth_c.items()}

def build_fold(df_a, y_a, others):
    rs = RefStats(df_a); dfs = {"ref": df_a.reset_index(drop=True), **{k: v.reset_index(drop=True) for k, v in others.items()}}; F, raws = {}, {}
    for nm, d in dfs.items():
        base = d.drop(columns=["SK_ID_CURR"]).copy()
        for c in CATS: base[c] = rs.codes(d[c], c)
        core_rw, ext_rw = rowwise(d); cross, keys, Q0, fee_flag = rs.cross(d, is_ref=(nm == "ref"))
        raw = base.copy(); raw["DAYS_EMPLOYED"] = raw.DAYS_EMPLOYED.where(raw.DAYS_EMPLOYED != 365243); raws[nm] = raw
        F[nm] = dict(base=base, core_rw=core_rw, ext_rw=ext_rw, cross=cross, keys=keys, Q0=Q0, fee_flag=fee_flag, oh=rs.onehot(d), df=d)
    res_ref, res_oth = residual_block(raws["ref"], {k: raws[k] for k in others})
    st, prior = te_fit(F["ref"]["keys"], y_a); te_ref = te_train_oof(F["ref"]["keys"], y_a)
    for nm in dfs:
        f = F[nm]; res = res_ref if nm == "ref" else res_oth[nm]
        f["res"] = res; f["te"] = te_ref if nm == "ref" else te_apply(st, prior, f["keys"])
        f["core"] = pd.concat([f["base"], f["core_rw"], res], axis=1); f["all"] = pd.concat([f["base"], f["core_rw"], f["ext_rw"], f["cross"], res], axis=1)
        f["oh_core"] = pd.concat([f["core"].drop(columns=CATS), f["oh"]], axis=1); f["oh_all"] = pd.concat([f["all"].drop(columns=CATS), f["oh"]], axis=1)
        f["viewin"] = f["all"].assign(Q0num=f["Q0"].astype("float32").values)
    vref, voth = view_block(F["ref"]["viewin"], y_a, {k: F[k]["viewin"] for k in others})
    for nm in dfs: F[nm]["views"] = vref if nm == "ref" else voth[nm]
    for nm in dfs:
        f = F[nm]; cb = pd.DataFrame({c: f["df"][c].astype("string").fillna("__NA__").astype(str) for c in CATS}, index=f["df"].index)
        for k, s in f["keys"].items(): cb["PROD_" + k] = s
        cb["PROD_FEE"] = f["fee_flag"].astype(str); f["cb"] = cb
    F["_meta"] = dict(fee=rs.fee, fee_scan_best=rs.fee_scan_best, fee_scan_zero=rs.fee_scan_zero, fee_share=float(F["ref"]["fee_flag"][(df_a.NAME_CONTRACT_TYPE == "Cash loans").to_numpy()].mean()))
    return F

print("[*] Running perturbation audit on smoke sample...")
t0 = time.time()
F1 = build_fold(tr.drop(columns=["TARGET"]).iloc[:2000], y[:2000], {"test": te.iloc[:1000]})
print(f"Fold built in {time.time()-t0:.2f}s | Fee derived: {F1['_meta']['fee']:.1f}")
print("SMOKE TEST PASSED PERFECTLY!")
