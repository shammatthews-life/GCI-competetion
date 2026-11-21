# ==================================================
# CODE CELL 1
# ==================================================

# Colab setup (lightgbm, catboost, optuna; torch is preinstalled on Colab)
import subprocess, sys
for p_ in ("lightgbm", "catboost", "optuna", "psutil"):
    try: __import__(p_)
    except ImportError: subprocess.run([sys.executable, "-m", "pip", "-q", "install", p_])

# ==================================================
# CODE CELL 2
# ==================================================

# cell 1 - setup (same input discovery as v15: finds train/test/sample in /content, Drive or cwd)
import os, sys, glob, time, warnings; warnings.filterwarnings("ignore")
try:
    from google.colab import drive; drive.mount("/content/drive")
except Exception as e: print("Drive not mounted (ok if files are in /content):", e)
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
# ---- v16 switches ----
SETS = ["old", "e1", "e1et", "e1Aet"]      # LightGBM sets to run. Drop some (e.g. ["e1et","e1Aet"]) for a faster ~4h run; the final formula adapts.
REUSE_RESID = True                         # re-use v15's label-free residual file if present. If the shift diagnostic (cell 3b) flags RP_/RR_ columns, set False and recompute
REUSE_VIEWS = False                        # v18: views are RECOMPUTED on the enriched v16+ feature set (v15's cached views know nothing about partner/person/product features)
EXTRA_VIEWS = True                         # v18: 3 extra views (EXT+age, everything-but-EXT, groups/categoricals). Evidence: 1-fold proxy +0.0019 -> validate with cell 8-style acceptance rule
MONO_TE = True                             # v18: target-encoded + partner-label features monotone (+1) in the v2_mono member only (5-fold HGB proxy: +0.0003 mean)
RUN_TUNING = os.environ.get("RUN_TUNING") == "1"   # set True in Colab to run the optional Optuna search (cell 12)
TUNE_TRIALS, TUNE_FOLDS, TUNE_SECONDS = 12, (0, 1), 3000
USE_PARTNER = True                         # repeat-applicant (partner) features
USE_PERSON = True                          # person-pooled features (EXT scores averaged over a person's applications, partner deltas); 5-fold test: mean +0.0005 AUC
USE_COMMUNITY_KEYS = True                  # rounded CREDIT/ANNUITY keys (1/2/3 dec), combos with goods/credit/annuity, 3 smoothing levels
SMOKE = os.environ.get("SMOKE") == "1"                   # SMOKE=1 -> tiny quick test of the code path only
def find(name, roots=("/content", "/content/drive/MyDrive", os.getcwd())):
    for r in roots:
        for pat in (name, "*/"+name, "*/*/"+name, "*/*/*/"+name, "*/*/*/*/"+name):
            hit = glob.glob(os.path.join(r, pat))
            if hit: return hit[0]
    return None
def find_data():   # validated finder: 34 columns, TARGET present, 171,202 rows (skips any other train.csv with the same name)
    cands = []
    for r in ("/content", "/content/drive/MyDrive", os.getcwd()):
        for pat in ("train.csv", "*/train.csv", "*/*/train.csv", "*/*/*/train.csv", "*/*/*/*/train.csv"): cands += glob.glob(os.path.join(r, pat))
    for f in sorted(set(cands)):
        d = os.path.dirname(f)
        if not all(os.path.exists(os.path.join(d, x)) for x in ("test.csv", "sample_submission.csv")): continue
        try: head = open(f, "rb").readline().decode("utf-8", "ignore").strip().split(","); nrows = sum(1 for _ in open(f, "rb")) - 1
        except Exception: continue
        if len(head) == 34 and "TARGET" in head and "AMT_ANNUITY" in head and nrows == 171202: print("using competition files in", d); return [os.path.join(d, x) for x in ("train.csv", "test.csv", "sample_submission.csv")]
        print("skipping a different train.csv:", f, "| columns", len(head), "| rows", nrows)
    raise SystemExit("Real competition files not found: upload train.csv, test.csv, sample_submission.csv from the competition input folder to /content")
TRAIN_PATH, TEST_PATH, SAMPLE_PATH = find_data()
assert TRAIN_PATH and TEST_PATH and SAMPLE_PATH, "Could not find train.csv / test.csv / sample_submission.csv - upload them to /content or put them in Drive"
HGB_PATH = find("GCI_hgb_product_test.npy"); HGB_OOF = find("GCI_hgb_product_oof.npy")
base_dir = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else "/content"
CACHE_DIR = os.path.join(base_dir, "gci_cache_v18") if not SMOKE else "/tmp/gci_cache_smoke"; OUT_DIR = os.path.join(base_dir, "gci_output_v18") if not SMOKE else "/tmp/gci_out_smoke"
os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)
THREADS = os.cpu_count() or 1
OLD_CACHE = os.path.join(base_dir, "gci_cache_final")
print("train:", TRAIN_PATH, "\ntest:", TEST_PATH, "\nsample:", SAMPLE_PATH, "\nhgb oof:", HGB_OOF, "\nhgb npy:", HGB_PATH, "\ncache:", CACHE_DIR, "| threads:", THREADS)

# ==================================================
# CODE CELL 3
# ==================================================

# cell 2 - OPTIMISED: features (v15 families) + decoded product keys (v15) + community keys + partner features (v16)
CATS = ["NAME_CONTRACT_TYPE","CODE_GENDER","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","OCCUPATION_TYPE","ORGANIZATION_TYPE"]
tr = pd.read_csv(TRAIN_PATH); te = pd.read_csv(TEST_PATH); sample = pd.read_csv(SAMPLE_PATH)
y = tr["TARGET"].values; ntr = len(tr)
al = pd.concat([tr.drop(columns=["TARGET"]), te], ignore_index=True)
def new_cols(d): return pd.DataFrame(d, index=al.index)
def base(al):
    X = al.drop(columns=["SK_ID_CURR"]).copy()
    for c in CATS: X[c] = X[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
    return X
def f_clean(al, X):
    d = {"DE_SENT": (al.DAYS_EMPLOYED == 365243).astype(int), "DAYS_EMPLOYED_CLEAN": al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243),
         "ORG_XNA": (al.ORGANIZATION_TYPE == "XNA").astype(int), "CAR_AGE_OUT": (al.OWN_CAR_AGE >= 60).astype(int), "MISS_CNT": al.isna().sum(axis=1)}
    for c in ["EXT_SOURCE_1","EXT_SOURCE_3","OWN_CAR_AGE","AMT_REQ_CREDIT_BUREAU_YEAR"]: d[c + "_NA"] = al[c].isna().astype(int)
    return pd.concat([X, new_cols(d)], axis=1)
def f_ext(al, X):
    e = al[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]]; d = {}
    d["EXT_MEAN"] = e.mean(axis=1); d["EXT_MIN"] = e.min(axis=1); d["EXT_MAX"] = e.max(axis=1); d["EXT_STD"] = e.std(axis=1)
    d["EXT_RANGE"] = d["EXT_MAX"] - d["EXT_MIN"]; d["EXT_NOBS"] = e.notna().sum(axis=1); d["EXT_PROD"] = e.prod(axis=1)
    d["EXT_W"] = (al.EXT_SOURCE_1*2 + al.EXT_SOURCE_2*3 + al.EXT_SOURCE_3*4) / 9
    d["E12"] = al.EXT_SOURCE_1*al.EXT_SOURCE_2; d["E13"] = al.EXT_SOURCE_1*al.EXT_SOURCE_3; d["E23"] = al.EXT_SOURCE_2*al.EXT_SOURCE_3
    d["D12"] = al.EXT_SOURCE_1-al.EXT_SOURCE_2; d["D13"] = al.EXT_SOURCE_1-al.EXT_SOURCE_3; d["D23"] = al.EXT_SOURCE_2-al.EXT_SOURCE_3
    return pd.concat([X, new_cols(d)], axis=1)
def f_fin(al, X):
    inc, cr, an, gp = al.AMT_INCOME_TOTAL, al.AMT_CREDIT, al.AMT_ANNUITY, al.AMT_GOODS_PRICE
    d = {"CREDIT_INC": cr/inc, "ANN_INC": an/inc, "ANN_CREDIT": an/cr, "TERM": cr/an, "GOODS_CREDIT": gp/cr, "CREDIT_MINUS_GOODS": cr-gp, "GOODS_INC": gp/inc,
         "INC_PER_FAM": inc/al.CNT_FAM_MEMBERS, "CREDIT_PER_FAM": cr/al.CNT_FAM_MEMBERS, "ANN_PER_FAM": an/al.CNT_FAM_MEMBERS}
    for c in ["AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY","AMT_GOODS_PRICE"]: d["LOG_" + c] = np.log1p(al[c])
    return pd.concat([X, new_cols(d)], axis=1)
def f_age(al, X):
    age = -al.DAYS_BIRTH; de = al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243)
    d = {"AGE": age/365.25, "EMP_AGE": de/al.DAYS_BIRTH, "EMP_YEARS": -de/365.25, "REG_AGE": al.DAYS_REGISTRATION/al.DAYS_BIRTH, "ID_AGE": al.DAYS_ID_PUBLISH/al.DAYS_BIRTH,
         "PHONE_AGE": al.DAYS_LAST_PHONE_CHANGE/al.DAYS_BIRTH, "CAR_AGE_RATIO": al.OWN_CAR_AGE/(age/365.25), "CREDIT_PER_AGE": al.AMT_CREDIT/(age/365.25), "INC_PER_AGE": al.AMT_INCOME_TOTAL/(age/365.25)}
    return pd.concat([X, new_cols(d)], axis=1)
def f_bureau(al, X):
    b = [c for c in al.columns if c.startswith("AMT_REQ")]
    d = {"BUR_SUM": al[b].sum(axis=1, min_count=1), "BUR_RECENT": al[["AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_MON"]].sum(axis=1, min_count=1),
         "BUR_MON_YEAR": al.AMT_REQ_CREDIT_BUREAU_MON/(al.AMT_REQ_CREDIT_BUREAU_YEAR+1)}
    return pd.concat([X, new_cols(d)], axis=1)
def f_count(al, X):
    d = {"CNT_" + c: al[c].map(al[c].value_counts()) for c in CATS + ["REGION_POPULATION_RELATIVE","DAYS_BIRTH","AMT_CREDIT","AMT_GOODS_PRICE","AMT_ANNUITY","AMT_INCOME_TOTAL"]}
    return pd.concat([X, new_cols(d)], axis=1)
def f_group(al, X):
    tmp = al.copy(); tmp["CREDIT_INC"] = al.AMT_CREDIT/al.AMT_INCOME_TOTAL; d = {}
    for k in ["ORGANIZATION_TYPE","OCCUPATION_TYPE","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","REGION_POPULATION_RELATIVE"]:
        for v in ["AMT_INCOME_TOTAL","AMT_CREDIT","EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","DAYS_BIRTH","CREDIT_INC"]:
            d[f"G_{k[:6]}_{v[-6:]}_DIFF"] = tmp[v] - tmp.groupby(k)[v].transform("mean")
    return pd.concat([X, new_cols(d)], axis=1)
def f_kdec(al, X):    # credit = goods * (1 + 0.0132*k) for ~90% of cash loans with credit > goods -> integer k (insurance/tariff tier); NaN where another tariff family applies
    C, G = al.AMT_CREDIT, al.AMT_GOODS_PRICE; cash = al.NAME_CONTRACT_TYPE == "Cash loans"; kf = (C / G - 1) / 0.0132; k = kf.round(); ok = cash & (C > G) & ((kf - k).abs() < 0.1)
    K = pd.Series(np.where(ok, k, np.where(cash & (C == G), 0.0, np.nan)), index=al.index)
    return pd.concat([X, new_cols({"K_DEC": K, "CG_RATIO": C / G})], axis=1)
# ---- PARTNER / REPEAT-APPLICANT FEATURES (new in v16) ----
# same person re-applying keeps constant gaps: DAYS_REGISTRATION-DAYS_BIRTH and DAYS_ID_PUBLISH-DAYS_BIRTH. Two fingerprints: strict (+ education, region, mobile flag) and loose (gender + gaps).
_i1 = al.DAYS_REGISTRATION - al.DAYS_BIRTH; _i2 = al.DAYS_ID_PUBLISH - al.DAYS_BIRTH
def _groups(parts):
    k = al[parts].astype(str).agg("|".join, axis=1) + "|" + _i1.astype(str) + "|" + _i2.astype(str); k = k.where(al.DAYS_REGISTRATION != 0)
    sz = k.map(k.value_counts()); k = k.where(sz > 1); g = pd.Series(pd.factorize(k)[0]).replace(-1, np.nan).values
    mem = pd.DataFrame({"g": g}).groupby("g").groups; return g, sz.where(k.notna()).values, mem
_gb = al.DAYS_BIRTH.values
def _plist(g, mem):         # for every row: (indices of the other applications, signed gap: >0 means the partner applied EARLIER)
    out = {}
    for _, ix in mem.items():
        ix = np.asarray(ix)
        for i in ix: o = ix[ix != i]; out[i] = (o, _gb[o] - _gb[i])
    return out
G_S, SZ_S, MEM_S = _groups(["CODE_GENDER", "NAME_EDUCATION_TYPE", "REGION_POPULATION_RELATIVE", "FLAG_MOBIL"]); PL_S = _plist(G_S, MEM_S)
G_L, SZ_L, MEM_L = _groups(["CODE_GENDER"]); PL_L = _plist(G_L, MEM_L)
def f_partner(al, X):
    gap = np.full(len(al), np.nan)
    for i, (o, gp_) in PL_S.items(): gap[i] = gp_[np.argmin(np.abs(gp_))]
    d = {"HAS_PARTNER": (~np.isnan(G_S)).astype(int), "PARTNER_N": SZ_S, "PARTNER_GAP": gap, "PARTNER_ABSGAP": np.abs(gap), "HAS_PARTNER_LOOSE": (~np.isnan(G_L)).astype(int), "PARTNER_N_LOOSE": SZ_L}
    return pd.concat([X, new_cols(d)], axis=1)
def f_person(al, X):      # label-free PERSON-level pooling: repeat applications are noisy re-measurements of the same person (EXT corr with partner 0.72-0.83) -> pool them
    N = len(al); E = al[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].values.astype(float); pe = np.full((N, 3), np.nan); dif = np.full((N, 3), np.nan)
    cr = np.full(N, np.nan); an = np.full(N, np.nan); inc = np.full(N, np.nan); same = np.full(N, np.nan); qd = np.full(N, np.nan)
    Qv = (al.AMT_ANNUITY / al.AMT_CREDIT).values; ct = (al.NAME_CONTRACT_TYPE == "Cash loans").values; CR, AN_, INC = al.AMT_CREDIT.values, al.AMT_ANNUITY.values, al.AMT_INCOME_TOTAL.values
    for i, (o, g) in PL_S.items():
        j = o[np.argmin(np.abs(g))]; allr = np.r_[i, o]
        for k in range(3):
            v = E[allr, k]; pe[i, k] = np.nanmean(v) if np.isfinite(v).any() else np.nan; dif[i, k] = E[i, k] - E[j, k]
        cr[i] = CR[i] / CR[j]; an[i] = AN_[i] / AN_[j]; inc[i] = INC[i] / INC[j]; same[i] = float(ct[i] == ct[j]); qd[i] = Qv[i] - Qv[j]
    d = {f"PERS_EXT{k+1}": pe[:, k] for k in range(3)}; d.update({f"PDIF_EXT{k+1}": dif[:, k] for k in range(3)}); d.update(PCR_RATIO=cr, PAN_RATIO=an, PINC_RATIO=inc, PSAME_CT=same, PQ_DIFF=qd)
    P = pd.DataFrame(d); P["PERS_EXT_MEAN"] = P[["PERS_EXT1", "PERS_EXT2", "PERS_EXT3"]].mean(axis=1)
    return pd.concat([X, new_cols({c: P[c].values for c in P.columns})], axis=1)
N_PARTNER_ENC = 6
def partner_enc(fit, idx):     # label-based: uses ONLY labels of rows in `fit` (so for training rows it is leave-self-out and, inside train_enc, nested)
    idx = np.asarray(idx); cols = ["PY_ANY", "PY_N", "PY_NEAR180", "PY_EARLIER", "PY_LATER", "PYL_ANY"]
    out = np.full((len(idx), 6), np.nan); out[:, 1] = 0
    if SMOKE or not USE_PARTNER: return pd.DataFrame(out, columns=cols)
    lab = np.full(len(al), np.nan); lab[fit] = y[fit]
    for j, i in enumerate(idx):
        if i in PL_S:
            o, gp_ = PL_S[i]; v = lab[o]; k = ~np.isnan(v)
            if k.any():
                v, gp_ = v[k], gp_[k]; out[j, 0] = v.max(); out[j, 1] = k.sum()
                if (np.abs(gp_) <= 180).any(): out[j, 2] = v[np.abs(gp_) <= 180].max()
                if (gp_ > 0).any(): out[j, 3] = v[gp_ > 0].max()
                if (gp_ < 0).any(): out[j, 4] = v[gp_ < 0].max()
        if i in PL_L:
            o, gp_ = PL_L[i]; v = lab[o]; v = v[~np.isnan(v)]
            if len(v): out[j, 5] = v.max()
    return pd.DataFrame(out, columns=cols)
FAMS = {"person": f_person, "partner": f_partner, "kdec": f_kdec, "clean": f_clean, "ext": f_ext, "fin": f_fin, "age": f_age, "bureau": f_bureau, "count": f_count, "group": f_group}
def build(fams):
    X = base(al)
    for f in fams: X = FAMS[f](al, X)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
CORE = ["clean", "fin", "age", "bureau"] + (["partner"] if USE_PARTNER else []) + (["person"] if (USE_PARTNER and USE_PERSON) else []); ALLF = ["clean", "ext", "fin", "age", "bureau", "count", "group"] + (["partner"] if USE_PARTNER else []) + (["person"] if (USE_PARTNER and USE_PERSON) else [])
Xc, Xct = build(CORE); Xa, Xat = build(ALLF)                              # members WITHOUT k
Xck, Xckt = build(CORE + ["kdec"]); Xak, Xakt = build(ALLF + ["kdec"])   # members WITH decoded k (v2 members)
# keep the frames WITHOUT residual features; residuals are attached per experiment
BASEF = dict(Xc=Xc, Xct=Xct, Xa=Xa, Xat=Xat, Xck=Xck, Xckt=Xckt, Xak=Xak, Xakt=Xakt)
if SMOKE: y = y[:8000]
n, nt = len(y), len(Xct); tidx = np.arange(ntr, ntr + nt) if not SMOKE else np.arange(8000, 8000 + nt)
# ---- q, and the decoded product: annuity = Q0*credit + F  (F ~ 129.5 monthly fee on ~75% of cash loans); label-free ----
q = al.AMT_ANNUITY / al.AMT_CREDIT; _cash = (al.NAME_CONTRACT_TYPE == "Cash loans").values; _qb = (al.AMT_ANNUITY - 129.5) / al.AMT_CREDIT
_q5, _qb5 = q.round(5), _qb.round(5)
FEE = ((_qb5.map(_qb5.value_counts()).values > _q5.map(_q5.value_counts()).values) & _cash).astype(int); Q0 = pd.Series(np.where(FEE, _qb, q))
KS = {"Q3": q.round(3).astype(str), "Q4": q.round(4).astype(str), "Q5": q.round(5).astype(str),
      "P3": Q0.round(3).astype(str), "P4": Q0.round(4).astype(str), "P5": Q0.round(5).astype(str), "PF4": Q0.round(4).astype(str) + "|" + pd.Series(FEE).astype(str)}
M0 = {"Q3": 100., "Q4": 30., "Q5": 30., "P3": 100., "P4": 30., "P5": 30., "PF4": 30.}
# ---- COMMUNITY KEYS (research: rounded CREDIT/ANNUITY = interest-rate tiers; 1/2/3 decimals; combos; counts; smoothing 10/50/200) ----
if USE_COMMUNITY_KEYS:
    CT = al.AMT_CREDIT / al.AMT_ANNUITY            # CREDIT_TERM ~ number of payments
    CT1, CT2, CT3 = CT.round(1).astype(str), CT.round(2).astype(str), CT.round(3).astype(str)
    KS.update({"CT1": CT1, "CT2": CT2, "CT3": CT3,
               "CT1_s10": CT1, "CT1_s200": CT1,                                  # same key, different smoothing (community tried 10 and 200)
               "CT1_GP": CT1 + "|" + al.AMT_GOODS_PRICE.astype(str), "CT1_CR": CT1 + "|" + al.AMT_CREDIT.astype(str), "CT1_AN": CT1 + "|" + al.AMT_ANNUITY.astype(str),
               "GP": al.AMT_GOODS_PRICE.astype(str), "CR": al.AMT_CREDIT.astype(str), "AN": al.AMT_ANNUITY.astype(str)})
    M0.update({"CT1": 50., "CT2": 50., "CT3": 50., "CT1_s10": 10., "CT1_s200": 200., "CT1_GP": 50., "CT1_CR": 50., "CT1_AN": 50., "GP": 50., "CR": 50., "AN": 50.})
if SMOKE: KS = {k: v.iloc[list(range(8000)) + list(range(ntr, ntr + nt))].reset_index(drop=True) for k, v in KS.items()}
print("decoded products:", Q0.round(5).nunique(), "(q5 distinct:", q.round(5).nunique(), ") | fee share of cash loans:", round(float(FEE[_cash].mean()), 3), "| features core/all:", Xc.shape[1], Xa.shape[1])
import hashlib
_h = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
KC = {k: pd.factorize(v)[0] for k, v in KS.items()}; KN = {k: int(v.max()) + 1 for k, v in KC.items()}     # integer-coded keys: bincount encoder is ~8x faster than pandas groupby on strings and bit-identical
_MEMO = {}                                                                                              # every one of the ~20 members needs the SAME fold encodings: compute them once (key includes a hash of the labels, so shuffled-label tests never reuse real ones)
def enc(fit, qi, keys=None, partner=True):
    mk = ("enc", _h(y), _h(fit), _h(qi), partner) if keys is None else None
    if mk is not None and mk in _MEMO: return _MEMO[mk]
    pr = float(y[fit].mean()); yf = y[fit].astype(float); out = {}
    for k in (keys or KS):
        c = KC[k]; cnt_t = np.bincount(c[fit], minlength=KN[k]).astype(float); sm_t = np.bincount(c[fit], weights=yf, minlength=KN[k]); cq = c[qi]
        out[f"TE_{k}"] = (sm_t[cq] + M0[k] * pr) / (cnt_t[cq] + M0[k]); out[f"CNT_{k}"] = cnt_t[cq]
    res = pd.concat([pd.DataFrame(out), partner_enc(fit, qi)], axis=1) if (keys is None and partner) else pd.DataFrame(out)
    if mk is not None: _MEMO[mk] = res
    return res
def train_enc(a):     # nested out-of-fold encodings for training rows (a row's own label never enters its own feature)
    mk = ("tr", _h(y), _h(a))
    if mk in _MEMO: return _MEMO[mk]
    cols = enc(a[:10], a[:5]).columns; inner = np.zeros((len(a), len(cols)))
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=7).split(a, y[a]): inner[ib] = enc(a[ia], a[ib]).values
    _MEMO[mk] = pd.DataFrame(inner, columns=cols); return _MEMO[mk]
def onehot(Xa_, Xt_):
    z = pd.concat([Xa_, Xt_], ignore_index=True); oh = pd.get_dummies(z[CATS].astype("category"), dummy_na=True, dtype=np.float32)
    oh.columns = [f"OH_{i}" for i in range(oh.shape[1])]; Z = pd.concat([z.drop(columns=CATS), oh], axis=1)
    return Z.iloc[:len(Xa_)].reset_index(drop=True), Z.iloc[len(Xa_):].reset_index(drop=True)
Xc, Xa = BASEF["Xc"], BASEF["Xa"]   # for the print above
BASE = dict(objective="binary", force_col_wise=True, bagging_freq=1, verbose=-1, num_threads=THREADS, seed=42)
rk = lambda v: (rankdata(v) - 1) / (len(v) - 1)
FOLDS = 2 if SMOKE else 5
W = {"qf_shallow": 0.24, "qf_reg": 0.12, "v1_allfam_oh": 0.32, "v2_allfam_oh": 0.16, "v2_mono": 0.16}   # the fixed weights of the submitted blend

# ---- learned-expectation (residual) features; seeds = how many different fold splits are averaged (E2 uses 3) ----
from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC
from sklearn.model_selection import KFold, train_test_split
def compute_resid(seeds=(0,)):
    raw = al.drop(columns=["SK_ID_CURR"]).copy()
    for c in CATS: raw[c] = raw[c].astype("category").cat.codes.replace(-1, np.nan).astype(float)
    raw["DAYS_EMPLOYED"] = raw.DAYS_EMPLOYED.where(raw.DAYS_EMPLOYED != 365243)
    demo = [c for c in raw.columns if c.startswith("EXT_SOURCE") or c.startswith("AMT_")]
    T = {"EXT1": (raw.EXT_SOURCE_1, ["EXT_SOURCE_1"]), "EXT2": (raw.EXT_SOURCE_2, ["EXT_SOURCE_2"]), "EXT3": (raw.EXT_SOURCE_3, ["EXT_SOURCE_3"]),
         "LOGINC": (np.log(raw.AMT_INCOME_TOTAL), ["AMT_INCOME_TOTAL"]), "LOGCR": (np.log(raw.AMT_CREDIT), ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]),
         "EMP": (raw.DAYS_EMPLOYED, ["DAYS_EMPLOYED"]), "REG": (raw.DAYS_REGISTRATION, ["DAYS_REGISTRATION"]), "AGE": (raw.DAYS_BIRTH, ["DAYS_BIRTH"]),
         "IDP": (raw.DAYS_ID_PUBLISH, ["DAYS_ID_PUBLISH"]), "PHONE": (raw.DAYS_LAST_PHONE_CHANGE, ["DAYS_LAST_PHONE_CHANGE"]), "REGPOP": (raw.REGION_POPULATION_RELATIVE, ["REGION_POPULATION_RELATIVE"]),
         "FAM": (raw.CNT_FAM_MEMBERS, ["CNT_FAM_MEMBERS"]), "CAR": (raw.OWN_CAR_AGE, ["OWN_CAR_AGE"]),
         "EXT1D": (raw.EXT_SOURCE_1, demo), "EXT2D": (raw.EXT_SOURCE_2, demo), "EXT3D": (raw.EXT_SOURCE_3, demo)}
    if SMOKE: T = dict(list(T.items())[:2])
    out = {}; t0 = time.time()
    for nm, (yt, drop_cols) in T.items():
        Xp = raw.drop(columns=drop_cols); pred = np.zeros(len(raw)); obs = yt.notna().values
        for sd in seeds:
            for tri, tei in KFold(2 if SMOKE else 5, shuffle=True, random_state=sd).split(Xp):
                m = HR(max_iter=5 if SMOKE else 120, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=100, random_state=0).fit(Xp.iloc[tri][obs[tri]], yt.iloc[tri][obs[tri]]); pred[tei] += m.predict(Xp.iloc[tei]) / len(seeds)
        out[f"RP_{nm}"] = pred; out[f"RR_{nm}"] = np.where(obs, yt.values - pred, np.nan); print(f"  residual target {nm} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(out)
def get_resid(seeds, name):
    path = os.path.join(CACHE_DIR, name)
    old = next((p_ for p_ in [os.path.join(base_dir, "gci_cache_v6", "resid16.pkl"), os.path.join(OLD_CACHE, "resid16.pkl")] if os.path.exists(p_) and REUSE_RESID), "/nonexistent")
    if not SMOKE and os.path.exists(path): return pd.read_pickle(path)
    if not SMOKE and seeds == (0,) and os.path.exists(old): return pd.read_pickle(old)       # reuse the v6 cache (identical recipe)
    R = compute_resid(seeds)
    if not SMOKE: R.to_pickle(path)
    return R
def with_resid(R):
    R_tr, R_te = R.iloc[:ntr].reset_index(drop=True), R.iloc[ntr:].reset_index(drop=True)
    assert len(R) == ntr + len(BASEF["Xct"])
    return {k: pd.concat([BASEF[k], R_te if k.endswith("t") else R_tr], axis=1) for k in BASEF}
def build_members(f):
    Xc_, Xct_, Xa_, Xat_, Xck_, Xckt_, Xak_, Xakt_ = [f[k] for k in ("Xc", "Xct", "Xa", "Xat", "Xck", "Xckt", "Xak", "Xakt")]
    if SMOKE: Xc_, Xa_, Xck_, Xak_ = [d.iloc[:8000] for d in (Xc_, Xa_, Xck_, Xak_)]
    Xco, Xcot = onehot(Xc_, Xct_); Xao, Xaot = onehot(Xa_, Xat_); Xaok, Xaokt = onehot(Xak_, Xakt_)
    MONO_K = [-1 if c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") else 0 for c in Xck_.columns] + ([s_ for _ in KS for s_ in (1, 0)] + [1, 0, 1, 1, 1, 1] if MONO_TE else [0] * (2 * len(KS) + N_PARTNER_ENC))   # enc columns are TE_k, CNT_k per key, then PY_ANY, PY_N, PY_NEAR180, PY_EARLIER, PY_LATER, PYL_ANY
    return {
     "qf_shallow":   (Xco,  Xcot,  dict(learning_rate=0.05, num_leaves=7, max_depth=3, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20), 1300, 1430, [42, 7, 2024, 11, 99]),
     "qf_reg":       (Xco,  Xcot,  dict(learning_rate=0.03, num_leaves=11, min_child_samples=300, feature_fraction=0.4, bagging_fraction=0.8, lambda_l2=40), 1939, 2133, [42, 7, 2024, 11, 99]),
     "v1_allfam_oh": (Xao,  Xaot,  dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20), 881, 969, [42, 7, 2024]),
     "v2_allfam_oh": (Xaok, Xaokt, dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20), 881, 969, [42, 7, 2024]),
     "v2_mono":      (Xck_.reset_index(drop=True), Xckt_.reset_index(drop=True), dict(learning_rate=0.02, num_leaves=15, min_child_samples=150, feature_fraction=0.5, bagging_fraction=0.8, lambda_l2=20, monotone_constraints=MONO_K), 1571, 1729, [42, 7, 2024])}
def run_member(name, spec, tag, view=None):
    X, Xt, p, R, Rf, seeds = spec[name]; f = os.path.join(CACHE_DIR, f"{tag}_{name}.npz")
    if view is not None and "monotone_constraints" in p: p = {**p, "monotone_constraints": list(p["monotone_constraints"]) + [0] * view["tr"][0].shape[1]}   # extra view columns are unconstrained
    if os.path.exists(f) and not SMOKE:
        z = np.load(f); print(f"{tag}_{name}: loaded from cache | held-out AUC {roc_auc_score(y, z['oof']):.5f}"); return z["test"], z["oof"]
    tf = np.zeros(nt); oof = np.zeros(n); t0 = time.time(); ex = lambda key, k=None: ([] if view is None else [view[key] if k is None else view[key][k]])
    for k, (a, b) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(n), y)):
        fa = pd.concat([X.iloc[a].reset_index(drop=True), train_enc(a)] + ex("tr", k), axis=1); fb = pd.concat([X.iloc[b].reset_index(drop=True), enc(a, b)] + ex("va", k), axis=1); ft = pd.concat([Xt.reset_index(drop=True), enc(a, tidx)] + ex("te", k), axis=1)
        assert "monotone_constraints" not in p or len(p["monotone_constraints"]) == fa.shape[1], f"monotone constraints ({len(p['monotone_constraints'])}) != features ({fa.shape[1]})"
        m = lgb.train({**BASE, **p}, lgb.Dataset(fa, y[a]), 60 if SMOKE else R); oof[b] = m.predict(fb); tf += m.predict(ft) / FOLDS
        print(f"  {tag}_{name} fold {k+1}/{FOLDS}: AUC {roc_auc_score(y[b], oof[b]):.5f} ({time.time()-t0:.0f}s)", flush=True)
    allr = np.arange(n); fa = pd.concat([X.reset_index(drop=True), train_enc(allr)] + ex("all_tr"), axis=1); ft = pd.concat([Xt.reset_index(drop=True), enc(allr, tidx)] + ex("all_te"), axis=1); fr = []
    for sd in ([42] if SMOKE else seeds):
        fr.append(rk(lgb.train({**BASE, **p, "seed": sd}, lgb.Dataset(fa, y), 60 if SMOKE else Rf).predict(ft))); print(f"  {tag}_{name} full-data seed {sd} done ({time.time()-t0:.0f}s)", flush=True)
    test = rk(0.5 * rk(tf) + 0.5 * np.mean(fr, 0)); print(f"{tag}_{name}: held-out AUC {roc_auc_score(y, oof):.5f}", flush=True)
    if not SMOKE: np.savez(f, test=test, oof=oof)
    return test, oof
RESV = {}
def nested_generic(a, fn):      # nested out-of-fold wrapper for any label-based encoder fn(fit_idx, apply_idx) -> DataFrame (used by CatBoost cell and the ablation)
    parts = []
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=7).split(a, y[a]):
        f = fn(a[ia], a[ib]); f.index = ib; parts.append(f)
    return pd.concat(parts).sort_index().reset_index(drop=True)
print("keys:", list(KS), "| partner rows (strict):", int((~np.isnan(G_S)).sum()), "| loose:", int((~np.isnan(G_L)).sum()))


# ==================================================
# CODE CELL 4
# ==================================================

# cell 3 - residual features (v6 recipe, one fold seed) and the baseline member frames
RESID1 = get_resid((0,), "resid16.pkl"); F1 = with_resid(RESID1); SPEC1 = build_members(F1)
print("v6-recipe features per member family (core / all-families):", F1["Xc"].shape[1], F1["Xa"].shape[1])

# ==================================================
# CODE CELL 5
# ==================================================

# cell 4 - multi-view stacking features: ORIGINAL variant (used by E1 and E1ET) and CONSISTENT variant A (used by E1Aet)
def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def vf(d):
    df = pd.DataFrame({f"VIEW_{k}": v for k, v in d.items()}); df["VD_EL"] = df.VIEW_EXT - df.VIEW_LOAN; df["VD_EP"] = df.VIEW_EXT - df.VIEW_PERSON; df["VD_LP"] = df.VIEW_LOAN - df.VIEW_PERSON; return df
def compute_views(variant):
    path = os.path.join(CACHE_DIR, ("e1Ax_views.pkl" if EXTRA_VIEWS else "e1A_views.pkl") if variant == "A" else ("e1x_views.pkl" if EXTRA_VIEWS else "e1_views.pkl"))
    if os.path.exists(path) and not SMOKE: return pd.read_pickle(path)
    oldp = os.path.join(OLD_CACHE, os.path.basename(path))
    if REUSE_VIEWS and os.path.exists(oldp) and not SMOKE: print("re-using v15 view features:", oldp); return pd.read_pickle(oldp)
    Xs = F1["Xa"].iloc[:n].copy(); Xts = F1["Xat"].copy(); Xs["Q0num"] = Q0.values[:n]; Xts["Q0num"] = Q0.values[ntr:]
    cols = list(Xs.columns)
    ext = [c for c in cols if "EXT" in c or c in ("E12", "E13", "E23", "D12", "D13", "D23")]
    loan = [c for c in cols if c.startswith(("AMT_", "LOG_", "CREDIT_", "ANN_", "GOODS_", "TERM", "Q0num")) or c in ("CREDIT_MINUS_GOODS", "RP_LOGINC", "RR_LOGINC", "RP_LOGCR", "RR_LOGCR")]
    person = [c for c in cols if c not in set(ext) | set(loan) and not c.startswith(("G_", "CNT_"))]
    views = {"EXT": ext, "LOAN": loan, "PERSON": person}
    if EXTRA_VIEWS:      # v18 extra views: different feature subsets = different inductive bias; the final GBDT sees their out-of-fold logits
        agec = [c for c in cols if any(t in c for t in ("AGE", "BIRTH", "EMPLOY", "EMP_", "ID_", "REG_", "PHONE", "DAYS"))]; grpc = [c for c in cols if c.startswith(("G_", "CNT_")) or c in CATS]
        views.update({"EXTAGE": sorted(set(ext + agec)), "NOEXT": sorted(set(loan + person)), "GROUPS": sorted(set(grpc + loan[:6]))})
    print(f"view variant {variant}: sizes", {k: len(v) for k, v in views.items()})
    mk = lambda: HC(max_iter=5 if SMOKE else 100, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=20, random_state=0)
    out = {"tr": [], "va": [], "te": []}; t0 = time.time()
    for k, (a, b) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(n), y)):
        trv, vav, tev = {}, {}, {}
        for vn, vc in views.items():
            inner = np.zeros(len(a)); pvb = []; pte = []
            for ia, ib in StratifiedKFold(3, shuffle=True, random_state=11).split(a, y[a]):
                m = mk().fit(Xs.iloc[a[ia]][vc], y[a[ia]]); inner[ib] = m.predict_proba(Xs.iloc[a[ib]][vc])[:, 1]
                if variant == "A": pvb.append(m.predict_proba(Xs.iloc[b][vc])[:, 1]); pte.append(m.predict_proba(Xts[vc])[:, 1])
            trv[vn] = lg(inner)
            if variant == "A": vav[vn] = lg(np.mean(pvb, 0)); tev[vn] = lg(np.mean(pte, 0))
            else: m = mk().fit(Xs.iloc[a][vc], y[a]); vav[vn] = lg(m.predict_proba(Xs.iloc[b][vc])[:, 1]); tev[vn] = lg(m.predict_proba(Xts[vc])[:, 1])
        out["tr"].append(vf(trv)); out["va"].append(vf(vav)); out["te"].append(vf(tev)); print(f"  views fold {k+1}/{FOLDS} ({time.time()-t0:.0f}s)", flush=True)
    trv, tev = {}, {}
    for vn, vc in views.items():
        inner = np.zeros(n); pte = []
        for ia, ib in StratifiedKFold(3, shuffle=True, random_state=11).split(np.zeros(n), y):
            m = mk().fit(Xs.iloc[ia][vc], y[ia]); inner[ib] = m.predict_proba(Xs.iloc[ib][vc])[:, 1]
            if variant == "A": pte.append(m.predict_proba(Xts[vc])[:, 1])
        trv[vn] = lg(inner); tev[vn] = lg(np.mean(pte, 0)) if variant == "A" else lg(mk().fit(Xs[vc], y).predict_proba(Xts[vc])[:, 1])
    out["all_tr"] = vf(trv); out["all_te"] = vf(tev)
    if not SMOKE: pd.to_pickle(out, path)
    return out
VIEW_O = compute_views("orig"); VIEW_A = compute_views("A")

# ==================================================
# CODE CELL 6
# ==================================================

# cell 5 - the four LightGBM sets (about 1.5-2 h each from scratch; cached)
SPEC0 = build_members(BASEF); SPEC1_ET = {nm: (X, Xt, {**p, "extra_trees": True}, R, Rf, seeds) for nm, (X, Xt, p, R, Rf, seeds) in SPEC1.items()}
if "old" in SETS:
  for nm in W: RESV[("old", nm)] = run_member(nm, SPEC0, "old")
if "e1" in SETS:
  for nm in W: RESV[("e1", nm)] = run_member(nm, SPEC1, "e1", VIEW_O)
if "e1et" in SETS:
  for nm in W: RESV[("e1et", nm)] = run_member(nm, SPEC1_ET, "e1et", VIEW_O)
if "e1Aet" in SETS:
  for nm in W: RESV[("e1Aet", nm)] = run_member(nm, SPEC1_ET, "e1Aet", VIEW_A)

# ==================================================
# CODE CELL 7
# ==================================================

# cell 6 - CatBoost with the decoded product keys as native categoricals (5 folds, about 50 min from scratch; cached)
from catboost import CatBoostClassifier
cb_numeric = pd.concat([BASEF["Xa"], BASEF["Xat"]], ignore_index=True).drop(columns=CATS).reset_index(drop=True)
cb_cat = al[CATS].copy().reset_index(drop=True)
for c in CATS: cb_cat[c] = cb_cat[c].astype("string").fillna("__MISSING__").astype(str)
cb_product = pd.concat([cb_numeric, cb_cat], axis=1); product_names = []
for k_ in ["Q3", "Q4", "Q5", "P3", "P4", "P5", "PF4"] + [k for k in ("CT1", "CT2", "CT3", "CT1_GP", "CT1_CR", "CT1_AN") if k in KS]:
    cb_product["PROD_" + k_] = KS[k_].reset_index(drop=True).astype("string").fillna("__MISSING__").astype(str) if not SMOKE else pd.Series(list(KS[k_].iloc[:n].astype(str)) + [""] * (len(cb_product) - n)).astype(str); product_names.append("PROD_" + k_)
kd = pd.concat([BASEF["Xak"]["K_DEC"], BASEF["Xakt"]["K_DEC"]], ignore_index=True)
cb_product["PROD_K_DEC"] = kd.round().astype("Int64").astype("string").fillna("__MISSING__").astype(str); product_names.append("PROD_K_DEC")
cb_product["PROD_FEE"] = pd.Series(FEE).reset_index(drop=True).astype(str); product_names.append("PROD_FEE")
CB_CATS = CATS + product_names; cb_train, cb_test = cb_product.iloc[:n].reset_index(drop=True), cb_product.iloc[ntr:].reset_index(drop=True)
CBP = dict(iterations=60 if SMOKE else 900, depth=7, learning_rate=0.05, loss_function="Logloss", eval_metric="AUC", l2_leaf_reg=8, random_seed=42, thread_count=THREADS, verbose=False,
           allow_writing_files=False, random_strength=0.5, bootstrap_type="Bernoulli", subsample=0.8)
folds_cb = list(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(n), y)); cb_oof = np.zeros(n); cb_tests = []; t0 = time.time()
for fi, (a, b) in enumerate(folds_cb, 1):
    fc = os.path.join(CACHE_DIR, f"catboost_v18_fold_{fi}.npz")
    if os.path.exists(fc) and not SMOKE: z = np.load(fc); pv, pt = z["oof"], z["test"]; print(f"CatBoost fold {fi}: loaded from cache")
    else:
        pe_a = nested_generic(a, partner_enc); pe_b = partner_enc(a, b); pe_t = partner_enc(a, tidx)      # label-based partner features, fold-safe
        ca = pd.concat([cb_train.iloc[a].reset_index(drop=True), pe_a], axis=1); cbv = pd.concat([cb_train.iloc[b].reset_index(drop=True), pe_b], axis=1); cbt = pd.concat([cb_test.reset_index(drop=True), pe_t], axis=1)
        m = CatBoostClassifier(**CBP); m.fit(ca, y[a], cat_features=CB_CATS); pv = m.predict_proba(cbv)[:, 1]; pt = m.predict_proba(cbt)[:, 1]
        if not SMOKE: np.savez_compressed(fc, oof=pv, test=pt)
    cb_oof[b] = pv; cb_tests.append(pt); print(f"CatBoost-product fold {fi}/{len(folds_cb)}: AUC {roc_auc_score(y[b], pv):.5f} ({time.time()-t0:.0f}s)", flush=True)
cb_test_mean = np.mean(cb_tests, 0)

# ==================================================
# CODE CELL 8
# ==================================================

# cell 7 - final v16 formula (v15 rank-blend recipe on the enriched members), held-out check and the submission file
blend = lambda tag, i: sum(W[k] * (RESV[(tag, k)][0] if i == 0 else rk(RESV[(tag, k)][1])) for k in W)      # i = 0 test ranks, 1 out-of-fold ranks
_have = [t for t in ("old", "e1", "e1et") if t in SETS]
B = lambda i: sum(blend(t, i) for t in _have) / len(_have)
champ = (lambda i, cb: 0.9 * rk(0.75 * rk(B(i)) + 0.25 * blend("e1Aet", i)) + 0.1 * rk(cb)) if ("e1Aet" in SETS and _have) else (lambda i, cb: 0.9 * rk(sum(blend(t, i) for t in SETS) / len(SETS)) + 0.1 * rk(cb))
oof_final, test_final = champ(1, cb_oof), champ(0, cb_test_mean)
print(f"held-out AUC of the champion formula (weights fixed in advance): {roc_auc_score(y, oof_final):.5f}   [v15 reference without partner/community keys: 0.77404]")
for t in SETS: print(f"   {t:6s} blend held-out AUC {roc_auc_score(y, blend(t, 1)):.5f}")
ids = sample.SK_ID_CURR; out = pd.DataFrame({"SK_ID_CURR": ids, "TARGET": test_final})
assert list(out.columns) == ["SK_ID_CURR", "TARGET"] and len(out) == 61500 and out.SK_ID_CURR.is_unique and np.isfinite(out.TARGET).all() and np.array_equal(out.SK_ID_CURR.values, ids.values)
if not SMOKE:
    from scipy.stats import spearmanr
    p = os.path.join(OUT_DIR, "GCI_submission_v18_reproduced.csv"); out.to_csv(p, index=False); print("wrote", p)
    orig = find("GCI_submission_v15_rankspace_consistent_view.csv")      # the file that was submitted (public 0.77891)
    if orig:
        o = pd.read_csv(orig); assert np.array_equal(o.SK_ID_CURR.values, ids.values)
        print("compared with the submitted v15 file:", orig, "| Spearman:", round(spearmanr(o.TARGET, out.TARGET).statistic, 6), "| max |difference|:", float(np.abs(o.TARGET.values - out.TARGET.values).max()))
    else: print("Upload GCI_submission_v15_rankspace_consistent_view.csv to /content to compare automatically.")
    try:
        pass
    except Exception: print("download manually from", OUT_DIR)
else: print("SMOKE run: file not written")

# ==================================================
# CODE CELL 9
# ==================================================

# ===== LAB CORE: memory hygiene, fold frames, evidence tools =====
import gc, json, psutil
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
LAB_DIR = os.path.join(base_dir, "gci_cache_v24"); os.makedirs(LAB_DIR, exist_ok=True)
RUN = dict(calibration=True, level2=True, catboost=True, tune=True, woe_lr=True, leaf_lr=True, extratrees=True, mlp=True)      # switch stages off here
BUDGET_MIN = dict(calibration=10, level2=10, catboost=90, tune=100, woe_lr=25, leaf_lr=40, extratrees=60, mlp=90)               # per-stage wall-clock cap (minutes); a stage stops cleanly when it is reached
ram = lambda: f"RAM used {psutil.Process().memory_info().rss/1e9:.1f} GB, free {psutil.virtual_memory().available/1e9:.1f} GB"
# ---- keep only what the lab needs, then free the big v18 frames (v21 crashed on RAM here) ----
XA_TR, XA_TE = [d.reset_index(drop=True) for d in SPEC1["v2_allfam_oh"][:2]]           # e1 member frame (all families + K_DEC), label-free
P_ET, R_ET, RF_ET, SEEDS_ET = SPEC1_ET["v2_allfam_oh"][2], SPEC1_ET["v2_allfam_oh"][3], SPEC1_ET["v2_allfam_oh"][4], SPEC1_ET["v2_allfam_oh"][5]
al_tr, al_te = al.iloc[:ntr], al.iloc[ntr:ntr + nt]
def segid(al_, xa_): return (al_.NAME_CONTRACT_TYPE.values == "Revolving loans").astype(int) * 8 + (al_.DAYS_EMPLOYED.values == 365243).astype(int) * 4 + np.clip(xa_["EXT_NOBS"].values.astype(int), 0, 3)
SEG_TR, SEG_TE = segid(al_tr, F1["Xa"])[:n], segid(al_te, F1["Xat"])
L2_COLS = [c for c in ("EXT_MEAN", "EXT_MIN", "EXT_NOBS", "HAS_PARTNER", "AGE", "CR_INC", "TERM") if c in F1["Xa"].columns]
L2_TR, L2_TE = F1["Xa"][L2_COLS].iloc[:n].reset_index(drop=True), F1["Xat"][L2_COLS].reset_index(drop=True)
for nm_ in ("SPEC0", "SPEC1", "SPEC1_ET", "F1", "BASEF", "cb_train", "cb_test", "cb_product", "cb_numeric", "cb_cat", "Zf"):
    if nm_ in globals(): del globals()[nm_]
gc.collect(); print("after cleanup:", ram())
FOLD_SPLITS = list(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(n), y))        # identical to the members' folds
def lab_frames(k, view=None):
    view = VIEW_O if view is None else view; a, b = FOLD_SPLITS[k]
    fa = pd.concat([XA_TR.iloc[a].reset_index(drop=True), train_enc(a), view["tr"][k]], axis=1); fb = pd.concat([XA_TR.iloc[b].reset_index(drop=True), enc(a, b), view["va"][k]], axis=1)
    ft = pd.concat([XA_TE, enc(a, tidx), view["te"][k]], axis=1)
    return a, b, fa.astype(np.float32), fb.astype(np.float32), ft[fa.columns].astype(np.float32)
ORANK = rk(oof_final); TRANK = rk(test_final)                                                      # champion (v18 recipe) in rank space
print(f"champion held-out AUC {roc_auc_score(y, ORANK):.5f}")
# ---- stage clock ----
STAGE = {"t_end": 1e18}
def begin_stage(name):
    STAGE["t_end"] = time.time() + (10 ** 6 if SMOKE else BUDGET_MIN[name] * 60); print(f"\n===== STAGE {name} (cap {BUDGET_MIN[name]} min) | {ram()}")
time_up = lambda: time.time() > STAGE["t_end"]
# ---- evidence tools ----
EVID, ACCEPTED = [], {}
def crossfit_blend(cand_rank, grid=np.linspace(0, 0.6, 13), seed=2026):
    pred = np.zeros(n); ws = []
    for fa_, fb_ in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(n), y):
        w = grid[int(np.argmax([roc_auc_score(y[fa_], (1 - w) * ORANK[fa_] + w * cand_rank[fa_]) for w in grid]))]; ws.append(float(w)); pred[fb_] = (1 - w) * ORANK[fb_] + w * cand_rank[fb_]
    return pred, ws
def paired_boot(p_new, p_base, B=200, seed=0):
    rs = np.random.RandomState(seed); d = []
    for _ in range(B): i = rs.randint(0, n, n); d.append(roc_auc_score(y[i], p_new[i]) - roc_auc_score(y[i], p_base[i]))
    d = np.array(d); return float(np.percentile(d, 5)), float(np.percentile(d, 95)), float((d <= 0).mean())
def judge(name, cand_oof, cand_test=None, note=""):
    cr = rk(cand_oof); pred, ws = crossfit_blend(cr); g = roc_auc_score(y, pred) - roc_auc_score(y, ORANK); lo, hi, p = paired_boot(pred, ORANK, B=20 if SMOKE else 200)
    fg = [roc_auc_score(y[b], pred[b]) - roc_auc_score(y[b], ORANK[b]) for _, b in FOLD_SPLITS]; nn_ = sum(x >= 0 for x in fg)
    ok = ((g >= 0.0002) and (lo > 0) and (nn_ >= min(4, len(fg)))) or (SMOKE and os.environ.get('LAB_FORCE') == '1')      # LAB_FORCE only exists to test the assembly code in SMOKE mode
    rec = dict(method=name, single_auc=round(float(roc_auc_score(y, cand_oof)), 5), corr_with_champion=round(float(np.corrcoef(cr, ORANK)[0, 1]), 4), crossfit_gain=round(float(g), 5), ci90=(round(lo, 5), round(hi, 5)), p_not_better=round(p, 3), folds_nonneg=f"{nn_}/{len(fg)}", weights_used=ws, decision="ACCEPT" if ok else "reject", note=note)
    EVID.append(rec); json.dump(EVID, open(os.path.join(LAB_DIR, "ledger.json"), "w"), indent=1, default=str); print("VERDICT", {k: v for k, v in rec.items() if k != "weights_used"})
    if ok and cand_test is not None: ACCEPTED[name] = (np.asarray(cand_oof), np.asarray(cand_test))
    return rec
def screen(name, pb0, floor=0.755, w=0.2, thr=0.0003):         # fold-1 screen: is this learner worth 5 folds of compute?
    a, b = FOLD_SPLITS[0]; base_b = ORANK[b]; s = roc_auc_score(y[b], pb0); g = roc_auc_score(y[b], (1 - w) * base_b + w * rk(pb0)) - roc_auc_score(y[b], base_b)
    ok = ((s >= floor) and (g >= thr)) or SMOKE; print(f"SCREEN {name}: fold-1 single AUC {s:.5f} | blend gain at w={w}: {g:+.5f} -> {'PASS' if ok else 'STOP (not worth the compute)'}"); return ok
def run_learner(name, fit_predict, screen_first=True, note=""):
    f0 = os.path.join(LAB_DIR, f"{name}.npz")
    if os.path.exists(f0) and not SMOKE: z = np.load(f0); print(f"{name}: loaded from cache"); return judge(name, z["oof"], z["test"], note)
    oof = np.zeros(n); tst = np.zeros(nt); t0 = time.time()
    for k in range(FOLDS):
        if time_up(): print(f"{name}: stage time cap reached, stopping"); return None
        a, b, fa, fb, ft = lab_frames(k); pb, pt = fit_predict(fa, y[a], fb, ft); oof[b] = pb; tst += rk(pt) / FOLDS
        print(f"  {name} fold {k+1}/{FOLDS}: AUC {roc_auc_score(y[b], pb):.5f} ({time.time()-t0:.0f}s) | {ram()}", flush=True)
        del fa, fb, ft; gc.collect()
        if k == 0 and screen_first and not screen(name, pb):
            EVID.append(dict(method=name, decision="stopped at screen", note=note)); return None
    if not SMOKE: np.savez(f0, oof=oof, test=tst)
    return judge(name, oof, tst, note)


# ==================================================
# CODE CELL 10
# ==================================================

# ===== candidate blocks (label-free ones are computed once on train+test; the label-based one is built per fold) =====
BUDGET_MIN["blocks"] = 150
import itertools
E3 = al[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]; em = E3.mean(axis=1); cr_, an_, inc_, gp_ = al.AMT_CREDIT, al.AMT_ANNUITY, al.AMT_INCOME_TOTAL, al.AMT_GOODS_PRICE; age_ = -al.DAYS_BIRTH / 365.25
def blk_e_poly():
    return pd.DataFrame({"P_EXTMEAN_PAY": em * an_ / cr_, "P_EXTMEAN_CTI": em * cr_ / inc_, "P_EXTMEAN_PAY_AGE": em * an_ / cr_ * age_, "P_CREDIT_GOODS_INC": (cr_ - gp_) / inc_, "P_EXTMEAN_ANN_INC": em * an_ / inc_})
def blk_ext_pct():
    agegrp = (age_ // 5).astype(int).astype(str); pct = lambda keys: em.groupby(keys).rank(pct=True)
    return pd.DataFrame({"EXTF_PCT_OCC": pct(al.OCCUPATION_TYPE.astype(str)), "EXTF_PCT_GEN_AGE": pct(al.CODE_GENDER.astype(str) + "|" + agegrp), "EXTF_PCT_ORG": pct(al.ORGANIZATION_TYPE.astype(str)), "EXTF_PCT_INCTYPE": pct(al.NAME_INCOME_TYPE.astype(str))})
def _date_diffs():
    dc = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]; D = al[dc].copy(); D["DAYS_EMPLOYED"] = D["DAYS_EMPLOYED"].where(D["DAYS_EMPLOYED"] != 365243)
    return pd.DataFrame({f"RDIFF_{a_[5:]}_{b_[5:]}": (D[a_] - D[b_]).values for a_, b_ in itertools.combinations(dc, 2)})
def _implied_peaks():
    cr = al.AMT_CREDIT.astype(float).values; an = al.AMT_ANNUITY.astype(float).values; q = an / cr; term = cr / an; d = {}
    for n_ in (12, 24, 36, 48, 60):
        lo, hi = np.full(len(q), 1e-9), np.full(len(q), 0.5)
        for _ in range(50):
            mid = (lo + hi) / 2; f_ = mid / (1 - (1 + mid) ** (-n_)) - q; hi = np.where(f_ > 0, mid, hi); lo = np.where(f_ > 0, lo, mid)
        d[f"R_IMPLIED_RATE_{n_}M"] = np.where(q > 1.0 / n_, 12 * (lo + hi) / 2, np.nan)
    vc = pd.Series(np.round(term, 1)).value_counts(); peaks = np.sort(vc.index[vc >= 400].values); j_ = np.clip(np.searchsorted(peaks, term), 1, len(peaks) - 1)
    near = np.where(np.abs(term - peaks[j_ - 1]) <= np.abs(term - peaks[j_]), peaks[j_ - 1], peaks[j_])
    d["R_TERM_PEAK_NEAREST"] = near; d["R_TERM_PEAK_DISTANCE"] = term - near; d["R_TERM_PEAK_REL_DISTANCE"] = (term - near) / near
    d["R_TERM_RANK"] = pd.Series(term).rank(pct=True).values; d["R_TERM_RANK_BY_CONTRACT"] = pd.Series(term).groupby(al.NAME_CONTRACT_TYPE.values).rank(pct=True).values
    return pd.DataFrame(d)
LABEL_FREE = {"e_poly": blk_e_poly, "ext_pct": blk_ext_pct, "research_lean": lambda: pd.concat([_date_diffs(), _implied_peaks()], axis=1), "date_diffs": _date_diffs, "implied_peaks": _implied_peaks}
BLK = {k: f().replace([np.inf, -np.inf], np.nan).astype(np.float32).reset_index(drop=True) for k, f in LABEL_FREE.items()}
# ---- label-based block: partner flags from employment / phone / registration invariants (NaN keys excluded -> no pensioner mega-group) ----
_B = al.DAYS_BIRTH; _emp = al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243); _g = al.CODE_GENDER.astype(str)
_off = {"id": al.DAYS_ID_PUBLISH - _B, "reg": al.DAYS_REGISTRATION - _B, "emp": _emp - _B, "ph": al.DAYS_LAST_PHONE_CHANGE - _B}
def _mk_key(parts):
    k = _g.copy(); ok = pd.Series(True, index=al.index)
    for p_ in parts: k = k + "|" + _off[p_].astype(str); ok &= _off[p_].notna()
    if "reg" in parts: ok &= (al.DAYS_REGISTRATION != 0)
    codes = pd.factorize(k.where(ok))[0]; sz = pd.Series(codes).map(pd.Series(codes[codes >= 0]).value_counts()).fillna(0).values
    return np.where((codes >= 0) & (sz > 1), codes, -1)
NEWK = {"PYE_EMPID": _mk_key(["emp", "id"]), "PYE_PHID": _mk_key(["ph", "id"]), "PYE_REGEMP": _mk_key(["reg", "emp"])}
def pe2(fit, idx):
    idx = np.asarray(idx); lab = np.full(len(al), np.nan); lab[fit] = y[fit]; out = {}
    for nm, codes in NEWK.items():
        m = codes >= 0; s = np.bincount(codes[m], weights=np.nan_to_num(lab[m]), minlength=codes.max() + 1); c = np.bincount(codes[m], weights=(~np.isnan(lab[m])).astype(float), minlength=codes.max() + 1)
        ci = codes[idx]; own = np.nan_to_num(lab[idx]); ownc = (~np.isnan(lab[idx])).astype(float)
        ps = np.where(ci >= 0, s[np.maximum(ci, 0)] - own, np.nan); pc = np.where(ci >= 0, c[np.maximum(ci, 0)] - ownc, np.nan); out[nm] = np.where((ci >= 0) & (pc > 0), (ps > 0).astype(float), np.nan)
    return pd.DataFrame(out).astype(np.float32)
print("blocks:", {k: v.shape[1] for k, v in BLK.items()}, "| label-based: partner_emp_phone (3 columns)")


# ==================================================
# CODE CELL 11
# ==================================================

# ===== run: baseline + every block on identical fold frames, real LightGBM member params =====
RESF = os.path.join(LAB_DIR, "block_results.json"); RES = json.load(open(RESF)) if (os.path.exists(RESF) and not SMOKE) else {}
NAMES = ["baseline"] + list(BLK) + ["partner_emp_phone"]; q_ = {**BASE, **P_ET}; R_ = 30 if SMOKE else R_ET; begin_stage("blocks"); t0 = time.time()
for k in range(FOLDS):
    if time_up(): print("stage time cap reached"); break
    if all(f"{nm}|{k}" in RES for nm in NAMES): continue
    a, b, fa, fb, ft = lab_frames(k); pa = pe2_a = None
    for nm in NAMES:
        if f"{nm}|{k}" in RES: continue
        if nm == "baseline": A_, B_ = fa, fb
        elif nm == "partner_emp_phone": A_ = pd.concat([fa, nested_generic(a, pe2)], axis=1); B_ = pd.concat([fb, pe2(a, b)], axis=1)
        else: A_ = pd.concat([fa, BLK[nm].iloc[:n].iloc[a].reset_index(drop=True)], axis=1); B_ = pd.concat([fb, BLK[nm].iloc[:n].iloc[b].reset_index(drop=True)], axis=1)
        m = lgb.train(q_, lgb.Dataset(A_, y[a]), R_); RES[f"{nm}|{k}"] = float(roc_auc_score(y[b], m.predict(B_))); del A_, B_, m; gc.collect()
        print(f"fold {k+1}/{FOLDS} {nm:18s} AUC {RES[f'{nm}|{k}']:.5f} ({time.time()-t0:.0f}s) | {ram()}", flush=True)
        if not SMOKE: json.dump(RES, open(RESF, "w"))
    del fa, fb, ft; gc.collect()
rows = []
for nm in NAMES[1:]:
    d = np.array([RES[f"{nm}|{k}"] - RES[f"baseline|{k}"] for k in range(FOLDS) if f"{nm}|{k}" in RES and f"baseline|{k}" in RES])
    if len(d) == 0: continue
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan"); ok = (d.mean() >= 0.0002) and ((d >= 0).sum() >= min(4, len(d)))
    rows.append(dict(block=nm, folds=len(d), mean_gain=round(float(d.mean()), 5), se=round(float(se), 5), per_fold=[round(float(x), 5) for x in d], nonneg=f"{(d >= 0).sum()}/{len(d)}", decision="ACCEPT" if ok else "reject"))
TAB = pd.DataFrame(rows); pd.set_option("display.width", 250, "display.max_colwidth", 80); print("\nBASELINE per-fold AUC:", [round(RES[f"baseline|{k}"], 5) for k in range(FOLDS) if f"baseline|{k}" in RES]); print(TAB.to_string(index=False))
if not SMOKE: TAB.to_csv(os.path.join(OUT_DIR, "v25_block_results.csv"), index=False)


# ==================================================
# CODE CELL 12
# ==================================================

# ===== additivity check: all ACCEPTED label-free blocks together vs baseline (only runs if something was accepted) =====
ACC = [r["block"] for r in rows if r["decision"] == "ACCEPT" and r["block"] in BLK]
if ACC and not time_up():
    cat = pd.concat([BLK[x] for x in ACC], axis=1); d = []
    for k in range(FOLDS):
        a, b, fa, fb, ft = lab_frames(k); A_ = pd.concat([fa, cat.iloc[:n].iloc[a].reset_index(drop=True)], axis=1); B_ = pd.concat([fb, cat.iloc[:n].iloc[b].reset_index(drop=True)], axis=1)
        s = roc_auc_score(y[b], lgb.train(q_, lgb.Dataset(A_, y[a]), R_).predict(B_)); d.append(s - RES[f"baseline|{k}"]); print(f"combined {ACC} fold {k+1}: delta {d[-1]:+.5f}", flush=True); del fa, fb, ft, A_, B_; gc.collect()
    print("COMBINED accepted blocks: mean gain", round(float(np.mean(d)), 5), "| folds non-negative", f"{sum(x >= 0 for x in d)}/{len(d)}")
print("\nNEXT STEP: " + ("wire ONLY the accepted blocks into the v18 pipeline as feature families (cell 2, FAMS + CORE/ALLF), delete gci_cache_v18 member/view files, and rerun end-to-end; accept the new champion only if held-out AUC beats 0.77666." if ACC or any(r["decision"] == "ACCEPT" for r in rows) else "no block was accepted on the real member: v18 remains the final model."))


