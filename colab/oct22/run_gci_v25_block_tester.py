"""
GCI World 2026 - Execution of GCI_v25_block_tester
Exact replication of GCI_v25_block_tester.ipynb:
Tests candidate feature blocks against the real LightGBM ExtraTrees member (v2_allfam_oh)
on identical paired fold frames to determine statistically significant positive additions.
"""

import os, time, gc, json, itertools, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

t0 = time.time()
print("==========================================================================")
print(" GCI World 2026 - Running GCI_v25_block_tester on Paired Folds")
print("==========================================================================")

WORK_DIR = 'd:/gci/gemini'
CACHE_DIR = os.path.join(WORK_DIR, 'cache')
OCT22_DIR = 'd:/gci/pre info/oct22'
LAB_DIR = os.path.join(OCT22_DIR, 'gci_cache_v25')
OUT_DIR = OCT22_DIR
os.makedirs(LAB_DIR, exist_ok=True)

tr = pd.read_csv('d:/gci/inpuy/train.csv')
te = pd.read_csv('d:/gci/inpuy/test.csv')
y = tr['TARGET'].to_numpy()
n = len(tr)
nt = len(te)
al = pd.concat([tr.drop(columns=['TARGET']), te], ignore_index=True)

FOLDS = 5
FOLD_SPLITS = list(StratifiedKFold(FOLDS, shuffle=True, random_state=42).split(np.zeros(n), y))

print("[1/4] Loading Pre-Cached Feature Scaffolds (Xa, Views, Encodings)...")
Xa = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xa.feather'))
Xat = pd.read_feather(os.path.join(CACHE_DIR, 'v16_Xat.feather'))
views = pd.read_pickle(os.path.join(CACHE_DIR, 'e1_views.pkl'))

# Model Parameters: Exact v2_allfam_oh LightGBM ExtraTrees member
BASE = dict(objective="binary", force_col_wise=True, bagging_freq=1, verbose=-1, n_jobs=-1, seed=42)
P_ET = dict(learning_rate=0.04, num_leaves=15, min_child_samples=150, feature_fraction=0.3, bagging_fraction=0.8, lambda_l2=20, extra_trees=True)
R_ET = 881
q_ = {**BASE, **P_ET}

# Function to build lab frames for fold k using cached artifacts
def lab_frames(k):
    a, b = FOLD_SPLITS[k]
    enc_k = pd.read_pickle(os.path.join(CACHE_DIR, f'v16_enc_fold_{k+1}.pkl'))
    fa = pd.concat([Xa.iloc[a].reset_index(drop=True), enc_k['fa_enc'], views['tr'][k]], axis=1)
    fb = pd.concat([Xa.iloc[b].reset_index(drop=True), enc_k['fb_enc'], views['va'][k]], axis=1)
    ft = pd.concat([Xat.reset_index(drop=True), enc_k['ft_enc'], views['te'][k]], axis=1)
    return a, b, fa.astype(np.float32), fb.astype(np.float32), ft.astype(np.float32)

print("[2/4] Constructing Candidate Feature Blocks from Cell 10...")
E3 = al[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
em = E3.mean(axis=1)
cr_, an_, inc_, gp_ = al.AMT_CREDIT, al.AMT_ANNUITY, al.AMT_INCOME_TOTAL, al.AMT_GOODS_PRICE
age_ = -al.DAYS_BIRTH / 365.25

def blk_e_poly():
    return pd.DataFrame({
        "P_EXTMEAN_PAY": em * an_ / cr_,
        "P_EXTMEAN_CTI": em * cr_ / inc_,
        "P_EXTMEAN_PAY_AGE": em * an_ / cr_ * age_,
        "P_CREDIT_GOODS_INC": (cr_ - gp_) / inc_,
        "P_EXTMEAN_ANN_INC": em * an_ / inc_
    })

def blk_ext_pct():
    agegrp = (age_ // 5).astype(int).astype(str)
    pct = lambda keys: em.groupby(keys).rank(pct=True)
    return pd.DataFrame({
        "EXTF_PCT_OCC": pct(al.OCCUPATION_TYPE.astype(str)),
        "EXTF_PCT_GEN_AGE": pct(al.CODE_GENDER.astype(str) + "|" + agegrp),
        "EXTF_PCT_ORG": pct(al.ORGANIZATION_TYPE.astype(str)),
        "EXTF_PCT_INCTYPE": pct(al.NAME_INCOME_TYPE.astype(str))
    })

def _date_diffs():
    dc = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]
    D = al[dc].copy()
    D["DAYS_EMPLOYED"] = D["DAYS_EMPLOYED"].where(D["DAYS_EMPLOYED"] != 365243)
    return pd.DataFrame({f"RDIFF_{a_[5:]}_{b_[5:]}": (D[a_] - D[b_]).values for a_, b_ in itertools.combinations(dc, 2)})

def _implied_peaks():
    cr = al.AMT_CREDIT.astype(float).values
    an = al.AMT_ANNUITY.astype(float).values
    q = an / cr
    term = cr / an
    d = {}
    for n_ in (12, 24, 36, 48, 60):
        lo, hi = np.full(len(q), 1e-9), np.full(len(q), 0.5)
        for _ in range(50):
            mid = (lo + hi) / 2
            f_ = mid / (1 - (1 + mid) ** (-n_)) - q
            hi = np.where(f_ > 0, mid, hi)
            lo = np.where(f_ > 0, lo, mid)
        d[f"R_IMPLIED_RATE_{n_}M"] = np.where(q > 1.0 / n_, 12 * (lo + hi) / 2, np.nan)
    vc = pd.Series(np.round(term, 1)).value_counts()
    peaks = np.sort(vc.index[vc >= 400].values)
    j_ = np.clip(np.searchsorted(peaks, term), 1, len(peaks) - 1)
    near = np.where(np.abs(term - peaks[j_ - 1]) <= np.abs(term - peaks[j_]), peaks[j_ - 1], peaks[j_])
    d["R_TERM_PEAK_NEAREST"] = near
    d["R_TERM_PEAK_DISTANCE"] = term - near
    d["R_TERM_PEAK_REL_DISTANCE"] = (term - near) / near
    d["R_TERM_RANK"] = pd.Series(term).rank(pct=True).values
    d["R_TERM_RANK_BY_CONTRACT"] = pd.Series(term).groupby(al.NAME_CONTRACT_TYPE.values).rank(pct=True).values
    return pd.DataFrame(d)

LABEL_FREE = {
    "e_poly": blk_e_poly,
    "ext_pct": blk_ext_pct,
    "research_lean": lambda: pd.concat([_date_diffs(), _implied_peaks()], axis=1),
    "date_diffs": _date_diffs,
    "implied_peaks": _implied_peaks
}

BLK = {k: f().replace([np.inf, -np.inf], np.nan).astype(np.float32).reset_index(drop=True) for k, f in LABEL_FREE.items()}

# Label-based block: partner flags from employment / phone / registration invariants (NaN pensioner keys excluded)
_B = al.DAYS_BIRTH
_emp = al.DAYS_EMPLOYED.where(al.DAYS_EMPLOYED != 365243)
_g = al.CODE_GENDER.astype(str)
_off = {"id": al.DAYS_ID_PUBLISH - _B, "reg": al.DAYS_REGISTRATION - _B, "emp": _emp - _B, "ph": al.DAYS_LAST_PHONE_CHANGE - _B}

def _mk_key(parts):
    k = _g.copy()
    ok = pd.Series(True, index=al.index)
    for p_ in parts:
        k = k + "|" + _off[p_].astype(str)
        ok &= _off[p_].notna()
    if "reg" in parts:
        ok &= (al.DAYS_REGISTRATION != 0)
    codes = pd.factorize(k.where(ok))[0]
    sz = pd.Series(codes).map(pd.Series(codes[codes >= 0]).value_counts()).fillna(0).values
    return np.where((codes >= 0) & (sz > 1), codes, -1)

NEWK = {
    "PYE_EMPID": _mk_key(["emp", "id"]),
    "PYE_PHID": _mk_key(["ph", "id"]),
    "PYE_REGEMP": _mk_key(["reg", "emp"])
}

def pe2(fit, idx):
    idx = np.asarray(idx)
    lab = np.full(len(al), np.nan)
    lab[fit] = y[fit]
    out = {}
    for nm, codes in NEWK.items():
        m = codes >= 0
        s = np.bincount(codes[m], weights=np.nan_to_num(lab[m]), minlength=codes.max() + 1)
        c = np.bincount(codes[m], weights=(~np.isnan(lab[m])).astype(float), minlength=codes.max() + 1)
        ci = codes[idx]
        own = np.nan_to_num(lab[idx])
        ownc = (~np.isnan(lab[idx])).astype(float)
        ps = np.where(ci >= 0, s[np.maximum(ci, 0)] - own, np.nan)
        pc = np.where(ci >= 0, c[np.maximum(ci, 0)] - ownc, np.nan)
        out[nm] = np.where((ci >= 0) & (pc > 0), (ps > 0).astype(float), np.nan)
    return pd.DataFrame(out).astype(np.float32)

def nested_generic(a, fn):
    out = pd.DataFrame(index=range(len(a)))
    inner = list(StratifiedKFold(3, shuffle=True, random_state=11).split(np.zeros(len(a)), y[a]))
    for ia, ib in inner:
        sub = fn(a[ia], a[ib])
        for c in sub.columns:
            if c not in out.columns:
                out[c] = np.nan
            out.iloc[ib, out.columns.get_loc(c)] = sub[c].values
    return out.astype(np.float32)

print("Constructed Candidate Blocks:")
for k, v in BLK.items():
    print(f"  {k:15s}: {v.shape[1]} columns")
print(f"  {'partner_emp_phone':15s}: 3 columns (label-based, fold-safe)")

# -------------------------------------------------------------------------
# RUN PAIRED BLOCK TESTER
# -------------------------------------------------------------------------
print("\n[3/4] Running Paired Fold Evaluations (Baseline vs Blocks)...")
RESF = os.path.join(LAB_DIR, "block_results.json")
RES = json.load(open(RESF)) if os.path.exists(RESF) else {}
NAMES = ["baseline"] + list(BLK) + ["partner_emp_phone"]

for k in range(FOLDS):
    t_k = time.time()
    print(f"\n--- FOLD {k+1}/{FOLDS} ---", flush=True)
    a, b, fa, fb, ft = lab_frames(k)
    
    for nm in NAMES:
        if f"{nm}|{k}" in RES:
            print(f"  {nm:18s} cached: AUC = {RES[f'{nm}|{k}']:.5f}")
            continue
        
        t_m = time.time()
        if nm == "baseline":
            A_, B_ = fa, fb
        elif nm == "partner_emp_phone":
            A_ = pd.concat([fa, nested_generic(a, pe2)], axis=1)
            B_ = pd.concat([fb, pe2(a, b)], axis=1)
        else:
            A_ = pd.concat([fa, BLK[nm].iloc[:n].iloc[a].reset_index(drop=True)], axis=1)
            B_ = pd.concat([fb, BLK[nm].iloc[:n].iloc[b].reset_index(drop=True)], axis=1)
        
        m = lgb.train(q_, lgb.Dataset(A_, y[a]), R_ET)
        score = float(roc_auc_score(y[b], m.predict(B_)))
        RES[f"{nm}|{k}"] = score
        json.dump(RES, open(RESF, "w"))
        
        del A_, B_, m
        gc.collect()
        print(f"  {nm:18s} AUC = {score:.5f} ({time.time() - t_m:.1f}s)", flush=True)
    
    del fa, fb, ft
    gc.collect()

# -------------------------------------------------------------------------
# COMPILE DECISION TABLE
# -------------------------------------------------------------------------
print("\n[4/4] Statistical Summary & Acceptance Decisions:")
rows = []
for nm in NAMES[1:]:
    d = np.array([RES[f"{nm}|{k}"] - RES[f"baseline|{k}"] for k in range(FOLDS) if f"{nm}|{k}" in RES and f"baseline|{k}" in RES])
    if len(d) == 0:
        continue
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    ok = (d.mean() >= 0.0002) and ((d >= 0).sum() >= min(4, len(d)))
    rows.append(dict(
        block=nm,
        folds=len(d),
        mean_gain=round(float(d.mean()), 5),
        se=round(float(se), 5),
        per_fold=[round(float(x), 5) for x in d],
        nonneg=f"{(d >= 0).sum()}/{len(d)}",
        decision="ACCEPT" if ok else "reject"
    ))

TAB = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_colwidth", 80)
print("\nBASELINE per-fold AUC:", [round(RES[f"baseline|{k}"], 5) for k in range(FOLDS) if f"baseline|{k}" in RES])
print(TAB.to_string(index=False))

out_csv = os.path.join(OUT_DIR, "v25_block_results.csv")
TAB.to_csv(out_csv, index=False)
print(f"\nSaved official results table to: {out_csv}")

# Check additivity if any block was accepted
ACC = [r["block"] for r in rows if r["decision"] == "ACCEPT" and r["block"] in BLK]
if ACC:
    print(f"\nChecking Additivity of Accepted Blocks: {ACC}...")
    cat = pd.concat([BLK[x] for x in ACC], axis=1)
    d = []
    for k in range(FOLDS):
        a, b, fa, fb, ft = lab_frames(k)
        A_ = pd.concat([fa, cat.iloc[:n].iloc[a].reset_index(drop=True)], axis=1)
        B_ = pd.concat([fb, cat.iloc[:n].iloc[b].reset_index(drop=True)], axis=1)
        s = roc_auc_score(y[b], lgb.train(q_, lgb.Dataset(A_, y[a]), R_ET).predict(B_))
        delta = s - RES[f"baseline|{k}"]
        d.append(delta)
        print(f"  Combined Fold {k+1}: delta = {delta:+.5f}", flush=True)
        del fa, fb, ft, A_, B_
        gc.collect()
    print("COMBINED accepted blocks: mean gain =", round(float(np.mean(d)), 5), "| folds non-negative =", f"{sum(x >= 0 for x in d)}/{len(d)}")
else:
    print("\nResult: No individual block met the pre-registered acceptance gate (+0.0002 mean gain with >=4/5 non-negative folds).")

print(f"\nGCI_v25_block_tester completed in {time.time() - t0:.1f}s.")
