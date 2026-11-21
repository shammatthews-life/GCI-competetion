"""Raw-CSV replacement for the CatBoost + deep-LightGBM zoo expansion.

Inputs: ../inpuy/train.csv and ../inpuy/test.csv only.
Output: outputs/train_catboost_and_deep_lgb.csv (rank blend of the two models).
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent; DATA = ROOT.parent / "inpuy"; OUT = ROOT / "outputs"
def features(a, b):
    n=len(a); z=pd.concat([a.drop(columns="TARGET"), b], ignore_index=True); d={}
    for c in z:
        if c=="SK_ID_CURR": continue
        if not pd.api.types.is_numeric_dtype(z[c]):
            q,_=pd.factorize(z[c], sort=True); d[c]=pd.Series(q).replace(-1,np.nan)
        else: d[c]=z[c]
    e=z[[c for c in ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"] if c in z]]
    d["EXT_MEAN"]=e.mean(1); d["EXT_STD"]=e.std(1); d["MISSING_COUNT"]=z.isna().sum(1)
    for x,y,name in [("AMT_CREDIT","AMT_INCOME_TOTAL","CREDIT_INCOME"),("AMT_ANNUITY","AMT_INCOME_TOTAL","ANNUITY_INCOME"),("AMT_CREDIT","AMT_ANNUITY","CREDIT_TERM")]:
        d[name]=z[x]/(z[y]+1) if x in z and y in z else np.nan
    x=pd.DataFrame(d).replace([np.inf,-np.inf],np.nan).astype("float32"); return x.iloc[:n],x.iloc[n:]
def check(out, test):
    assert list(out)==["SK_ID_CURR","TARGET"] and len(out)==len(test)
    assert out.TARGET.notna().all() and np.isfinite(out.TARGET).all() and out.TARGET.between(0,1).all()
def main():
    p=argparse.ArgumentParser();p.add_argument("--fast",action="store_true");args=p.parse_args(); OUT.mkdir(exist_ok=True)
    tr=pd.read_csv(DATA/"train.csv");te=pd.read_csv(DATA/"test.csv"); y=tr.TARGET.to_numpy();x,xt=features(tr,te)
    cv=StratifiedKFold(3 if args.fast else 5,shuffle=True,random_state=42); rounds=150 if args.fast else 800
    o1=np.zeros(len(tr));t1=np.zeros(len(te));o2=o1.copy();t2=t1.copy()
    for f,(a,b) in enumerate(cv.split(x,y)):
        cb=CatBoostClassifier(iterations=rounds,learning_rate=.04,depth=6,l2_leaf_reg=5,loss_function="Logloss",eval_metric="AUC",random_seed=42+f,verbose=False,allow_writing_files=False)
        cb.fit(x.iloc[a],y[a],eval_set=(x.iloc[b],y[b]),early_stopping_rounds=30 if args.fast else 60,verbose=False);o1[b]=cb.predict_proba(x.iloc[b])[:,1];t1+=cb.predict_proba(xt)[:,1]/cv.n_splits
        params={"objective":"binary","metric":"auc","verbosity":-1,"learning_rate":.025,"num_leaves":127,"max_depth":8,"min_child_samples":80,"feature_fraction":.65,"bagging_fraction":.8,"bagging_freq":1,"lambda_l1":10,"lambda_l2":30,"seed":42+f}
        m=lgb.train(params,lgb.Dataset(x.iloc[a],label=y[a]),rounds,valid_sets=[lgb.Dataset(x.iloc[b],label=y[b])],callbacks=[lgb.early_stopping(30 if args.fast else 50,verbose=False)]);o2[b]=m.predict(x.iloc[b]);t2+=m.predict(xt)/cv.n_splits
    print("CatBoost AUC",roc_auc_score(y,o1),"Deep LGB AUC",roc_auc_score(y,o2))
    pred=(rankdata(t1)+rankdata(t2))/(2*(len(te)+1));out=pd.DataFrame({"SK_ID_CURR":te.SK_ID_CURR,"TARGET":pred});check(out,te);out.to_csv(OUT/"train_catboost_and_deep_lgb.csv",index=False)
if __name__=="__main__": main()
