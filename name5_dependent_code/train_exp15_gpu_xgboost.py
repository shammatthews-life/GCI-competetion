"""Independent EXP15-style XGBoost pipeline; raw train.csv/test.csv only.

The former v17 champion blend is unavailable without its bundle.  This script
therefore trains the enriched XGBoost candidate and a fresh LightGBM baseline,
then selects their OOF blend weight legitimately.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
R=Path(__file__).resolve().parent; D=R.parent/"inpuy"; O=R/"outputs"
def make(tr,te):
 n=len(tr);z=pd.concat([tr.drop(columns="TARGET"),te],ignore_index=True);q={}
 for c in z:
  if c=="SK_ID_CURR":continue
  q[c]=pd.factorize(z[c],sort=True)[0] if not pd.api.types.is_numeric_dtype(z[c]) else z[c]
 e=z[[c for c in ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"] if c in z]];q["ext_mean"]=e.mean(1);q["ext_nans"]=e.isna().sum(1);q["missing"]=z.isna().sum(1)
 for a,b,nm in [("AMT_CREDIT","AMT_INCOME_TOTAL","debt_income"),("AMT_ANNUITY","AMT_INCOME_TOTAL","ann_income"),("DAYS_EMPLOYED","DAYS_BIRTH","emp_age")]:q[nm]=z[a]/(z[b]+1)
 x=pd.DataFrame(q).replace([np.inf,-np.inf],np.nan).astype("float32");return x.iloc[:n],x.iloc[n:]
def main():
 p=argparse.ArgumentParser();p.add_argument("--fast",action="store_true");p.add_argument("--device",default="cpu",choices=["cpu","cuda"]);a=p.parse_args();O.mkdir(exist_ok=True)
 tr=pd.read_csv(D/"train.csv");te=pd.read_csv(D/"test.csv");y=tr.TARGET.values;x,xt=make(tr,te);cv=StratifiedKFold(3 if a.fast else 5,shuffle=True,random_state=42);N=180 if a.fast else 1200
 ox=np.zeros(len(tr));tx=np.zeros(len(te));ol=ox.copy();tl=tx.copy()
 for f,(i,j) in enumerate(cv.split(x,y)):
  xm=xgb.XGBClassifier(n_estimators=N,learning_rate=.035,max_depth=6,subsample=.85,colsample_bytree=.6,reg_alpha=.1,reg_lambda=5,tree_method="hist",device=a.device,eval_metric="auc",early_stopping_rounds=25 if a.fast else 60,random_state=42+f)
  xm.fit(x.iloc[i],y[i],eval_set=[(x.iloc[j],y[j])],verbose=False);ox[j]=xm.predict_proba(x.iloc[j])[:,1];tx+=xm.predict_proba(xt)[:,1]/cv.n_splits
  lm=lgb.train({"objective":"binary","metric":"auc","verbosity":-1,"learning_rate":.03,"num_leaves":31,"max_depth":6,"feature_fraction":.65,"bagging_fraction":.8,"bagging_freq":1,"seed":42+f},lgb.Dataset(x.iloc[i],label=y[i]),N,valid_sets=[lgb.Dataset(x.iloc[j],label=y[j])],callbacks=[lgb.early_stopping(25 if a.fast else 50,verbose=False)]);ol[j]=lm.predict(x.iloc[j]);tl+=lm.predict(xt)/cv.n_splits
 best=(0,roc_auc_score(y,ol))
 for w in np.linspace(0,1,21):
  s=roc_auc_score(y,(1-w)*rankdata(ol)+w*rankdata(ox));best=max(best,(w,s),key=lambda k:k[1])
 w,score=best;print("OOF XGB",roc_auc_score(y,ox),"OOF blend",score,"weight",w)
 pred=rankdata((1-w)*rankdata(tl)+w*rankdata(tx))/(len(te)+1);out=pd.DataFrame({"SK_ID_CURR":te.SK_ID_CURR,"TARGET":pred});assert len(out)==len(te) and out.TARGET.notna().all() and out.TARGET.between(0,1).all();out.to_csv(O/"train_exp15_gpu_xgboost.csv",index=False)
if __name__=="__main__":main()
