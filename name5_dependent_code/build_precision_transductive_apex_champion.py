"""Raw-data replacement for the precision transductive apex post-processor.

The historical anchor CSV and Hasan zoo are unavailable by design.  This script
trains five new diverse models from raw CSVs, applies the original-style
disagreement gate, and then uses only documented train/test identity heuristics
for a small transductive adjustment.  It does not claim historical LB parity.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.stats import norm,rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent;D=R.parent/"inpuy";O=R/"outputs"
def rp(v):return norm.ppf(np.clip(rankdata(v)/(len(v)+1),1e-6,1-1e-6))
def feat(tr,te):
 n=len(tr);z=pd.concat([tr.drop(columns="TARGET"),te],ignore_index=True);d={}
 for c in z:
  if c!="SK_ID_CURR":d[c]=pd.factorize(z[c],sort=True)[0] if not pd.api.types.is_numeric_dtype(z[c]) else z[c]
 e=z[[c for c in ("EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3") if c in z]];d.update(ext_mean=e.mean(1),ext_std=e.std(1),ext_missing=e.isna().sum(1),missing=z.isna().sum(1))
 for a,b,k in [("AMT_CREDIT","AMT_INCOME_TOTAL","credit_income"),("AMT_ANNUITY","AMT_INCOME_TOTAL","ann_income"),("AMT_CREDIT","AMT_ANNUITY","term")]:d[k]=z[a]/(z[b]+1)
 x=pd.DataFrame(d).replace([np.inf,-np.inf],np.nan).astype("float32");return x.iloc[:n],x.iloc[n:]
def graph(tr,te,z):
 n=len(tr);both=pd.concat([tr.drop(columns="TARGET"),te],ignore_index=True);keys=both.DAYS_BIRTH.astype(str)+"|"+both.DAYS_REGISTRATION.astype(str)+"|"+both.DAYS_ID_PUBLISH.astype(str)+"|"+both.CODE_GENDER.astype(str)+"|"+both.NAME_EDUCATION_TYPE.astype(str)
 for _,ix in pd.Series(np.arange(len(both))).groupby(keys).groups.items():
  ix=np.asarray(list(ix));a=ix[ix<n];b=ix[ix>=n]-n
  if not len(b):continue
  if len(a):z[b]=.88*z[b]+.12*(2.5 if tr.TARGET.iloc[a].mean()>=.5 else -1.3)
  elif len(b)>1:z[b]=.85*z[b]+.15*z[b].mean()
 return z
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--fast",action="store_true");ap.add_argument("--device",choices=["cpu","cuda"],default="cpu");a=ap.parse_args();O.mkdir(exist_ok=True);tr=pd.read_csv(D/"train.csv");te=pd.read_csv(D/"test.csv");y=tr.TARGET.values;x,xt=feat(tr,te);cv=StratifiedKFold(3 if a.fast else 5,shuffle=True,random_state=42);N=150 if a.fast else 750;oo=[];tt=[]
 for member in range(5):
  o=np.zeros(len(tr));t=np.zeros(len(te))
  for f,(i,j) in enumerate(cv.split(x,y)):
   if member<3:
    pa={"objective":"binary","metric":"auc","verbosity":-1,"learning_rate":.03,"num_leaves":[15,31,127][member],"max_depth":[5,6,8][member],"feature_fraction":[.5,.65,.65][member],"bagging_fraction":.8,"bagging_freq":1,"lambda_l2":[20,8,30][member],"seed":42+f};m=lgb.train(pa,lgb.Dataset(x.iloc[i],label=y[i]),N,valid_sets=[lgb.Dataset(x.iloc[j],label=y[j])],callbacks=[lgb.early_stopping(20 if a.fast else 50,verbose=False)]);o[j]=m.predict(x.iloc[j]);t+=m.predict(xt)/cv.n_splits
   elif member==3:
    m=xgb.XGBClassifier(n_estimators=N,learning_rate=.035,max_depth=6,subsample=.85,colsample_bytree=.6,reg_alpha=.1,reg_lambda=5,tree_method="hist",device=a.device,eval_metric="auc",early_stopping_rounds=20 if a.fast else 50,random_state=42+f);m.fit(x.iloc[i],y[i],eval_set=[(x.iloc[j],y[j])],verbose=False);o[j]=m.predict_proba(x.iloc[j])[:,1];t+=m.predict_proba(xt)[:,1]/cv.n_splits
   else:
    m=CatBoostClassifier(iterations=N,learning_rate=.04,depth=6,l2_leaf_reg=5,verbose=False,allow_writing_files=False,random_seed=42+f);m.fit(x.iloc[i],y[i],eval_set=(x.iloc[j],y[j]),early_stopping_rounds=20 if a.fast else 50,verbose=False);o[j]=m.predict_proba(x.iloc[j])[:,1];t+=m.predict_proba(xt)[:,1]/cv.n_splits
  oo.append(o);tt.append(t);print("member",member,roc_auc_score(y,o))
 z=np.column_stack([rp(v) for v in tt]);base=z.mean(1);spread=z.std(1);gate=1/(1+np.exp(-(spread-np.percentile(spread,80))/(spread.std()*.35+1e-9)));reg=te[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]].isna().sum(1).values;delta=np.where(reg==0,.6*(z[:,4]-z[:,0])+.4*(z[:,2]-z[:,0]),np.where(reg==1,z[:,0]-.5*(z[:,1]+z[:,2]),.7*(z[:,4]-z[:,3])));final=rankdata(graph(tr,te,base+.045*gate*delta))/(len(te)+1);out=pd.DataFrame({"SK_ID_CURR":te.SK_ID_CURR,"TARGET":final});assert len(out)==len(te) and out.TARGET.notna().all() and out.TARGET.between(0,1).all();out.to_csv(O/"v_20261027_optimal_transductive_apex.csv",index=False)
if __name__=="__main__":main()
