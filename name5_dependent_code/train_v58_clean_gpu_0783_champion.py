"""Independent raw-data v58-style multi-engine ensemble.

The unavailable v55/v43 artifacts are replaced by seven freshly trained OOF
members: three LightGBM variants, ExtraTrees, CatBoost and two XGBoost variants.
Weights and the solvency adjustment are fitted on OOF predictions only.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent;D=R.parent/"inpuy";O=R/"outputs"
def fx(tr,te):
 n=len(tr);z=pd.concat([tr.drop(columns="TARGET"),te],ignore_index=True);d={}
 for c in z:
  if c!="SK_ID_CURR":d[c]=pd.factorize(z[c],sort=True)[0] if not pd.api.types.is_numeric_dtype(z[c]) else z[c]
 e=z[[c for c in ("EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3") if c in z]];d.update(ext_mean=e.mean(1),ext_std=e.std(1),missing=z.isna().sum(1))
 for a,b,name in [("AMT_CREDIT","AMT_INCOME_TOTAL","ci"),("AMT_ANNUITY","AMT_INCOME_TOTAL","ai"),("AMT_CREDIT","AMT_ANNUITY","term")]:d[name]=z[a]/(z[b]+1)
 x=pd.DataFrame(d).replace([np.inf,-np.inf],np.nan).astype("float32");return x.iloc[:n],x.iloc[n:]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--fast",action="store_true");ap.add_argument("--device",choices=["cpu","cuda"],default="cpu");a=ap.parse_args();O.mkdir(exist_ok=True);tr=pd.read_csv(D/"train.csv");te=pd.read_csv(D/"test.csv");y=tr.TARGET.values;x,xt=fx(tr,te);cv=StratifiedKFold(3 if a.fast else 5,shuffle=True,random_state=42);N=120 if a.fast else 700; oo=[];tt=[]
 for member in range(7):
  o=np.zeros(len(tr));t=np.zeros(len(te))
  for f,(i,j) in enumerate(cv.split(x,y)):
   if member<3:
    pa={"objective":"binary","metric":"auc","verbosity":-1,"learning_rate":.03,"num_leaves":[15,31,63][member],"max_depth":[5,6,8][member],"feature_fraction":[.5,.65,.75][member],"bagging_fraction":.8,"bagging_freq":1,"lambda_l2":[20,8,3][member],"seed":42+f};m=lgb.train(pa,lgb.Dataset(x.iloc[i],label=y[i]),N,valid_sets=[lgb.Dataset(x.iloc[j],label=y[j])],callbacks=[lgb.early_stopping(20 if a.fast else 50,verbose=False)]);o[j]=m.predict(x.iloc[j]);t+=m.predict(xt)/cv.n_splits
   elif member==3:
    m=ExtraTreesClassifier(n_estimators=150 if a.fast else 500,min_samples_leaf=20,max_features=.7,n_jobs=1,random_state=42+f);m.fit(x.iloc[i].fillna(-999),y[i]);o[j]=m.predict_proba(x.iloc[j].fillna(-999))[:,1];t+=m.predict_proba(xt.fillna(-999))[:,1]/cv.n_splits
   elif member==4:
    m=CatBoostClassifier(iterations=N,learning_rate=.04,depth=6,l2_leaf_reg=5,verbose=False,allow_writing_files=False,random_seed=42+f);m.fit(x.iloc[i],y[i],eval_set=(x.iloc[j],y[j]),early_stopping_rounds=20 if a.fast else 50,verbose=False);o[j]=m.predict_proba(x.iloc[j])[:,1];t+=m.predict_proba(xt)[:,1]/cv.n_splits
   else:
    m=xgb.XGBClassifier(n_estimators=N,learning_rate=.035,max_depth=6 if member==5 else 8,subsample=.8,colsample_bytree=.65,tree_method="hist",device=a.device,eval_metric="auc",early_stopping_rounds=20 if a.fast else 50,random_state=42+f);m.fit(x.iloc[i],y[i],eval_set=[(x.iloc[j],y[j])],verbose=False);o[j]=m.predict_proba(x.iloc[j])[:,1];t+=m.predict_proba(xt)[:,1]/cv.n_splits
  oo.append(o);tt.append(t);print("member",member,roc_auc_score(y,o))
 M=np.column_stack([rankdata(v)/(len(tr)+1) for v in oo]);loss=lambda w:-roc_auc_score(y,M@(np.maximum(w,0)/(np.maximum(w,0).sum()+1e-12)));w=np.maximum(minimize(loss,np.ones(7)/7,method="Nelder-Mead",options={"maxiter":500}).x,0);w/=w.sum();base=M@w
 inc=tr.AMT_INCOME_TOTAL.values;ann=tr.AMT_ANNUITY.values;cr=tr.AMT_CREDIT.values;sol=np.column_stack([rankdata((inc-ann)/tr.CNT_FAM_MEMBERS.fillna(1))/(len(tr)+1),rankdata(inc/(ann+1))/(len(tr)+1),rankdata(cr/(inc+1))/(len(tr)+1)])
 q=minimize(lambda v:-roc_auc_score(y,base+sol@v),[0,0,0],method="Nelder-Mead").x;mt=np.column_stack([rankdata(v)/(len(te)+1) for v in tt]);bt=mt@w;st=np.column_stack([rankdata((te.AMT_INCOME_TOTAL-te.AMT_ANNUITY)/te.CNT_FAM_MEMBERS.fillna(1))/(len(te)+1),rankdata(te.AMT_INCOME_TOTAL/(te.AMT_ANNUITY+1))/(len(te)+1),rankdata(te.AMT_CREDIT/(te.AMT_INCOME_TOTAL+1))/(len(te)+1)]);pred=rankdata(bt+st@q)/(len(te)+1);print("OOF",roc_auc_score(y,base+sol@q),"weights",w);out=pd.DataFrame({"SK_ID_CURR":te.SK_ID_CURR,"TARGET":pred});assert len(out)==len(te) and out.TARGET.notna().all() and out.TARGET.between(0,1).all();out.to_csv(O/"train_v58_clean_gpu_0783_champion.csv",index=False)
if __name__=="__main__":main()
