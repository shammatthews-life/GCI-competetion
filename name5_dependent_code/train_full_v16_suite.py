"""Raw-data-only v16-style multi-configuration LightGBM ensemble.

Historical residual/view/encoding caches and v6/v8 bundles are deliberately
not read.  The five original LightGBM member configurations are retrained from
features constructed here and blended using the documented member weights.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
R=Path(__file__).resolve().parent;D=R.parent/"inpuy";O=R/"outputs"
def feat(tr,te):
 n=len(tr);z=pd.concat([tr.drop(columns="TARGET"),te],ignore_index=True);v={}
 for c in z:
  if c!="SK_ID_CURR":v[c]=pd.factorize(z[c],sort=True)[0] if not pd.api.types.is_numeric_dtype(z[c]) else z[c]
 e=z[[c for c in ("EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3") if c in z]];v.update(ext_mean=e.mean(1),ext_min=e.min(1),ext_std=e.std(1),missing=z.isna().sum(1))
 for a,b,k in [("AMT_CREDIT","AMT_INCOME_TOTAL","credit_income"),("AMT_ANNUITY","AMT_INCOME_TOTAL","ann_income"),("AMT_GOODS_PRICE","AMT_CREDIT","goods_credit"),("DAYS_EMPLOYED","DAYS_BIRTH","employment_age")]:v[k]=z[a]/(z[b]+1)
 x=pd.DataFrame(v).replace([np.inf,-np.inf],np.nan).astype("float32");return x.iloc[:n],x.iloc[n:]
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--fast",action="store_true");a=ap.parse_args();O.mkdir(exist_ok=True);tr=pd.read_csv(D/"train.csv");te=pd.read_csv(D/"test.csv");y=tr.TARGET.values;x,xt=feat(tr,te);cv=StratifiedKFold(3 if a.fast else 5,shuffle=True,random_state=42);rounds=150 if a.fast else 1000
 specs=[("qf_shallow",.05,7,3,150,.5,20), ("qf_reg",.03,11,-1,300,.4,40), ("v1_allfam_oh",.04,15,-1,150,.3,20), ("v2_allfam_oh",.04,15,-1,150,.3,20), ("v2_mono",.02,15,-1,150,.5,20)]; weights=np.array([.24,.12,.32,.16,.16]);oo=[];tt=[]
 for k,lr,leaves,depth,child,frac,l2 in specs:
  o=np.zeros(len(tr));t=np.zeros(len(te));
  for f,(i,j) in enumerate(cv.split(x,y)):
   mono=[-1 if c in ("EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3") else 0 for c in x] if k=="v2_mono" else None
   par={"objective":"binary","metric":"auc","verbosity":-1,"learning_rate":lr,"num_leaves":leaves,"max_depth":depth,"min_child_samples":child,"feature_fraction":frac,"bagging_fraction":.8,"bagging_freq":1,"lambda_l2":l2,"seed":42+f}
   if mono is not None:par["monotone_constraints"]=mono
   m=lgb.train(par,lgb.Dataset(x.iloc[i],label=y[i]),rounds,valid_sets=[lgb.Dataset(x.iloc[j],label=y[j])],callbacks=[lgb.early_stopping(25 if a.fast else 50,verbose=False)]);o[j]=m.predict(x.iloc[j]);t+=m.predict(xt)/cv.n_splits
  oo.append(o);tt.append(t);print(k,roc_auc_score(y,o))
 blend=sum(w*rankdata(o) for w,o in zip(weights,oo));print("v16-style OOF",roc_auc_score(y,blend));pred=rankdata(sum(w*rankdata(t) for w,t in zip(weights,tt)))/(len(te)+1);out=pd.DataFrame({"SK_ID_CURR":te.SK_ID_CURR,"TARGET":pred});assert len(out)==len(te) and out.TARGET.notna().all() and out.TARGET.between(0,1).all();out.to_csv(O/"train_full_v16_suite.csv",index=False)
if __name__=="__main__":main()
