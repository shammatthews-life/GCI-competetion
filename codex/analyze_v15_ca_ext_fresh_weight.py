"""Weight curve for the fresh-fold CA CatBoost OOF confirmation candidate."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;C=H/'v15_cache'/'gci_cache_final'
z=np.load(H/'v15_ca_ext_confirmation'/'oof.npz');y=z['y'];cand=z['candidate_catboost_oof']
rk=lambda x:rankdata(x,method='average')/len(x)
w={'qf_shallow':.24,'qf_reg':.12,'v1_allfam_oh':.32,'v2_allfam_oh':.16,'v2_mono':.16};br={}
for tag in ('old','e1','e1et','e1Aet'):
 p=np.zeros(len(y))
 for m,a in w.items():p+=a*rk(np.load(C/f'{tag}_{m}.npz')['oof'])
 br[tag]=p
b=(br['old']+br['e1']+br['e1et'])/3;main=rk(.75*rk(b)+.25*br['e1Aet'])
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y));old=np.zeros(len(y))
for i,(_,v) in enumerate(folds,1):old[v]=np.load(C/f'catboost_product_fold_{i}.npz')['oof']
base=.9*main+.1*rk(old);cr=rk(cand);res=[]
alphas=(0,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50)
for a in alphas:
 p=(1-a)*main+a*cr;fd=[float(roc_auc_score(y[v],p[v])-roc_auc_score(y[v],base[v])) for _,v in folds]
 res.append({'candidate_weight':a,'auc':float(roc_auc_score(y,p)),'delta_vs_v15':float(roc_auc_score(y,p)-roc_auc_score(y,base)),'positive_folds':sum(x>0 for x in fd),'fold_deltas':fd})
out=H/'v15_ca_ext_confirmation';(out/'weight_curve.json').write_text(json.dumps({'baseline_auc':float(roc_auc_score(y,base)),'weights':res,'note':'Candidate OOF uses a separate fold seed; alpha curve is exploratory and reuses labels.'},indent=2),encoding='utf-8');pd.DataFrame(res).to_csv(out/'weight_curve.csv',index=False);print(pd.DataFrame(res).to_string(index=False))
