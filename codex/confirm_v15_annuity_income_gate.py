"""Confirm the selected lowest-annuity/income gate on new-seed CA CatBoost OOF."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;C=H/'v15_cache'/'gci_cache_final'; z=np.load(H/'v15_ca_ext_confirmation'/'oof.npz')
y=z['y']; cand=z['candidate_catboost_oof']; tr=pd.read_csv(H.parent/'inpuy'/'train.csv')
rk=lambda x:rankdata(x,method='average')/len(x); w={'qf_shallow':.24,'qf_reg':.12,'v1_allfam_oh':.32,'v2_allfam_oh':.16,'v2_mono':.16};br={}
for tag in ('old','e1','e1et','e1Aet'):
 p=np.zeros(len(y))
 for m,a in w.items():p+=a*rk(np.load(C/f'{tag}_{m}.npz')['oof'])
 br[tag]=p
b=(br['old']+br['e1']+br['e1et'])/3; main=rk(.75*rk(b)+.25*br['e1Aet'])
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y));old=np.zeros(len(y))
for i,(_,v) in enumerate(folds,1):old[v]=np.load(C/f'catboost_product_fold_{i}.npz')['oof']
base=.9*main+.1*rk(old); p=(tr.AMT_ANNUITY/(tr.AMT_INCOME_TOTAL.abs()+1)).rank(method='average')
q=pd.qcut(p,5,labels=False,duplicates='drop').to_numpy();mask=q==0;prev=float(mask.mean());alpha=.10+.05*(mask.astype(float)-prev)
pred=(1-alpha)*main+alpha*rk(cand)
fd=[float(roc_auc_score(y[v],pred[v])-roc_auc_score(y[v],base[v])) for _,v in folds]
res={'baseline_auc':float(roc_auc_score(y,base)),'fixed_10pct_candidate_auc':float(roc_auc_score(y,.9*main+.1*rk(cand))),
 'fixed_10pct_candidate_delta':float(roc_auc_score(y,.9*main+.1*rk(cand))-roc_auc_score(y,base)),
 'frozen_gate':'increase CA candidate weight by 5 percentage points in lowest annuity/income quintile; reduce outside to preserve mean 10%',
 'gate_prevalence':prev,'weight_in_group':float(alpha[mask].mean()),'weight_outside':float(alpha[~mask].mean()),
 'gated_auc':float(roc_auc_score(y,pred)),'gated_delta_vs_v15':float(roc_auc_score(y,pred)-roc_auc_score(y,base)),
 'positive_folds':sum(x>0 for x in fd),'fold_deltas':fd,
 'note':'Candidate CA CatBoost OOF used a different outer-fold seed (20261021) and model seed (2024); the gate was selected from prior seed-42 analysis.'}
out=H/'v15_ca_ext_confirmation';(out/'annuity_income_gate_confirmation.json').write_text(json.dumps(res,indent=2),encoding='utf-8');print(json.dumps(res,indent=2))
