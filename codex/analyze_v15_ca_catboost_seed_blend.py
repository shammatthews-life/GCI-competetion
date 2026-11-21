"""Explore candidate CA CatBoost OOF seed ensembles against full v15."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

H=Path(__file__).resolve().parent; C=H/'v15_cache'/'gci_cache_final'; rk=lambda x:rankdata(x,method='average')/len(x)
z42=np.load(H/'v15_ca_ext_catboost'/'oof.npz'); z24=np.load(H/'v15_ca_ext_confirmation'/'oof.npz'); y=z42['y']
w={'qf_shallow':.24,'qf_reg':.12,'v1_allfam_oh':.32,'v2_allfam_oh':.16,'v2_mono':.16}; br={}
for tag in ('old','e1','e1et','e1Aet'):
 p=np.zeros(len(y))
 for m,a in w.items():p+=a*rk(np.load(C/f'{tag}_{m}.npz')['oof'])
 br[tag]=p
b=(br['old']+br['e1']+br['e1et'])/3; main=rk(.75*rk(b)+.25*br['e1Aet'])
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y));old=np.zeros(len(y))
for i,(_,v) in enumerate(folds,1):old[v]=np.load(C/f'catboost_product_fold_{i}.npz')['oof']
baseline=.9*main+.1*rk(old); results=[]
p42=rk(z42['candidate_catboost_oof']);p24=rk(z24['candidate_catboost_oof'])
for w24 in (0,.25,.5,.75,1):
 cb=(1-w24)*p42+w24*p24
 for alpha in (.10,.15,.20,.25):
  pred=(1-alpha)*main+alpha*cb
  fd=[float(roc_auc_score(y[v],pred[v])-roc_auc_score(y[v],baseline[v])) for _,v in folds]
  results.append({'seed2024_share':w24,'candidate_weight':alpha,'auc':float(roc_auc_score(y,pred)),
   'delta_vs_v15':float(roc_auc_score(y,pred)-roc_auc_score(y,baseline)),'positive_folds':sum(x>0 for x in fd),'fold_deltas':fd})
summary={'baseline_auc':float(roc_auc_score(y,baseline)),'seed42_vs_seed2024_rank_spearman':float(spearmanr(p42,p24).statistic),
 'results':results,'note':'Exploratory same-row OOF grid. Selection requires a new frozen confirmation and is not submission evidence.'}
out=H/'v15_ca_ext_catboost';(out/'seed_blend_sensitivity.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
pd.DataFrame(results).to_csv(out/'seed_blend_sensitivity.csv',index=False)
print('Spearman:',summary['seed42_vs_seed2024_rank_spearman'])
print(pd.DataFrame(results).sort_values('delta_vs_v15',ascending=False).head(12).to_string(index=False))
