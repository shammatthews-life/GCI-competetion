"""Test fold-local credit-annuity frequency gates for the CA CatBoost branch."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE=Path(__file__).resolve().parent; CACHE=HERE/'v15_cache'/'gci_cache_final'
Z=np.load(HERE/'v15_ca_ext_catboost'/'oof.npz'); y=Z['y']; cand_cb=Z['candidate_catboost_oof']
train=pd.read_csv(HERE.parent/'inpuy'/'train.csv'); n=len(y)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
rk=lambda x:rankdata(x,method='average')/len(x)
w={'qf_shallow':.24,'qf_reg':.12,'v1_allfam_oh':.32,'v2_allfam_oh':.16,'v2_mono':.16}
branch={}
for tag in ('old','e1','e1et','e1Aet'):
 p=np.zeros(n)
 for m,a in w.items():p+=a*rk(np.load(CACHE/f'{tag}_{m}.npz')['oof'])
 branch[tag]=p
b=(branch['old']+branch['e1']+branch['e1et'])/3
main=rk(.75*rk(b)+.25*branch['e1Aet'])
old=np.zeros(n)
for i,(_,v) in enumerate(folds,1):old[v]=np.load(CACHE/f'catboost_product_fold_{i}.npz')['oof']
oldr,cr=rk(old),rk(cand_cb)
baseline=.9*main+.1*oldr
count=np.zeros(n)
ca=(train.AMT_CREDIT.fillna(-1).round(0).astype('int64').astype(str)+'|'+train.AMT_ANNUITY.fillna(-1).round(0).astype('int64').astype(str)).to_numpy()
for fit,val in folds:
 c=pd.Series(ca[fit]).value_counts();count[val]=np.log1p(pd.Series(ca[val]).map(c).fillna(0).to_numpy())
freq=np.zeros(n)
for _,val in folds:freq[val]=pd.Series(count[val]).rank(method='average',pct=True).to_numpy()-.5
results=[]
for scale in (.05,.10,.20,.30):
 for sign,name in ((1,'more_candidate_for_common'),(-1,'more_candidate_for_rare')):
  alpha=np.clip(.1+sign*scale*freq,0,1)
  pred=(1-alpha)*main+alpha*cr
  fd=[float(roc_auc_score(y[v],pred[v])-roc_auc_score(y[v],baseline[v])) for _,v in folds]
  results.append({'gate':name,'scale':scale,'weight_min':float(alpha.min()),'weight_max':float(alpha.max()),
                  'auc':float(roc_auc_score(y,pred)),'delta':float(roc_auc_score(y,pred)-roc_auc_score(y,baseline)),
                  'positive_folds':sum(x>0 for x in fd),'fold_deltas':fd})
out=HERE/'v15_ca_ext_catboost';(out/'frequency_gate.json').write_text(json.dumps({'baseline_auc':float(roc_auc_score(y,baseline)),'results':results,'note':'Same reused folds; exploratory gated-weight diagnostics only.'},indent=2),encoding='utf-8')
pd.DataFrame(results).to_csv(out/'frequency_gate.csv',index=False)
print(pd.DataFrame(results).to_string(index=False))
