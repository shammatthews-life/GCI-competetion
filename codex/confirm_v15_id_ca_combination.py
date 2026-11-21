"""Fresh CA-seed check of the ID-context plus CA-CatBoost combination."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;C=H/'v15_cache'/'gci_cache_final'
id_z=np.load(H/'v15_id_context_experiment'/'oof.npz');fresh=np.load(H/'v15_ca_ext_confirmation'/'oof.npz')
y=id_z['y'];id_member=id_z['candidate_member_oof'];fresh_cb=fresh['candidate_catboost_oof']
rk=lambda x:rankdata(x,method='average')/len(x);W={'qf_shallow':.24,'qf_reg':.12,'v1_allfam_oh':.32,'v2_allfam_oh':.16,'v2_mono':.16}
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
def formula(use_id,use_cb):
 br={}
 for tag in ('old','e1','e1et','e1Aet'):
  p=0
  for m,w in W.items():
   arr=id_member if (use_id and tag=='e1Aet' and m=='v1_allfam_oh') else np.load(C/f'{tag}_{m}.npz')['oof']
   p+=w*rk(arr)
  br[tag]=p
 b=(br['old']+br['e1']+br['e1et'])/3
 main=rk(.75*rk(b)+.25*br['e1Aet']);cb=np.zeros(len(y))
 for i,(_,v) in enumerate(folds,1):
  if use_cb: cb[v]=fresh_cb[v]
  else: cb[v]=np.load(C/f'catboost_product_fold_{i}.npz')['oof']
 return .90*main+.10*rk(cb)
base=formula(False,False);id_only=formula(True,False);ca_only=formula(False,True);combo=formula(True,True)
assert np.array_equal(y,np.load(H/'claude_7_matched_v15_specialist_oof.npz')['y'])
rows=[]
for name,p in [('ID_context_only',id_only),('fresh_CA_CatBoost_only',ca_only),('ID_plus_fresh_CA',combo)]:
 fd=[float(roc_auc_score(y[v],p[v])-roc_auc_score(y[v],base[v])) for _,v in folds]
 rows.append({'variant':name,'auc':float(roc_auc_score(y,p)),'delta_vs_v15':float(roc_auc_score(y,p)-roc_auc_score(y,base)),
              'positive_folds':sum(x>0 for x in fd),'fold_deltas':fd})
out=H/'v15_id_ca_combination';(out/'fresh_confirmation.json').write_text(json.dumps({'baseline_auc':float(roc_auc_score(y,base)),'candidate_ca_cv_seed':20261021,'candidate_ca_model_seed':2024,'variants':rows,'note':'CA CatBoost candidate predictions are out-of-fold under new split seed; ID member remains seed-42 matched; no submission.'},indent=2),encoding='utf-8')
pd.DataFrame(rows).to_csv(out/'fresh_confirmation.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))
