"""Combine the matched ID-context member and CA CatBoost branch candidates."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
H=Path(__file__).resolve().parent;C=H/'v15_cache'/'gci_cache_final'
id_z=np.load(H/'v15_id_context_experiment'/'oof.npz');ca_z=np.load(H/'v15_ca_ext_catboost'/'oof.npz')
y=id_z['y'];id_member=id_z['candidate_member_oof'];ca_cb=ca_z['candidate_catboost_oof']
rk=lambda x:rankdata(x,method='average')/len(x);W={'qf_shallow':.24,'qf_reg':.12,'v1_allfam_oh':.32,'v2_allfam_oh':.16,'v2_mono':.16}
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y))
def build(id_replace,ca_replace):
 br={}
 for tag in ('old','e1','e1et','e1Aet'):
  p=0
  for m,w in W.items():
   if tag=='e1Aet' and m=='v1_allfam_oh' and id_replace:raw=id_member
   else:raw=np.load(C/f'{tag}_{m}.npz')['oof']
   p=p+w*rk(raw)
  br[tag]=p
 b=(br['old']+br['e1']+br['e1et'])/3
 main=.90*rk(.75*rk(b)+.25*br['e1Aet'])
 cb=np.zeros(len(y))
 for i,(_,v) in enumerate(folds,1):
  fname='catboost_product_fold_'+str(i)+'.npz'
  cb[v]=ca_cb[v] if ca_replace else np.load(C/fname)['oof']
 return main+.10*rk(cb)
ref=np.load(H/'claude_7_matched_v15_specialist_oof.npz');base=build(False,False)
assert np.array_equal(y,ref['y']) and np.allclose(base,ref['v15_oof'],atol=1e-10)
variants={'ID_context_only':build(True,False),'CA_CatBoost_only':build(False,True),'ID_plus_CA':build(True,True)}
rows=[]
for name,p in variants.items():
 fd=[float(roc_auc_score(y[v],p[v])-roc_auc_score(y[v],base[v])) for _,v in folds]
 rows.append({'variant':name,'auc':float(roc_auc_score(y,p)),'delta_vs_v15':float(roc_auc_score(y,p)-roc_auc_score(y,base)),
              'positive_folds':sum(x>0 for x in fd),'fold_deltas':fd})
out=H/'v15_id_ca_combination';out.mkdir(exist_ok=True)
(out/'metrics.json').write_text(json.dumps({'v15_auc':float(roc_auc_score(y,base)),'variants':rows,'note':'Matched fixed v15 OOF rows; no weights tuned; no submission.'},indent=2),encoding='utf-8')
pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
np.savez_compressed(out/'oof.npz',y=y,v15_oof=base,**{k.lower()+'_oof':v for k,v in variants.items()})
print(pd.DataFrame(rows).to_string(index=False))
