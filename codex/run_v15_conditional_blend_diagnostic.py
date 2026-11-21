"""Fixed, label-free subgroup gates for the CA CatBoost replacement."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

H=Path(__file__).resolve().parent; C=H/'v15_cache'/'gci_cache_final'
z=np.load(H/'v15_ca_ext_catboost'/'oof.npz'); y=z['y']; cand_cb=z['candidate_catboost_oof']
train=pd.read_csv(H.parent/'inpuy'/'train.csv'); n=len(y)
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
rk=lambda x:rankdata(x,method='average')/len(x)
w={'qf_shallow':.24,'qf_reg':.12,'v1_allfam_oh':.32,'v2_allfam_oh':.16,'v2_mono':.16}; branches={}
for tag in ('old','e1','e1et','e1Aet'):
 p=np.zeros(n)
 for member,weight in w.items():p+=weight*rk(np.load(C/f'{tag}_{member}.npz')['oof'])
 branches[tag]=p
b=(branches['old']+branches['e1']+branches['e1et'])/3
main=rk(.75*rk(b)+.25*branches['e1Aet'])
old_cb=np.zeros(n)
for i,(_,v) in enumerate(folds,1):old_cb[v]=np.load(C/f'catboost_product_fold_{i}.npz')['oof']
baseline=.9*main+.1*rk(old_cb)
candidate_rank=rk(cand_cb)
assert np.allclose(baseline,np.load(H/'claude_7_matched_v15_specialist_oof.npz')['v15_oof'],atol=1e-10)

fields={}
for col in ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE']:
 fields[col]=train[col].astype('string').fillna('__MISSING__')
for col in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']:
 fields[col+'_MISSING']=train[col].isna().astype(str)
for col in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']:
 fields[col+'_DECILE']=pd.qcut(train[col].rank(method='average'),10,labels=False,duplicates='drop').astype('Int64').astype('string').fillna('MISSING')
fields['AGE_QUINTILE']=pd.qcut((-train.DAYS_BIRTH).rank(method='average'),5,labels=False,duplicates='drop').astype('Int64').astype('string')
credit_income=train.AMT_CREDIT/(train.AMT_INCOME_TOTAL.abs()+1)
fields['CREDIT_INCOME_QUINTILE']=pd.qcut(credit_income.rank(method='average'),5,labels=False,duplicates='drop').astype('Int64').astype('string')
ann_income=train.AMT_ANNUITY/(train.AMT_INCOME_TOTAL.abs()+1)
fields['ANNUITY_INCOME_QUINTILE']=pd.qcut(ann_income.rank(method='average'),5,labels=False,duplicates='drop').astype('Int64').astype('string')

rows=[]; delta_weight=.05
for field,series in fields.items():
 values=series.astype('string').fillna('__MISSING__').to_numpy()
 for value in pd.unique(values):
  mask=values==value; p=float(mask.mean())
  if mask.sum()<500 or np.unique(y[mask]).size<2:continue
  alpha=np.clip(.10+delta_weight*(mask.astype(float)-p),.02,.30)
  pred=(1-alpha)*main+alpha*candidate_rank
  fds=[]
  for fi,(_,valid) in enumerate(folds,1):
   fds.append(float(roc_auc_score(y[valid],pred[valid])-roc_auc_score(y[valid],baseline[valid])))
  rows.append({'field':field,'value':str(value),'n':int(mask.sum()),'prevalence':p,
    'weight_in_group':float(.10+delta_weight*(1-p)),
    'weight_outside':float(.10-delta_weight*p),
    'delta_auc':float(roc_auc_score(y,pred)-roc_auc_score(y,baseline)),
    'positive_folds':int(sum(d>0 for d in fds)),'fold_deltas':fds})
out=H/'v15_conditional_blend_diagnostic';out.mkdir(exist_ok=True)
df=pd.DataFrame(rows).sort_values('delta_auc',ascending=False)
df.to_csv(out/'all_gates.csv',index=False)
summary={'baseline_auc':float(roc_auc_score(y,baseline)),'candidate_fixed_10pct_auc':float(roc_auc_score(y,.9*main+.1*candidate_rank)),
 'candidate_fixed_10pct_delta':float(roc_auc_score(y,.9*main+.1*candidate_rank)-roc_auc_score(y,baseline)),
 'delta_weight':delta_weight,'gates_evaluated':len(df),
 'note':'Exploratory subgroup search over existing OOF predictions. Results are selected on reused labels and need frozen independent confirmation.',
 'top_gates':df.head(20).to_dict(orient='records')}
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(df.head(25)[['field','value','n','weight_in_group','delta_auc','positive_folds']].to_string(index=False))
