"""Matched v15 member replacement with LightGBM DART boosting."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
NB=ROOT/'pre info'/'GCI_FINAL_v15_end_to_end_colab.ipynb';CACHE=HERE/'v15_cache'/'gci_cache_final'
RESID=HERE/'v_3_cache'/'resid16.pkl';OUT=HERE/'v15_dart_member';OUT.mkdir(exist_ok=True)
W={'qf_shallow':.24,'qf_reg':.12,'v1_allfam_oh':.32,'v2_allfam_oh':.16,'v2_mono':.16}
def main():
 t0=time.time();nb=json.loads(NB.read_text(encoding='utf-8'));read=lambda i:''.join(nb['cells'][i].get('source',[]));ns={'__name__':'__v15_dart_experiment__'}
 setup=read(1)
 old=('base_dir = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else "/content"\n'
      'CACHE_DIR = os.path.join(base_dir, "gci_cache_final") if not SMOKE else "/tmp/gci_cache_smoke"; OUT_DIR = os.path.join(base_dir, "gci_output_final") if not SMOKE else "/tmp/gci_out_smoke"\n'
      'os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)')
 new=f'base_dir = {str(HERE)!r}\nCACHE_DIR = {str(CACHE)!r}; OUT_DIR = {str(OUT)!r}\nos.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)'
 if old not in setup:raise RuntimeError('Could not redirect notebook cache')
 exec(compile(setup.replace(old,new),str(NB)+'#cell1','exec'),ns);ns['CACHE_DIR'],ns['OUT_DIR'],ns['THREADS']=str(CACHE),str(OUT),4
 exec(compile(read(2),str(NB)+'#cell2','exec'),ns);ns['get_resid']=lambda seeds,name:pd.read_pickle(RESID)
 exec(compile(read(3),str(NB)+'#cell3','exec'),ns);exec(compile(read(4),str(NB)+'#cell4','exec'),ns)
 spec={}
 for name,(X,Xt,p,r,rf,seeds) in ns['SPEC1'].items():spec[name]=(X,Xt,p,r,rf,seeds)
 X,Xt,p,_,_,seeds=spec['v1_allfam_oh']
 dart={**p,'boosting_type':'dart','drop_rate':.10,'skip_drop':.50,'max_drop':50,'uniform_drop':False}
 dart.pop('extra_trees',None)
 spec['v1_allfam_oh']=(X,Xt,dart,1500,1650,seeds)
 print('Training 5-fold DART replacement for E1Aet v1_allfam_oh (1500 rounds; full refit 1650)',flush=True)
 test,oof=ns['run_member']('v1_allfam_oh',spec,'dart_candidate',ns['VIEW_A'])
 y=np.asarray(ns['y'],dtype=np.int8);n=len(y);folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y));rk=lambda x:rankdata(x,method='average')/len(x)
 def cache(tag,m):return np.load(CACHE/f'{tag}_{m}.npz')['oof'].copy()
 branch={}
 for tag in ('old','e1','e1et','e1Aet'):
  vals={m:rk(oof if tag=='e1Aet' and m=='v1_allfam_oh' else cache(tag,m)) for m in W}
  branch[tag]=sum(W[m]*vals[m] for m in W)
 b=(branch['old']+branch['e1']+branch['e1et'])/3
 cb=np.zeros(n)
 for i,(_,v) in enumerate(folds,1):cb[v]=np.load(CACHE/f'catboost_product_fold_{i}.npz')['oof']
 candidate=.90*rk(.75*rk(b)+.25*branch['e1Aet'])+.10*rk(cb)
 ref=np.load(HERE/'claude_7_matched_v15_specialist_oof.npz')
 if not np.array_equal(y,ref['y']) or not np.allclose(candidate if False else .90*rk(.75*rk((branch['old']+branch['e1']+branch['e1et'])/3)+.25*branch['e1Aet'])+.10*rk(cb),ref['v15_oof'],atol=1e-10):
  # Reconstruct explicit baseline separately to ensure cached rank formula alignment.
  base_branch={}
  for tag in ('old','e1','e1et','e1Aet'):
   vals={m:rk(cache(tag,m)) for m in W};base_branch[tag]=sum(W[m]*vals[m] for m in W)
  base=.90*rk(.75*rk((base_branch['old']+base_branch['e1']+base_branch['e1et'])/3)+.25*base_branch['e1Aet'])+.10*rk(cb)
  if not np.allclose(base,ref['v15_oof'],atol=1e-10):raise RuntimeError('Cached baseline formula mismatch')
 base_branch={}
 for tag in ('old','e1','e1et','e1Aet'):
  vals={m:rk(cache(tag,m)) for m in W};base_branch[tag]=sum(W[m]*vals[m] for m in W)
 base=.90*rk(.75*rk((base_branch['old']+base_branch['e1']+base_branch['e1et'])/3)+.25*base_branch['e1Aet'])+.10*rk(cb)
 rows=[]
 for i,(_,v) in enumerate(folds,1):
  a0=float(roc_auc_score(y[v],base[v]));a1=float(roc_auc_score(y[v],candidate[v]))
  rows.append({'fold':i,'v15_auc':a0,'dart_auc':a1,'delta_auc':a1-a0})
 result={'hypothesis':'DART dropout creates a useful ranking different from the incumbent E1Aet v1 all-family ExtraTrees member.',
  'candidate_member':'E1Aet/v1_allfam_oh','boosting_type':'dart','drop_rate':.10,'skip_drop':.50,'rounds':1500,'full_rounds':1650,
  'v15_oof_auc':float(roc_auc_score(y,base)),'candidate_oof_auc':float(roc_auc_score(y,candidate)),
  'delta_auc':float(roc_auc_score(y,candidate)-roc_auc_score(y,base)),'positive_folds':sum(x['delta_auc']>0 for x in rows),
  'folds':rows,'member_rank_spearman':float(spearmanr(oof,cache('e1Aet','v1_allfam_oh')).statistic),
  'elapsed_seconds':round(time.time()-t0,1),'submission_written':False}
 (OUT/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8');pd.DataFrame(rows).to_csv(OUT/'fold_metrics.csv',index=False)
 np.savez_compressed(OUT/'oof.npz',y=y,v15_oof=base,candidate_oof=candidate,member_oof=oof,candidate_test=test)
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
