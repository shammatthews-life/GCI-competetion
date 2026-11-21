"""Fold-safe Bayesian target rates for repeated applicant profiles."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; HERE=Path(__file__).resolve().parent
OUT=HERE/'v15_profile_target_rate_experiment'; OUT.mkdir(exist_ok=True)

def keys(frame, cols):
    z=frame[cols].astype('string').fillna('__MISSING__')
    return pd.util.hash_pandas_object(z,index=False).to_numpy(np.uint64)

def main():
    t=time.time(); df=pd.read_csv(ROOT/'inpuy'/'train.csv'); y=df.TARGET.to_numpy(np.int8); n=len(y)
    specs={
      'work_family':['CODE_GENDER','DAYS_BIRTH','DAYS_EMPLOYED','NAME_INCOME_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE','CNT_FAM_MEMBERS','NAME_FAMILY_STATUS','REGION_RATING_CLIENT'],
      'income_org_family':['CODE_GENDER','DAYS_BIRTH','NAME_INCOME_TYPE','ORGANIZATION_TYPE','CNT_FAM_MEMBERS','NAME_FAMILY_STATUS','REGION_RATING_CLIENT'],
      'occupation_family':['CODE_GENDER','DAYS_BIRTH','NAME_INCOME_TYPE','OCCUPATION_TYPE','CNT_FAM_MEMBERS','REGION_RATING_CLIENT']}
    kmap={name:keys(df,cols) for name,cols in specs.items()}
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
    pred=np.zeros(n); rows=[]
    for fi,(fit,val) in enumerate(folds,1):
        prior=float(y[fit].mean()); fold_scores=[]
        for name,k in kmap.items():
            train=pd.DataFrame({'k':k[fit],'y':y[fit]}).groupby('k',sort=False).y.agg(['sum','count'])
            kk=k[val]; sums=train['sum'].reindex(kk).fillna(0).to_numpy(); cnt=train['count'].reindex(kk).fillna(0).to_numpy()
            # Fixed 5-row prior keeps singleton profile estimates close to the fold rate.
            fold_scores.append((sums+5*prior)/(cnt+5))
        pred[val]=np.mean(fold_scores,axis=0)
        rows.append({'fold':fi,'profile_rate_auc':float(roc_auc_score(y[val],pred[val])),'rows':len(val),'elapsed_seconds':round(time.time()-t,1)})
        print(f'profile target-rate fold {fi}/5 AUC={rows[-1]["profile_rate_auc"]:.6f}',flush=True)
    z=np.load(HERE/'claude_7_matched_v15_specialist_oof.npz'); base=z['v15_oof']
    if not np.array_equal(y,z['y']): raise RuntimeError('Label alignment mismatch')
    cand=.99*rankdata(base,method='average')/n+.01*rankdata(pred,method='average')/n
    for fi,(_,val) in enumerate(folds):
        old=float(roc_auc_score(y[val],base[val])); new=float(roc_auc_score(y[val],cand[val]))
        rows[fi].update({'v15_auc':old,'candidate_auc':new,'delta_auc':new-old})
    result={'hypothesis':'Direct smoothed default rates for repeated applicant profiles expose cohort risk more efficiently than adding raw profile categories to CatBoost.',
      'profiles':specs,'smoothing_prior_rows':5,'profile_score':'mean of the three fold-safe posterior rates','fixed_v15_blend_weight':.01,
      'v15_oof_auc':float(roc_auc_score(y,base)),'profile_rate_oof_auc':float(roc_auc_score(y,pred)),
      'candidate_oof_auc':float(roc_auc_score(y,cand)),'delta_auc':float(roc_auc_score(y,cand)-roc_auc_score(y,base)),
      'positive_folds':int(sum(r['delta_auc']>0 for r in rows)),'folds':rows,
      'spearman_profile_rate_vs_v15':float(spearmanr(pred,base).statistic),'elapsed_seconds':round(time.time()-t,1),'submission_written':False}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); pd.DataFrame(rows).to_csv(OUT/'fold_metrics.csv',index=False)
    np.savez_compressed(OUT/'oof.npz',y=y,v15_oof=base,profile_rate_oof=pred,candidate_oof=cand)
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__': main()
