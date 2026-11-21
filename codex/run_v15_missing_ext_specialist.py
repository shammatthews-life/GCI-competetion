"""Leakage-safe specialist for applicants missing at least two EXT_SOURCE values."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT=Path(__file__).resolve().parents[1]; HERE=Path(__file__).resolve().parent
NOTEBOOK=ROOT/'pre info'/'GCI_FINAL_v15_end_to_end_colab.ipynb'
CACHE=HERE/'v15_cache'/'gci_cache_final'; OUT=HERE/'v15_missing_ext_specialist_experiment'
OUT.mkdir(exist_ok=True)

def main():
    t=time.time(); nb=json.loads(NOTEBOOK.read_text(encoding='utf-8')); read=lambda i: ''.join(nb['cells'][i].get('source',[]))
    ns={'__name__':'__v15_missing_ext_specialist__'}; setup=read(1)
    old=('base_dir = "/content/drive/MyDrive" if os.path.isdir("/content/drive/MyDrive") else "/content"\n'
         'CACHE_DIR = os.path.join(base_dir, "gci_cache_final") if not SMOKE else "/tmp/gci_cache_smoke"; OUT_DIR = os.path.join(base_dir, "gci_output_final") if not SMOKE else "/tmp/gci_out_smoke"\n'
         'os.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)')
    new=f'base_dir = {str(HERE)!r}\nCACHE_DIR = {str(CACHE)!r}; OUT_DIR = {str(OUT)!r}\nos.makedirs(CACHE_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)'
    if old not in setup: raise RuntimeError('Could not safely redirect notebook cache')
    exec(compile(setup.replace(old,new),str(NOTEBOOK)+'#cell1','exec'),ns)
    ns['CACHE_DIR'],ns['OUT_DIR'],ns['THREADS']=str(CACHE),str(OUT),6
    exec(compile(read(2),str(NOTEBOOK)+'#cell2','exec'),ns)
    y=np.asarray(ns['y'],dtype=np.int8); n=len(y); ntr=int(ns['ntr']); al=ns['al']
    extmiss=al[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].iloc[:n].isna().sum(axis=1).to_numpy()
    segment=extmiss>=2
    Xa,Xat=ns['onehot'](ns['BASEF']['Xa'],ns['BASEF']['Xat'])
    Xa=Xa.reset_index(drop=True); yseg=float(y[segment].mean())
    cats=ns['KS']; product_keys={k:cats[k].iloc[:n].reset_index(drop=True) for k in ('Q3','Q4','Q5','P3','P4','P5','PF4')}
    folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(np.zeros(n),y))
    params={'objective':'binary','bagging_freq':1,'verbose':-1,'num_threads':6,'seed':42,
            'learning_rate':0.04,'num_leaves':15,'min_child_samples':150,'feature_fraction':0.3,
            'bagging_fraction':0.8,'lambda_l2':20,'extra_trees':True}
    rounds=881; pred=np.zeros(n); fold_rows=[]

    def enc(fit,qi):
        prior=float(y[fit].mean()); cols={}
        for k,key in product_keys.items():
            g=pd.DataFrame({'key':key.iloc[fit].to_numpy(),'y':y[fit]}).groupby('key',sort=False).y.agg(['sum','count'])
            q=key.iloc[qi]; cnt=q.map(g['count']).fillna(0).to_numpy(float); sm=q.map(g['sum']).fillna(0).to_numpy(float)
            m={'Q3':100.,'Q4':30.,'Q5':30.,'P3':100.,'P4':30.,'P5':30.,'PF4':30.}[k]
            cols[f'TE_{k}']=(sm+m*prior)/(cnt+m); cols[f'CNT_{k}']=cnt
        return pd.DataFrame(cols)

    def nested_enc(fit):
        out=np.zeros((len(fit),14),dtype=np.float32)
        splitter=StratifiedKFold(5,shuffle=True,random_state=7)
        for ia,ib in splitter.split(np.zeros(len(fit)),y[fit]):
            out[ib]=enc(fit[ia],fit[ib]).to_numpy(dtype=np.float32)
        return pd.DataFrame(out,columns=[x for k in product_keys for x in (f'TE_{k}',f'CNT_{k}')])

    for fi,(fit,valid) in enumerate(folds,1):
        fs=fit[segment[fit]]; vs=valid[segment[valid]]
        if len(fs)<1000 or len(vs)==0: raise RuntimeError(f'Insufficient segment rows in fold {fi}')
        Xfit=pd.concat([Xa.iloc[fs].reset_index(drop=True),nested_enc(fs)],axis=1)
        Xval=pd.concat([Xa.iloc[vs].reset_index(drop=True),enc(fs,vs)],axis=1)
        model=lgb.train(params,lgb.Dataset(Xfit,label=y[fs]),num_boost_round=rounds)
        pred[vs]=model.predict(Xval)
        fold_rows.append({'fold':fi,'fit_segment_rows':len(fs),'valid_segment_rows':len(vs),
                          'specialist_auc':float(roc_auc_score(y[vs],pred[vs])),
                          'elapsed_seconds':round(time.time()-t,1)})
        print(f'missing-EXT specialist fold {fi}/5 segment AUC={fold_rows[-1]["specialist_auc"]:.6f}',flush=True)

    basefile=np.load(HERE/'claude_7_matched_v15_specialist_oof.npz'); base=basefile['v15_oof']; y0=basefile['y']
    if not np.array_equal(y,y0): raise RuntimeError('v15 baseline row order/labels do not match train.csv')
    # Replace 10% of within-cohort rank ordering, scaled by cohort share so the
    # subgroup's mean rank and its broad placement stay anchored to v15.
    cand=base.copy(); share=float(segment.mean())
    for _,valid in folds:
        vs=valid[segment[valid]]
        rb=rankdata(base[vs],method='average')/len(vs)
        rs=rankdata(pred[vs],method='average')/len(vs)
        cand[vs]=base[vs]+0.10*share*(rs-rb)
    for fi,(_,valid) in enumerate(folds,1):
        fold_rows[fi-1].update({'v15_full_auc':float(roc_auc_score(y[valid],base[valid])),
          'candidate_full_auc':float(roc_auc_score(y[valid],cand[valid])),
          'delta_auc':float(roc_auc_score(y[valid],cand[valid])-roc_auc_score(y[valid],base[valid]))})
    result={'hypothesis':'A model trained only on applicants with at least two missing external scores can learn the thin-file risk regime more directly than the global v15 blend.',
      'segment':'count_missing(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) >= 2','segment_rows':int(segment.sum()),
      'segment_fraction':share,'segment_target_rate':yseg,'features':'v15 label-free Xa one-hot features plus nested fold-safe Q/P target encodings',
      'model':'v15 v1_allfam_oh LightGBM ExtraTrees member fixed parameters, 881 rounds','outer_folds':'StratifiedKFold(5, shuffle=True, random_state=42)',
      'rank_correction':'10% within-segment ordering replacement; correction magnitude scaled by full segment share',
      'v15_oof_auc':float(roc_auc_score(y,base)),'candidate_oof_auc':float(roc_auc_score(y,cand)),
      'delta_auc':float(roc_auc_score(y,cand)-roc_auc_score(y,base)),
      'positive_folds':int(sum(r['delta_auc']>0 for r in fold_rows)),'folds':fold_rows,
      'segment_specialist_spearman_vs_v15':float(spearmanr(pred[segment],base[segment]).statistic),
      'promotion_gate_delta_auc':0.002,'submission_written':False,'elapsed_seconds':round(time.time()-t,1)}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame(fold_rows).to_csv(OUT/'fold_metrics.csv',index=False)
    np.savez_compressed(OUT/'oof.npz',y=y,segment=segment,v15_oof=base,candidate_oof=cand,specialist_oof=pred)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
