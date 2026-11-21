import os, time, json, hashlib, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')
ROOT='/mnt/data'; OUT=ROOT+'/GCI_v33_product_conditional_candidate'; os.makedirs(OUT,exist_ok=True); t0=time.time()
tr=pd.read_csv(ROOT+'/train.csv'); te=pd.read_csv(ROOT+'/test.csv'); sample=pd.read_csv(ROOT+'/sample_submission.csv')
y=tr.TARGET.to_numpy(); n=len(tr); raw=pd.concat([tr.drop(columns=['TARGET','SK_ID_CURR']),te.drop(columns=['SK_ID_CURR'])],ignore_index=True)
# Rebuild same formula-aware scaffold used in the confirmed research screen.
base=raw.copy(); base['DAYS_EMPLOYED_SENTINEL']=(base.DAYS_EMPLOYED==365243).astype('int8'); base['DAYS_EMPLOYED_CLEAN']=base.DAYS_EMPLOYED.where(base.DAYS_EMPLOYED!=365243); base['AGE_YEARS']=-base.DAYS_BIRTH/365.25
for a,b in [('AMT_CREDIT','AMT_INCOME_TOTAL'),('AMT_ANNUITY','AMT_INCOME_TOTAL'),('AMT_ANNUITY','AMT_CREDIT'),('AMT_GOODS_PRICE','AMT_CREDIT'),('AMT_GOODS_PRICE','AMT_INCOME_TOTAL')]: base[f'{a}_DIV_{b}']=base[a]/(base[b].abs()+1)
base['INCOME_PER_FAMILY']=base.AMT_INCOME_TOTAL/(base.CNT_FAM_MEMBERS.fillna(1)+1e-3); base['EXT_MEAN']=base[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].mean(axis=1); base['EXT_MIN']=base[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].min(axis=1); base['EXT_STD']=base[['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']].std(axis=1)
for c in base.select_dtypes(include='object').columns: base[c]=base[c].fillna('__MISSING__').astype(str)
for c in base.select_dtypes(include=[np.number]).columns:
 if base[c].isna().any(): base[c+'_MISS']=base[c].isna().astype('int8')
credit=raw.AMT_CREDIT.replace(0,np.nan); ann=raw.AMT_ANNUITY; cash=raw.NAME_CONTRACT_TYPE.eq('Cash loans'); qraw=ann/credit; qdec=pd.Series(np.where(cash,(ann-129.5)/credit,qraw),index=raw.index).replace([np.inf,-np.inf],np.nan); q4=qdec.round(4); q3=qdec.round(3); q5=qdec.round(5)
qfreq=q5.value_counts(); centers=np.sort(qfreq[qfreq>=30].index.to_numpy(float)); qv=qdec.to_numpy(float); dist=np.full(len(raw),np.nan); nearest=np.full(len(raw),np.nan); ok=np.isfinite(qv); ix=np.searchsorted(centers,qv[ok]); lo=np.clip(ix-1,0,len(centers)-1); hi=np.clip(ix,0,len(centers)-1); dl=np.abs(qv[ok]-centers[lo]); dh=np.abs(qv[ok]-centers[hi]); ch=dh<dl; dist[ok]=np.where(ch,dh,dl); nearest[ok]=np.where(ch,centers[hi],centers[lo]); fee_used=np.where(cash,129.5,0.); resid=ann.to_numpy()-np.where(np.isfinite(nearest),nearest,qdec.to_numpy(float))*credit.to_numpy()-fee_used
kg=(raw.AMT_CREDIT/raw.AMT_GOODS_PRICE.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan); kg5=kg.round(5); kgf=kg5.value_counts(); kgcent=np.sort(kgf[kgf>=30].index.to_numpy(float)); kgv=kg.to_numpy(float); kgdist=np.full(len(raw),np.nan); kgnear=np.full(len(raw),np.nan); ok=np.isfinite(kgv); ix=np.searchsorted(kgcent,kgv[ok]); lo=np.clip(ix-1,0,len(kgcent)-1); hi=np.clip(ix,0,len(kgcent)-1); dl=np.abs(kgv[ok]-kgcent[lo]); dh=np.abs(kgv[ok]-kgcent[hi]); ch=dh<dl; kgdist[ok]=np.where(ch,dh,dl); kgnear[ok]=np.where(ch,kgcent[hi],kgcent[lo])
formula=pd.DataFrame({'Q_RAW':qraw,'Q_DEC':qdec,'Q_RES5':(qdec-q5).abs(),'Q_RES4':(qdec-q4).abs(),'Q_RES3':(qdec-q3).abs(),'Q_NEAREST_DIST':dist,'Q_SUPPORT_LOG':np.log1p(q5.map(qfreq).to_numpy(float)),'ANN_RESID':resid,'ANN_RESID_ABS':np.abs(resid),'Q_FEE_EFFECT':qraw-qdec,'K_NEAR_DIST':kgdist,'K_NEAREST':kgnear,'K_SUPPORT':np.log1p(kg5.map(kgf).to_numpy(float)),'ANN_RESID_X_EXTMEAN':np.abs(resid)*base.EXT_MEAN.to_numpy(),'ANN_RESID_X_INCOME':np.abs(resid)/(raw.AMT_INCOME_TOTAL.to_numpy()+1)})
X=pd.concat([base.reset_index(drop=True),formula],axis=1).replace([np.inf,-np.inf],np.nan)
for c in X.select_dtypes(include='object').columns: X[c]=pd.Categorical(X[c],categories=pd.Index(X[c].unique()))
# Product-conditioned categories + formula/burden interactions from the repeatable screen.
def bin_q(s,nq): return pd.qcut(pd.Series(s).rank(method='first'),q=nq,labels=False,duplicates='drop').fillna(-1).astype(int).astype(str).to_numpy()
def combo(a,b): return (pd.Series(a).astype(str)+'|'+pd.Series(b).astype(str)).to_numpy(dtype=object)
q4k='Q4:'+q4.fillna(-999).astype(str).to_numpy(); q3k='Q3:'+q3.fillna(-999).astype(str).to_numpy(); contract='CT:'+raw.NAME_CONTRACT_TYPE.fillna('__MISSING__').astype(str).to_numpy(); inc='INC:'+raw.NAME_INCOME_TYPE.fillna('__MISSING__').astype(str).to_numpy(); occ='OCC:'+raw.OCCUPATION_TYPE.fillna('__MISSING__').astype(str).to_numpy(); edu='EDU:'+raw.NAME_EDUCATION_TYPE.fillna('__MISSING__').astype(str).to_numpy(); burden=raw.AMT_ANNUITY/(raw.AMT_INCOME_TOTAL+1); burdenbin='BURDEN:'+bin_q(burden,8); residbin='FRES:'+bin_q(pd.Series(np.abs(resid)),6)
keys={'q4_income':combo(q4k,inc),'q4_occupation':combo(q4k,occ),'q4_education':combo(q4k,edu),'q4_formula_resid':combo(q4k,residbin),'q4_burden':combo(q4k,burdenbin),'q4_contract':combo(q4k,contract),'q3_contract':combo(q3k,contract)}
# Train rows get five-fold out-of-fold target encodings; test rows use full-train smoothed maps.
folds=list(StratifiedKFold(5,shuffle=True,random_state=117).split(np.zeros(n),y)); alpha={'q4_income':250,'q4_occupation':250,'q4_education':250,'q4_formula_resid':250,'q4_burden':250,'q4_contract':200,'q3_contract':300}
TEtr={}; TEte={}
def enc_map(ktr, ytr, kval, a):
 g=pd.DataFrame({'k':np.asarray(ktr,dtype=object),'y':ytr}).groupby('k',dropna=False).y.agg(['sum','count']); prior=float(np.mean(ytr)); mp=((g['sum']+a*prior)/(g['count']+a)).to_dict(); cnt=g['count'].to_dict(); s=pd.Series(np.asarray(kval,dtype=object)); return s.map(mp).fillna(prior).to_numpy(np.float32), np.log1p(s.map(cnt).fillna(0).to_numpy(np.float32))
for name,kall in keys.items():
 kval=np.asarray(kall,dtype=object); enc=np.zeros(n,np.float32); cnt=np.zeros(n,np.float32)
 for aidx,bidx in folds:
  ev,cv=enc_map(kval[aidx],y[aidx],kval[bidx],alpha[name]); enc[bidx]=ev; cnt[bidx]=cv
 et,ct=enc_map(kval[:n],y,kval[n:],alpha[name]); TEtr[name+'_TE']=enc; TEtr[name+'_LOGCOUNT']=cnt; TEte[name+'_TE']=et; TEte[name+'_LOGCOUNT']=ct
Xtr=pd.concat([X.iloc[:n].reset_index(drop=True),pd.DataFrame(TEtr)],axis=1); Xte=pd.concat([X.iloc[n:].reset_index(drop=True),pd.DataFrame(TEte)],axis=1)
params=dict(n_estimators=420,learning_rate=.035,num_leaves=31,min_child_samples=120,max_bin=127,subsample=.85,subsample_freq=1,colsample_bytree=.85,reg_lambda=3,reg_alpha=.05,objective='binary',verbosity=-1,n_jobs=3,random_state=117)
model=LGBMClassifier(**params); model.fit(Xtr,y,categorical_feature='auto'); p=model.predict_proba(Xte)[:,1]
champ=pd.read_csv(ROOT+'/GCI_submission_v15_rankspace_consistent_view.csv'); assert len(champ)==len(sample)==len(p) and np.array_equal(champ.SK_ID_CURR.to_numpy(),sample.SK_ID_CURR.to_numpy())
# Small exploratory blend only: branch is complementary in compact OOF, but not yet validated against v15 OOF.
blendw=0.005; cp=rankdata(champ.TARGET.to_numpy(),method='average')/len(champ); pp=rankdata(p,method='average')/len(p); out=(1-blendw)*cp+blendw*pp
sub=sample.copy(); sub['TARGET']=out; path=OUT+'/GCI_v33_product_conditional_0p5pct_blend.csv'; sub.to_csv(path,index=False)
# model-only diagnostic as well
sm=sample.copy(); sm['TARGET']=p; sm.to_csv(OUT+'/GCI_v33_product_conditional_model_only.csv',index=False)
meta={'rows':len(sub),'features':int(Xtr.shape[1]),'blend_weight':blendw,'train_rows':n,'test_rows':len(te),'runtime_sec':round(time.time()-t0,2),'model_auc_oof_not_computed':True,'status':'exploratory; no direct champion OOF comparison available','note':'Feature mechanisms had positive compact-model OOF results on fresh seed, but this submission blend is not proven to improve public score.'}
json.dump(meta,open(OUT+'/metrics.json','w'),indent=2)
for f in [path,OUT+'/GCI_v33_product_conditional_model_only.csv']:
 print(f,os.path.getsize(f),hashlib.sha256(open(f,'rb').read()).hexdigest())
print(json.dumps(meta,indent=2))
