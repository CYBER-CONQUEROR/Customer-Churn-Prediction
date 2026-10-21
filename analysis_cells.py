# %% A. Reproducibility, source requirements and independent holdout
%matplotlib inline
from pathlib import Path
import json, sys, platform, subprocess, importlib.metadata as md
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, joblib
from IPython.display import display, Markdown, Image
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import *
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from churn_features import CustomerFeatures, NUM, CAT, ENG
SEED=42
CFG={'contact_cost':5,'retained_value':200,'success_rate':0.25,'capacity':0.2}
Path('figures').mkdir(exist_ok=True); Path('results').mkdir(exist_ok=True)
sns.set_theme(style='whitegrid',palette=['#197f86','#d48254'])
raw=pd.read_csv('customer_churn_advanced_dataset.csv')
assert raw.shape==(1200,20) and set(raw.churn)=={0,1} and raw.customer_id.is_unique
train_idx,test_idx=train_test_split(raw.index,test_size=.2,stratify=raw.churn,random_state=SEED)
train=raw.loc[train_idx].copy(); test=raw.loc[test_idx].copy()
X=train.drop(columns=['churn','customer_id']); y=train.churn
Xt=test.drop(columns=['churn','customer_id']); yt=test.churn
assert set(train_idx).isdisjoint(test_idx)
pd.DataFrame({'row':list(train_idx)+list(test_idx),'split':['train']*len(train_idx)+['test']*len(test_idx)}).to_csv('results/split.csv',index=False)
display(Markdown('One stratified 80/20 split is locked before analysis. EDA and all model decisions use the 960 training customers. The 240 test labels are used only after model and threshold selection. Seed: 42.'))
versions={p:md.version(p) for p in ['numpy','pandas','scikit-learn','matplotlib','seaborn','joblib','nbformat','nbclient','reportlab','pypdf','pdfplumber','ipykernel']}
display(versions)
Path('requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
# %% A. Audit and documented cleaning decisions
audit=pd.DataFrame({'dtype':raw.dtypes.astype(str),'missing':raw.isna().sum(),'missing_pct':raw.isna().mean()*100,'unique':raw.nunique()})
display(audit); display(raw[NUM].describe().T); display(raw.churn.value_counts().to_frame('count'))
print('Exact duplicates:',raw.duplicated().sum(),'Unique IDs:',raw.customer_id.nunique())
display({c:sorted(raw[c].dropna().unique().tolist()) for c in CAT})
domain={'age':(18,120),'tenure_months':(0,1200),'monthly_charges':(0,np.inf),'total_charges':(0,np.inf),'complaints_last_12m':(0,np.inf),'late_payments_last_12m':(0,np.inf),'satisfaction_score':(1,10),'monthly_usage_gb':(0,np.inf),'discount_pct':(0,100),'senior_citizen':(0,1)}
display(pd.Series({c:int(((raw[c]<lo)|(raw[c]>hi)).sum()) for c,(lo,hi) in domain.items()},name='domain_candidates'))
q1=train[NUM].quantile(.25); q3=train[NUM].quantile(.75); iqr=q3-q1
outliers=((train[NUM]<q1-1.5*iqr)|(train[NUM]>q3+1.5*iqr)).sum()
display(outliers.to_frame('training_IQR_flags'))
billing=train[['customer_id','tenure_months','monthly_charges','total_charges','monthly_usage_gb']].copy()
billing['historical_to_current_ratio']=billing.total_charges/(billing.monthly_charges*billing.tenure_months)
display(billing.nlargest(8,'monthly_charges')); display(billing.nlargest(8,'monthly_usage_gb')); display(billing.sort_values('historical_to_current_ratio').iloc[np.r_[0:5,-5:0]])
audit.to_csv('results/audit.csv'); billing.to_csv('results/billing_investigation.csv',index=False)
raw.to_csv('02_cleaned_dataset.csv',index=False)
display(Markdown('Cleaning decision: retain every row, identity, label, category and missing value. No confirmed invalid domain values or duplicates justify edits. The cleaned export is a validated, canonical CSV copy with unchanged values. Large charges/usage and discrepancies between historical charges and current monthly price times tenure are investigation candidates: discounts, price changes or billing errors cannot be distinguished without invoices. No caps, row deletions or inferred repairs. Missing values are imputed only inside training folds. Observed category vocabularies are reported; no external category dictionary was supplied. Timing of measurements and next-cycle target definition must be verified with the data owner.'))
# %% B. Training-only exploratory analysis and interpretations
fig_notes=[]
def savefig(name,note):
    plt.tight_layout(); plt.savefig('figures/'+name+'.png',dpi=180,bbox_inches='tight'); plt.show(); plt.close()
    fig_notes.append((name,note)); display(Markdown(note))
counts=train.churn.value_counts().sort_index(); counts.plot.bar(figsize=(7,3),rot=0,color=['#197f86','#d48254']); plt.title('Training target balance'); plt.xlabel('Churn'); plt.ylabel('Customers')
savefig('target',f'Training churn prevalence is {y.mean():.2%}. A classifier predicting churn for everyone already achieves this accuracy; random ranking AP is prevalence.')
eda=train.copy(); eda['tenure_band']=pd.cut(eda.tenure_months,[0,12,24,48,np.inf],labels=['0-12','13-24','25-48','49+'])
for col in ['contract_type','tenure_band','payment_method','internet_service','support_plan','tech_support','complaints_last_12m','late_payments_last_12m']:
    g=eda.groupby(col,observed=True).churn.agg(['mean','size']); display(g)
    g['mean'].plot.bar(figsize=(7,3),rot=20); plt.ylim(0,1.05); plt.ylabel('Observed churn rate'); plt.title('Churn by '+col.replace('_',' '))
    savefig(col,f'{col}: highest observed rate {g["mean"].idxmax()} ({g["mean"].max():.1%}); lowest {g["mean"].idxmin()} ({g["mean"].min():.1%}). Group counts above qualify these descriptive associations; they do not establish intervention effects.')
for col in ['monthly_charges','satisfaction_score','monthly_usage_gb','discount_pct']:
    sns.boxplot(data=eda,x='churn',y=col,showfliers=True); plt.title(col.replace('_',' ').title()+' by churn')
    med=eda.groupby('churn')[col].median()
    savefig(col,f'Median {col}: non-churn {med[0]:.2f}, churn {med[1]:.2f}. Overlap and extreme observations limit simple cutoffs; missing observations are omitted in this chart.')
sns.heatmap(train[NUM].corr(),cmap='vlag',center=0); plt.gcf().set_size_inches(10,7); plt.title('Predictor correlations (training customers)'); savefig('correlation','Correlated billing and tenure measures may share information. Marginal correlation does not describe nonlinear predictive relationships or causality.')
display(train.assign(low_satisfaction=train.satisfaction_score<5).groupby(['contract_type','low_satisfaction']).churn.agg(['mean','size']))
display(pd.DataFrame({c:train.groupby(train[c].isna()).churn.mean() for c in ['monthly_charges','total_charges','satisfaction_score','monthly_usage_gb']}))
display(Markdown('Interaction and missingness tables are exploratory training summaries. Small groups are unstable. Tree models allow nonlinearities; engineered interactions are compared by CV. Missingness may reflect collection practices and must be monitored.'))
# %% C-D. Serializable features, pipelines, baselines and efficient five-fold tuning
feature_rationale={'tenure_band':'Lifecycle groups, fixed boundaries; available at scoring','charge_per_tenure':'Current price relative to relationship length; denominator >=1','complaint_intensity':'Complaints / min(tenure,12); assumes full observed history','late_payment_intensity':'Late payments / min(tenure,12); same exposure assumption','dissatisfaction_index':'10 - satisfaction, preserving missingness','service_support_gap':'Internet subscription lacking either support service','usage_to_charge_ratio':'Usage / positive monthly charge; otherwise missing','dissatisfaction_complaint_interaction':'Joint dissatisfaction and complaint burden'}
display(feature_rationale)
def pipe(model,eng=True):
    num=NUM+(ENG if eng else []); cat=CAT+(['tenure_band'] if eng else [])
    pre=ColumnTransformer([('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
    return Pipeline([('features',CustomerFeatures(eng)),('preprocess',pre),('model',model)])
cv=StratifiedKFold(5,shuffle=True,random_state=SEED)
def top_precision(est,x,ytrue):
    p=est.predict_proba(x)[:,1]; k=int(np.ceil(.2*len(p))); return np.asarray(ytrue)[np.argsort(-p,kind='stable')[:k]].mean()
scoring={'ap':'average_precision','roc':'roc_auc','recall':'recall','f1':'f1','top20':top_precision}
candidates={'Logistic':(LogisticRegression(max_iter=2000,random_state=SEED),{'model__C':[.1,1,10],'model__class_weight':[None,'balanced']}),'Forest':(RandomForestClassifier(n_estimators=120,n_jobs=1,random_state=SEED),{'model__max_depth':[4,None],'model__min_samples_leaf':[2,8],'model__class_weight':[None,'balanced']}),'Boosting':(GradientBoostingClassifier(random_state=SEED),{'model__n_estimators':[60,120],'model__learning_rate':[.03,.1],'model__max_depth':[1,2]})}
rows=[]; fitted={}; spaces={}
for name,(model,grid) in {'Dummy':(DummyClassifier(strategy='most_frequent'),{}),**candidates}.items():
    base=pipe(model); scores=cross_validate(base,X,y,cv=cv,scoring=scoring,n_jobs=1)
    rows.append({'model':name,'version':'baseline',**{f'{m}_{s}':float(np.mean(scores['test_'+m]) if s=='mean' else np.std(scores['test_'+m],ddof=1)) for m in scoring for s in ['mean','std']}})
    if grid:
        search=GridSearchCV(base,grid,cv=cv,scoring=scoring,refit='ap',n_jobs=1); search.fit(X,y); fitted[name]=search.best_estimator_; spaces[name]=search.best_params_
        i=search.best_index_; rows.append({'model':name,'version':'tuned',**{f'{m}_{s}':float(search.cv_results_[f'{s}_test_{m}'][i]) for m in scoring for s in ['mean','std']}})
        pd.DataFrame(search.cv_results_).to_csv(f'results/search_{name}.csv',index=False)
comparison=pd.DataFrame(rows); display(comparison); comparison.to_csv('results/model_comparison.csv',index=False); display(spaces)
winner=comparison.query("version=='tuned'").sort_values(['ap_mean','top20_mean'],ascending=False).iloc[0]['model']; final=fitted[winner]
ablation=[]
for eng in [False,True]:
    est=pipe(final.named_steps['model'],eng); s=cross_validate(est,X,y,cv=cv,scoring=scoring)
    ablation.append({'engineered':eng,'ap_mean':np.mean(s['test_ap']),'ap_std':np.std(s['test_ap']),'top20':np.mean(s['test_top20'])})
display(pd.DataFrame(ablation))
use_eng=ablation[1]['ap_mean']>=ablation[0]['ap_mean']; final=pipe(final.named_steps['model'],use_eng)
print('Selected by CV average precision, with top-20 precision as tie break:',winner,'Engineered:',use_eng)
display(Markdown('Five-fold training CV selects AP because ranking drives retention. ROC, recall, F1 and top-20 precision provide context. Class weighting is searched for Logistic and Forest; no resampling is used. CV standard deviations are a robustness check, not confidence intervals. Searching and comparing on the same folds makes winning CV results optimistic; the locked test gives the final independent estimate. All eight features are pre-outcome hypotheses whose timing requires operational validation.'))
# %% E. Training-only threshold choice, then final held-out evaluation
oof=cross_val_predict(final,X,y,cv=cv,method='predict_proba')[:,1]
thresholds=np.unique(np.r_[0,np.linspace(.05,.99,95),1])
threshold_rows=[]
for t in thresholds:
    pred=oof>=t; tn,fp,fn,tp=confusion_matrix(y,pred).ravel()
    threshold_rows.append({'threshold':t,'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred),'f1':f1_score(y,pred),'utility':tp*CFG['success_rate']*CFG['retained_value']-pred.sum()*CFG['contact_cost'],'fp':fp,'fn':fn})
tt=pd.DataFrame(threshold_rows); threshold=float(tt.sort_values(['utility','threshold'],ascending=[False,False]).iloc[0].threshold)
display(tt.iloc[::10]); print('Locked training OOF utility threshold:',threshold)
final.fit(X,y); p=final.predict_proba(Xt)[:,1]; pred=p>=threshold
metrics={'accuracy':accuracy_score(yt,pred),'precision':precision_score(yt,pred,zero_division=0),'recall':recall_score(yt,pred),'f1':f1_score(yt,pred),'roc_auc':roc_auc_score(yt,p),'average_precision':average_precision_score(yt,p),'brier':brier_score_loss(yt,p),'test_prevalence':yt.mean(),'threshold':threshold}
display(metrics); print(classification_report(yt,pred,zero_division=0)); display(pd.DataFrame(confusion_matrix(yt,pred),index=['actual 0','actual 1'],columns=['pred 0','pred 1']))
test_threshold_table=pd.DataFrame([{'threshold':float(t),'precision':precision_score(yt,p>=t,zero_division=0),'recall':recall_score(yt,p>=t),'flagged':int((p>=t).sum())} for t in sorted(set([.1,.3,.5,.7,.9,.95,threshold]))])
display(test_threshold_table); test_threshold_table.to_csv('results/test_thresholds_descriptive.csv',index=False)
display(Markdown('This test threshold sweep is descriptive only; the locked OOF threshold is unchanged. Under the assumed generous benefits, the fixed threshold can flag nearly everyone. The 20% capacity queue remains the operational outreach policy.'))
fig,ax=plt.subplots(1,3,figsize=(15,4)); RocCurveDisplay.from_predictions(yt,p,ax=ax[0]); PrecisionRecallDisplay.from_predictions(yt,p,ax=ax[1]); ax[1].axhline(yt.mean(),ls='--',color='gray'); frac,mp=calibration_curve(yt,p,n_bins=6,strategy='quantile'); ax[2].plot(mp,frac,'o-'); ax[2].plot([0,1],[0,1],'--'); ax[2].set(title='Calibration (test)',xlabel='Mean predicted probability',ylabel='Observed fraction'); savefig('evaluation','Test curves are descriptive after all decisions are locked. AP must exceed the high prevalence benchmark to add ranking value. Calibration bins contain few non-churners; reliability estimates are uncertain.')
tt.plot(x='threshold',y=['precision','recall'],figsize=(7,3)); plt.title('Training OOF threshold behavior'); savefig('thresholds','False negatives miss potential retention value; false positives incur contact cost. Utility uses hypothetical homogeneous treatment effects and ignores capacity. The capacity policy is evaluated separately. OOF predictions are conditional on CV-selected hyperparameters and may retain selection optimism.')
print('Constant prevalence Brier benchmark:',brier_score_loss(yt,np.repeat(y.mean(),len(yt))))
joblib.dump(final,'03_model_artifact.joblib')
pd.DataFrame({'customer_id':test.customer_id,'actual':yt,'probability':p}).to_csv('results/test_predictions.csv',index=False)
# %% F. Global permutation and three real local risk explanations
imp=permutation_importance(final,Xt,yt,scoring='average_precision',n_repeats=8,random_state=SEED,n_jobs=1)
importance=pd.DataFrame({'feature':Xt.columns,'AP_drop':imp.importances_mean,'std':imp.importances_std}).sort_values('AP_drop',ascending=False)
display(Markdown(f'The strongest held-out ranking signal is {importance.iloc[0].feature}, with AP loss {importance.iloc[0].AP_drop:.3f} when permuted. Weak importance for other inputs means their subgroup associations should not automatically become targeting rules.'))
display(importance); importance.to_csv('results/importance.csv',index=False)
importance.head(10).sort_values('AP_drop').plot.barh(x='feature',y='AP_drop',xerr='std',figsize=(8,4)); plt.title('Raw-feature permutation importance (held-out)'); savefig('importance','Permuting raw columns aggregates categorical levels and propagates through engineered features. Correlated predictors can mask importance; test importances are explanatory and never fed back into selection.')
local=[]
for profile,pos in [('low',np.argmin(p)),('medium',np.argsort(p)[len(p)//2]),('high',np.argmax(p))]:
    row=Xt.iloc[[pos]].copy(); effects=[]
    for col in Xt.columns:
        changed=row.copy(); changed[col]=X[col].median() if col in NUM else X[col].mode().iloc[0]
        effects.append((col,float(p[pos]-final.predict_proba(changed)[:,1][0])))
    effects=sorted(effects,key=lambda z:abs(z[1]),reverse=True)[:4]
    item={'profile':profile,'customer_id':test.iloc[pos].customer_id,'probability':float(p[pos]),'reference_replacement_deltas':effects,'raw_record':row.iloc[0].to_dict()}; local.append(item); display(item)
display(Markdown('Local explanations replace one raw attribute with its training median/mode and recompute risk, including dependent engineered features. Deltas measure model sensitivity relative to that reference, are not additive attribution, may create implausible combinations, and are not causal effects. Medium means median predicted rank in this high-prevalence cohort.'))
# %% G. Exact capacity targeting and hypothetical economics
ranked=pd.DataFrame({'customer_id':test.customer_id.to_numpy(),'actual':yt.to_numpy(),'probability':p}).sort_values(['probability','customer_id'],ascending=[False,True])
k=int(np.ceil(CFG['capacity']*len(test))); reached=int(ranked.head(k).actual.sum()); random_reached=k*yt.mean()
target={'contacted':k,'true_churners_reached':reached,'precision_at_20':reached/k,'recall_at_20':reached/yt.sum(),'lift':(reached/k)/yt.mean(),'random_expected_churners':random_reached,'expected_retained':reached*CFG['success_rate'],'cost':k*CFG['contact_cost'],'gross_value':reached*CFG['success_rate']*CFG['retained_value'],'net_benefit':reached*CFG['success_rate']*CFG['retained_value']-k*CFG['contact_cost'],'incremental_net_vs_random':(reached-random_reached)*CFG['success_rate']*CFG['retained_value']}
display(target); ranked.assign(contact=np.arange(len(ranked))<k).to_csv('results/targeting.csv',index=False)
sensitivity=pd.DataFrame([{'success_rate':s,'retained_value':v,'model_net':reached*s*v-k*CFG['contact_cost'],'random_net':random_reached*s*v-k*CFG['contact_cost'],'incremental':(reached-random_reached)*s*v} for s in [.1,.25,.4] for v in [100,200,400]])
display(sensitivity); sensitivity.to_csv('results/sensitivity.csv',index=False)
plt.figure(figsize=(7,3)); plt.plot(np.arange(1,len(test)+1)/len(test),ranked.actual.cumsum()/yt.sum(),label='Model ranking'); plt.plot([0,1],[0,1],'--',label='Random expectation'); plt.axvline(.2,color='gray'); plt.xlabel('Fraction contacted'); plt.ylabel('Fraction of churners captured'); plt.title('Held-out cumulative gains'); plt.legend(); savefig('gains','Uniform random contact has expected precision equal to held-out prevalence and expected recall equal to contact fraction. Deterministic ties use customer ID only after prediction. The model cannot contact more than 48 test customers.')
rng=np.random.default_rng(SEED); random_captures=np.array([rng.choice(yt.to_numpy(),k,replace=False).sum() for _ in range(2000)])
print('Random capture 2.5/50/97.5 percentiles:',np.quantile(random_captures,[.025,.5,.975]))
display(Markdown('These are out-of-sample targeting diagnostics and hypothetical deployment economics, not observed retention outcomes. Success rates, homogeneous value and treatment effects are assumed. Random selection uncertainty, customer consent, operational costs and effect heterogeneity can materially change value. Run a randomized retention trial before claiming ROI.'))
# %% H. Business conclusions, report generation and reproducibility
recommendations=[f'Pilot the top-risk queue: contact {k} of {len(test)} comparable customers; measured capture is {reached} churners versus {random_reached:.1f} expected randomly.', 'Use the contract-rate evidence to test renewal reminders and contract-fit conversations; measure incremental retention by randomized assignment.', 'Use satisfaction and complaint evidence to triage service recovery; test response quality and timing rather than assuming a causal benefit.', 'Use late-payment and payment-method evidence to test opt-in billing assistance; avoid punitive targeting or sensitive financial profiling.', 'Use support and usage evidence to test relevant support onboarding and plan-fit reviews. Audit subgroup access and outcomes, including age and region.']
display(Markdown('\n\n'.join(recommendations)))
display(Markdown('Limitations: only 1,200 records, no timestamps, uncertain sampling/target provenance and feature timing, unusual churn prevalence, only 33 non-churners in the test set, one random split and no temporal/external validation. Protected/proxy variables require fairness review. Predictions can drift; monitor missingness, calibration, outreach burden and incremental effects. The selected model follows CV AP, with feature ablation on training folds; test outcomes are reported once.'))
summary={'winner':winner,'engineered':bool(use_eng),'metrics':{a:float(b) for a,b in metrics.items()},'targeting':{a:float(b) for a,b in target.items()},'config':CFG,'local':local,'recommendations':recommendations,'versions':versions}
Path('results/summary.json').write_text(json.dumps(summary,indent=2,default=str))
from report import create_report
create_report(summary,comparison,audit,fig_notes,importance,sensitivity)
print('Report and model saved. Full submission checks run after notebook execution.')
