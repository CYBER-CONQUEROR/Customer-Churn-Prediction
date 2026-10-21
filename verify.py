from pathlib import Path
import json, subprocess, sys
import numpy as np, pandas as pd, joblib, nbformat
from pypdf import PdfReader
from churn_features import NUM,CAT
root=Path(__file__).resolve().parent
raw=pd.read_csv(root/'customer_churn_advanced_dataset.csv'); cleaned=pd.read_csv(root/'02_cleaned_dataset.csv')
pd.testing.assert_frame_equal(raw,cleaned)
assert raw.shape==(1200,20) and set(raw.churn)=={0,1} and raw.customer_id.is_unique
split=pd.read_csv(root/'results/split.csv'); a=set(split.query("split=='train'").row); b=set(split.query("split=='test'").row)
assert len(a)==960 and len(b)==240 and not a&b
model=joblib.load(root/'03_model_artifact.joblib'); x=raw.loc[sorted(b)].drop(columns='churn')
assert not {'customer_id','churn'} & set(model.named_steps['features'].transform(x).columns)
assert np.isfinite(model.predict_proba(x)).all()
probe=x.head(3).copy(); probe[NUM]=np.nan; probe['contract_type']='UNSEEN'
assert np.isfinite(model.predict_proba(probe)).all()
np.testing.assert_allclose(model.predict_proba(x),model.predict_proba(x[x.columns[::-1]]))
saved=pd.read_csv(root/'results/test_predictions.csv').set_index('customer_id')
np.testing.assert_allclose(model.predict_proba(x)[:,1],saved.loc[x.customer_id,'probability'])
if '--child' not in sys.argv:
    subprocess.run([sys.executable,__file__,'--child'],check=True)
ranking=pd.read_csv(root/'results/targeting.csv'); top=ranking[ranking.contact]; s=json.loads((root/'results/summary.json').read_text()); t=s['targeting']
assert len(top)==int(np.ceil(.2*len(ranking)))==48
assert int(top.actual.sum())==t['true_churners_reached']
assert np.isclose(top.actual.mean(),t['precision_at_20']) and np.isclose(top.actual.sum()/ranking.actual.sum(),t['recall_at_20'])
assert np.isclose(top.actual.mean()/ranking.actual.mean(),t['lift'])
assert 5<=len(PdfReader(root/'04_business_report.pdf').pages)<=8
files=['01_data_science_notebook.ipynb','02_cleaned_dataset.csv','03_model_artifact.joblib','04_business_report.pdf','05_README.md']
assert all((root/f).stat().st_size>0 for f in files)
nb=nbformat.read(root/files[0],as_version=4)
assert all(c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs) for c in nb.cells if c.cell_type=='code')
assert sum('image/png' in o.get('data',{}) for c in nb.cells if c.cell_type=='code' for o in c.outputs)>=10
print('PASS: schema, preserved values, split independence, feature allowlist, missing/unseen inputs, order invariance, fresh-process reload parity, targeting arithmetic, PDF pages and all executed deliverables.')
