"""Build and execute the submission from raw sources. Python 3.12."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient
import sys, subprocess, json
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parent
sections=[
('Customer Churn Prediction and Retention Analytics','Objective: rank next-cycle churn risk under a 20% contact budget. Observational predictions guide experiments; intervention effects require validation.\n\nContents: A Audit; B EDA; C Features; D Models; E Evaluation; F Explanations; G Targeting; H Conclusions. The authoritative assignment is DS 01.pdf; all A-H and advanced requirements are implemented below.'),
]
cells=[nbf.v4.new_markdown_cell('# '+sections[0][0]+'\n\n'+sections[0][1])]
source=(ROOT/'analysis_cells.py').read_text(encoding='utf-8')
for block in source.split('# %% '):
    if not block.strip(): continue
    title,code=block.split('\n',1)
    cells += [nbf.v4.new_markdown_cell('## '+title), nbf.v4.new_code_cell(code)]
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=ROOT/'01_data_science_notebook.ipynb'
nbf.write(nb,path)
kernel_dir=ROOT/'.kernels'/'churn'
kernel_dir.mkdir(parents=True,exist_ok=True)
(kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Churn project','language':'python'}))
km=KernelManager(kernel_name='churn',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(ROOT/'.kernels')]))
NotebookClient(nb,km=km,timeout=1200,resources={'metadata':{'path':str(ROOT)}}).execute()
nbf.write(nb,path)
subprocess.run([sys.executable,'verify.py'],check=True,cwd=ROOT)
print('Executed notebook and passed submission checks.')
