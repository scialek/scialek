# Lifestyle DAG Causal Demo

Quick pipeline:

1.  pip install -r requirements.txt
2.  python generate_synthetic.py  # build LifestyleDAG.csv
3.  python normalize_data.py             # writes LifestyleDAG_norm.csv
4.  python estimate_multi_edge_betas.py  # writes LifestyleDAG_betas.csv

```powershell
python -m venv venv
.\venvin
elease (or .\venv\Scripts\activate on Windows)
pip install -r requirements.txt
python generate_synthetic.py
python normalize_data.py
python estimate_multi_edge_betas.py
```
