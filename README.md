# AFL-LightGBM: Adaptive Focal-Loss LightGBM for Mobile Money Fraud Detection in Ghana

An engineered LightGBM variant that **fabricates an adaptive focal loss objective**
(γ = log(N_neg / N_pos) / 2) inside LightGBM's custom-gradient API to detect fraud in
Ghanaian mobile money transaction data (<0.5% fraud prevalence), with SHAP-based
explainability aligned to Ghana's Data Protection Act (2012, Act 843).

**Author:** Caleb Kwabena Kyere Boateng (ID 20993061)  
**Supervisor:** Eric Opoku Osei (PhD)  
**Institution:** Department of Computer Science, KNUST, Ghana  
**Status:** Research proposal stage — scaffold only (experiments pending CHRPE ethics clearance)

---

## The adaptive focal loss (core contribution)

```python
import numpy as np

def afl_focal_obj(y_pred, dtrain, gamma):
    """Adaptive focal loss for LightGBM (returns grad, hess).
    gamma = log(N_neg / N_pos) / 2  — computed per fold.
    """
    p  = 1.0 / (1.0 + np.exp(-y_pred))
    y  = dtrain.get_label()
    pt = np.where(y == 1, p, 1.0 - p)
    grad = -((1 - pt) ** gamma) * (y - p) * (1 + gamma * np.log(pt + 1e-7))
    hess = p * (1 - p) * ((1 - pt) ** gamma)
    return grad, hess
```

## Research Questions
- **RQ1** — Does adaptive focal loss in AFL-LightGBM significantly improve AUC-PR over standard LightGBM?
- **RQ2** — How does AFL-LightGBM compare against LightGBM, XGBoost, Random Forest, and Logistic Regression (Wilcoxon signed-rank, 5-fold CV)?
- **RQ3** — Which transaction features most strongly predict fraud (SHAP), and how do they differ between PaySim and Ghanaian field data?

## Data
- **Public baseline:** PaySim synthetic mobile money dataset (~6.3M rows, 0.13% fraud)
  https://www.kaggle.com/datasets/ealaxi/paysim1
- **Primary field data:** De-identified Ghanaian MNO transaction logs (pending CHRPE ethics approval; not redistributed here)

## Reproducibility
- Fixed seed: `SEED = 42`
- 5-fold stratified CV + Optuna (1000 trials); baseline locked before engineering
- Primary metric: AUC-PR (not accuracy)
- Significance test: Wilcoxon signed-rank (two-sided, alpha=0.05), Cohen's d, 95% CI

## Repository structure (planned)
```
AFL-LightGBM-MoMo-Fraud/
├── README.md
├── requirements.txt
├── LICENSE
├── data/                           # NOT committed — see data/README for access
├── notebooks/
│   ├── 01_baseline_lightgbm.ipynb  # Phase 1 — LOCK the baseline, never touch again
│   ├── 02_afl_lightgbm.ipynb       # Phase 2 — engineered model on identical folds
│   └── 03_evaluation_shap.ipynb    # Phase 3 — comparison table + SHAP outputs
└── src/
    ├── afl_objective.py            # adaptive focal loss function
    ├── preprocess.py               # preprocessing pipeline (see 4E)
    └── evaluate.py                 # Wilcoxon tests, Cohen's d, AUC-PR
```

## How to run (once data is available post-ethics approval)
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_baseline_lightgbm.ipynb
```

## Ethics
Field data collection is covered by a KNUST CHRPE application submitted 25 June 2026.
Approval expected 16 July 2026. No raw participant data is stored in this repository.

## Citation
Zenodo DOI will be minted on first GitHub Release (after experiments complete). Cite as:
> Boateng, C. K. K. (2026). AFL-LightGBM: Adaptive focal-loss LightGBM for mobile money
> fraud detection in Ghana [Code]. GitHub. https://github.com/cLLeB/AFL-LightGBM-MoMo-Fraud

## License
MIT
