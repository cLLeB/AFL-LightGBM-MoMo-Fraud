# AFL-LightGBM: Adaptive Focal-Loss-Engineered LightGBM for Mobile Money Fraud Detection in Ghana

An engineered LightGBM variant that fabricates an **adaptive focal loss** objective
(γ = log(N_neg / N_pos) / 2) inside LightGBM's custom-gradient API to detect fraud in
extremely imbalanced (<0.5% fraud) Ghanaian mobile money transaction data, with
SHAP-based explainability aligned to Ghana's Data Protection Act (2012, Act 843).

**Author:** Caleb Kwabena Kyere Boateng (ID 20993061)
**Supervisor:** Eric Opoku Osei (PhD)
**Institution:** Department of Computer Science, KNUST, Ghana
**Status:** Research proposal stage — code scaffold only (experiments pending ethics clearance)

---

## Research questions
- **RQ1** — Does the adaptive focal loss in AFL-LightGBM significantly improve AUC-PR over standard LightGBM?
- **RQ2** — How does AFL-LightGBM compare against LightGBM, XGBoost, Random Forest, and Logistic Regression (Wilcoxon signed-rank, 5-fold)?
- **RQ3** — Which transaction features most strongly predict fraud (SHAP), and how do they differ between PaySim and Ghanaian field data?

## The engineering contribution (F/M/R/S = Fabricate)
Standard LightGBM minimises binary cross-entropy. AFL-LightGBM replaces it with an
adaptive focal loss that down-weights easy negatives and concentrates gradient updates
on hard-to-classify fraud, with the focusing parameter γ set per fold from the imbalance ratio.

```python
import numpy as np

def afl_focal_obj(y_pred, dtrain, gamma):
    """Adaptive focal loss objective for LightGBM (returns grad, hess)."""
    p  = 1.0 / (1.0 + np.exp(-y_pred))
    y  = dtrain.get_label()
    pt = np.where(y == 1, p, 1.0 - p)
    grad = -((1 - pt) ** gamma) * (y - p) * (1 + gamma * np.log(pt + 1e-7))
    hess = p * (1 - p) * ((1 - pt) ** gamma)
    return grad, hess

# gamma is computed per fold:  gamma = np.log(N_neg / N_pos) / 2
```

## Data
- **Public baseline:** PaySim synthetic mobile money dataset (~6.3M rows, 0.13% fraud) — https://www.kaggle.com/datasets/ealaxi/paysim1
- **Primary field data:** de-identified Ghanaian MNO transaction logs (pending CHRPE ethics approval; not redistributed here)

## Reproducibility
- Fixed seed: `SEED = 42`
- 5-fold stratified CV + Optuna (1000 trials); locked baseline before engineering
- Metrics: AUC-PR (primary), AUC-ROC, Macro F1, Accuracy, training time
- Significance: Wilcoxon signed-rank (α = 0.05), Cohen's d, 95% CI

## Repository structure (planned)
```
AFL-LightGBM-MoMo-Fraud/
├── README.md
├── requirements.txt
├── LICENSE
├── data/            # NOT committed — see data/README for access
├── notebooks/
│   ├── 01_baseline_lightgbm.ipynb     # Phase 1 — locked baseline
│   ├── 02_afl_lightgbm.ipynb          # Phase 2 — engineered model
│   └── 03_evaluation_shap.ipynb       # Phase 3 — comparison + SHAP
└── src/
    ├── afl_objective.py               # adaptive focal loss
    ├── preprocess.py
    └── evaluate.py
```

## How to run (once data is available)
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_baseline_lightgbm.ipynb
```

## Ethics
Field data collection is covered by an application to the KNUST Committee on Human
Research, Publications and Ethics (CHRPE). No raw participant data is stored in this repo.

## Citation
If you use this work, please cite the proposal (Zenodo DOI to be added on first release).

## License
MIT — see `LICENSE`.
