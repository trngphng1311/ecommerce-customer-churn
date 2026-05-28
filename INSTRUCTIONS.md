# Instructions — Quick Run Guide

Brief reproduce / extend guide. For the full project story, results, and limitations see [`README.md`](README.md).

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# macOS only — XGBoost needs OpenMP:
brew install libomp
```

## 2. Run the notebooks in order

| # | Notebook | Produces |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | EDA figures in `outputs/figures/` |
| 2 | `notebooks/02_feature_engineering.ipynb` | `data/processed/churn_features_*.csv` |
| 3 | `notebooks/03_modeling.ipynb` | Model + scaler + threshold in `outputs/models/`, SHAP plots, `data/processed/churn_scored.csv`, `dashboard.csv`, `shap_importance.csv` |

Run top‑to‑bottom in each notebook. Total runtime: ~3–5 min on a modern laptop (40k rows).

## 3. Rebuild / refresh the dashboard

Open `dashboard/tableau/Ecomerce.twb` in **Tableau Public (Desktop)**. The workbook is pre‑wired to `data/processed/dashboard.csv` — refresh the extract after re‑running the notebooks.

See [`dashboard/DASHBOARD_GUIDE.md`](dashboard/DASHBOARD_GUIDE.md) for the sheet list and what each chart shows.

## 4. Key design decisions (TL;DR)

- **Class imbalance** (~37 % churn) handled with `class_weight='balanced'` on Logistic Regression and `scale_pos_weight ≈ 1.7` on tree models.
- **Threshold tuned for recall** (0.462) — missing a churner costs lifetime revenue; a false alarm costs a single retention nudge.
- **Scaling applied only to LR**, not to tree models (trees are scale‑invariant). Scaler is fit on train, applied to test (no leakage).
- **Probability‑dependent charts dropped** — see Known Limitation in README.

## 5. Extending the project

- **Fix `churn_probability`**: in `03_modeling.ipynb`, ensure `model.predict_proba` runs on the **entire** customer base (not just the holdout) before exporting `dashboard.csv`. Then re‑add the probability histogram, risk‑band chart, and confusion matrix to the dashboard.
- **Try other models**: LightGBM, CatBoost, or a stacking ensemble. The notebook structure supports drop‑in replacement at the model‑fit cell.
- **Publish the dashboard**: in Tableau Public, *Server → Tableau Public → Save to Tableau Public As…*, then paste the resulting URL into `README.md`.
