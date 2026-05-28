# 🛒 E-commerce Customer Churn Project (ML Portfolio)

## 🎯 Objective

Predict customer churn and identify the key drivers behind exit behavior on an Indian e-commerce platform, using a 200,000-customer dataset spanning seven major cities and four subscription tiers.

---

## 📁 Project Structure

```
ecommerce-customer-churn/
│── data/
│   ├── raw/                  # Original CSV
│   ├── processed/            # Cleaned / engineered tables
│
│── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│
│── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate.py
│
│── outputs/
│   ├── figures/
│   ├── models/
│
│── dashboard/
│   ├── powerbi/              # Power BI Service (web) workbook + screenshots
│   ├── tableau/              # Tableau Public workbook (.twbx) + screenshots
│
│── INSTRUCTIONS.md
│── project.md
│── README.md
```

---

## 🔹 Stage 1: Data Understanding

* Load dataset
* Inspect schema and data types
* Identify target variable (`churn`)
* Check missing values and duplicates by `customer_id`

---

## 🔹 Stage 2: Exploratory Data Analysis (EDA)

### Key Analysis

* Customer demographics (age, gender, city)
* Distribution of tenure, order value, order count, support tickets
* Overall churn rate

### Comparative Analysis

* Churn by city
* Churn by subscription tier (Basic / Silver / Gold / Platinum)
* Churn by age group
* Churn by recency (`last_purchase_days_ago`)
* Churn by support-ticket volume

### Visualizations

* Histograms, boxplots
* Bar charts of churn rate per category
* Correlation heatmap

---

## 🔹 Stage 3: Feature Engineering

* Encode categoricals (`gender`, `city`, `subscription_type`)
* Derived features:
  * `orders_per_month = total_orders / tenure_months`
  * `revenue_per_month = (avg_order_value * total_orders) / tenure_months`
  * `tickets_per_order = support_tickets / total_orders`
  * `is_dormant = (last_purchase_days_ago > 180).astype(int)`
  * `age_group` bins
* Scale numerics for linear / distance-based models

---

## 🔹 Stage 4: Data Splitting

* Train/Test split (80/20), stratified on `churn`
* Optional: 5-fold StratifiedKFold

---

## 🔹 Stage 5: Model Building

Train multiple models:

* Logistic Regression (baseline, scaled)
* Random Forest
* XGBoost

---

## 🔹 Stage 6: Model Evaluation

Metrics:

* Accuracy
* Precision / Recall / F1
* ROC-AUC
* Confusion matrix, ROC curves

Focus: **recall on the churn class** — missing a churner costs lifetime revenue.

---

## 🔹 Stage 7: Feature Importance & Explainability

* Tree-based feature importance
* SHAP global summary + per-customer waterfall

Answer: **What drives churn on this platform?**

---

## 🔹 Stage 8: Customer Segmentation

* KMeans on behavioral features
* Profile + label clusters (e.g., "Dormant high-value", "Loyal Platinum", "At-risk Basic")

---

## 🔹 Stage 9: Business Insights

* Translate findings into retention recommendations
* Tie each recommendation to a driver and a segment

---

## 🔹 Stage 10: Dashboard / Visualization

Build the **same dashboard in both tools**:

* **Power BI Service (web)** — published to your Power BI workspace
* **Tableau Public** — shareable portfolio link

Include:

* KPI tiles, churn breakdowns, risk distribution, segment view, SHAP drivers

---

## 🔹 Stage 11: Final Deliverables

* Clean notebooks
* Trained model + scaler artifacts
* Power BI dashboard (web link) **and** Tableau dashboard (public link + `.twbx`)
* README with insights and both dashboard links

---

## 🏁 Final Goal

Deliver a complete ML pipeline + business insights that explain churn behavior on an e-commerce platform, surfaced through dashboards built in **both** Power BI and Tableau so the same story can be told in either tool.
