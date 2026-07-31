# SAP Revenue Assurance — Anomaly Detection & Forecasting

A financial-controls analytics case study built on synthetic SAP-style subscription billing data (document flow: business partner → contract account → billing document → GL posting).

> **Data disclaimer:** All data in this project is synthetically generated for demonstration purposes. Business partner names, contract accounts, and billing documents are fictional. This project does not use, reference, or reveal any real company's data, systems, or customers.

## Problem

Billing systems rarely throw an error for a *wrong* invoice — only for a *technically invalid* one. A stale rate card, a duplicate billing run, or an unexplained credit note will process cleanly and quietly erode revenue for months before anyone notices. This is the core problem revenue assurance functions exist to solve, and it's a natural fit for anomaly detection: the errors are rare, don't follow a fixed rule, and need to be caught relative to context (a ₹50,000 invoice is normal for one customer and alarming for another).

## Approach

**1. Data model** — Simulated 500 business partners, 500 contracts, and ~13,000 monthly billing documents over 30 months, with four distinct anomaly types deliberately seeded into the data: underbilling, billing spikes, unexplained credits, and duplicate charges.

**2. Feature engineering** — Every feature is *relative*, not absolute, since raw invoice amount alone is a poor signal:
- Deviation from the amount the contract terms would predict
- Deviation from the customer's own historical average (computed causally — expanding mean shifted by one period, so the model never sees the current invoice while forming its own baseline)
- Duplicate-billing-period flag
- Invoice-to-base-fee ratio

**3. Unsupervised detection** — Isolation Forest, chosen because real billing systems don't come with pre-labeled fraud/error examples; it isolates anomalies by how few random splits it takes to separate them from the rest of the data.

**4. Hybrid rule + ML layer** — Isolation Forest structurally cannot catch exact duplicate charges: a duplicate's amount is, by definition, statistically normal. Rather than force the ML model to solve something it can't, a deterministic duplicate-detection rule is OR'd together with the ML flag.

**5. Explainability** — Isolation Forest scores don't decompose into per-feature attribution, so a supervised surrogate model (Random Forest, trained to reproduce the hybrid model's decisions, 96.4% fidelity) is explained with SHAP — turning "flagged" into "flagged, and here's why."

![SHAP Feature Importance](/shap_feature_importance.png)

**6. Forecasting** — A separate monthly revenue forecast (ARIMA) evaluated against a seasonal-naive baseline using a strict chronological train/test split, because random splitting would let a time-series model "see the future" during training.

## Results

![Anomaly Detection Dashboard](/anomaly_dashboard.png)

| Metric | ML only | Hybrid (ML + rule) |
|---|---|---|
| Precision | 78.8% | 81.7% |
| Recall | 83.3% | **100%** |
| F1 | 0.809 | **0.899** |

- **100% of the dollar value** of seeded anomalies (₹40.0L) was caught by the hybrid model
- Recall broken out by anomaly type shows exactly where the pure-ML approach fell short (duplicates) and confirms the rule-based layer closed that specific gap — this diagnostic step is the part most anomaly-detection projects skip

![Revenue Forecast](/revenue_forecast.png)

**Forecasting finding, stated honestly:** a seasonal SARIMA model was tested first and performed *worse* than a simple seasonal-naive baseline (14% MAPE vs 5%). With only 24 months of training data — two seasonal cycles — there isn't enough history for statsmodels to reliably estimate seasonal parameters; it ends up fitting noise. The simpler non-seasonal ARIMA(1,1,1) is more honest about what the data can actually support, and even it underperforms the naive baseline (6.0% vs 5.1% MAPE) on this particular dataset. **Not every dataset needs — or rewards — a complex model, and knowing when to stop adding complexity is part of the analysis.**

## Tech stack
Python · pandas · scikit-learn (IsolationForest, RandomForest) · SHAP · statsmodels (ARIMA) · matplotlib/seaborn

## Project structure

```
project/
├── data/                              ← input CSVs (place these here)
│   ├── billing_documents.csv
│   ├── contracts.csv
│   ├── business_partners.csv
│   └── GROUND_TRUTH_anomalies.csv
├── outputs/                           ← generated automatically
├── revenue_assurance_pipeline.py      ← run 1st
├── revenue_forecast.py                ← run 2nd
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

python revenue_assurance_pipeline.py   # anomaly detection + explainability
python revenue_forecast.py             # revenue forecasting
```

### Step 1 — `revenue_assurance_pipeline.py`
Reads `data/billing_documents.csv`, `data/contracts.csv`, `data/business_partners.csv`, `data/GROUND_TRUTH_anomalies.csv`.
Writes:
- `scored_billing_documents_full.csv` — full scored dataset
- `anomaly_dashboard.png` — results dashboard
- `shap_feature_importance.png` — explainability chart

### Step 2 — `revenue_forecast.py`
Reads `data/billing_documents.csv`.
Writes:
- `revenue_forecast.png` — forecast chart
- `forecast_results.csv` — forecast vs actual, with 80% confidence interval
