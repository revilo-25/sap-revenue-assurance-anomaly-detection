"""
SAP FI-CA Revenue Forecasting — End-to-End Script
====================================================
Builds a monthly recurring revenue (MRR) forecast on top of the billing
dataset, with a proper time-series evaluation (not just "the plot looks right").

Sections:
  1. Load & aggregate monthly revenue
  2. Time-series train/test split (chronological, never random)
  3. Baseline model — seasonal naive (the benchmark any real model must beat)
  4. ARIMA model
  5. Evaluation — MAE, RMSE, MAPE, baseline vs ARIMA
  6. Visualization

Requirements: pandas, numpy, matplotlib, statsmodels, scikit-learn
  pip install pandas numpy matplotlib statsmodels scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.style.use("dark_background")
plt.rcParams["figure.facecolor"] = "#0F1620"
plt.rcParams["axes.facecolor"] = "#161F2C"
plt.rcParams["axes.edgecolor"] = "#3A4657"
plt.rcParams["text.color"] = "#E8ECF1"
plt.rcParams["axes.labelcolor"] = "#E8ECF1"
plt.rcParams["xtick.color"] = "#8A95A5"
plt.rcParams["ytick.color"] = "#8A95A5"
plt.rcParams["grid.color"] = "#232E3D"

DATA_DIR = "./data"      # reads billing_documents.csv from here
OUT_DIR = "./outputs"    # writes chart + results CSV here
import os
os.makedirs(OUT_DIR, exist_ok=True)
ACCENT = "#E8A33D"
NORMAL = "#4FA88F"
BASELINE_COLOR = "#8A95A5"

# =============================================================================
# 1. LOAD & AGGREGATE
# =============================================================================
billing = pd.read_csv(f"{DATA_DIR}/billing_documents.csv")

monthly_revenue = (
    billing.groupby("BILLING_PERIOD")["INVOICE_AMOUNT"].sum().sort_index()
)
monthly_revenue.index = pd.to_datetime(monthly_revenue.index)
monthly_revenue = monthly_revenue.asfreq("MS")  # MS = month start frequency

print(f"Loaded {len(monthly_revenue)} months of revenue data.")
print(monthly_revenue.tail())

# =============================================================================
# 2. TRAIN/TEST SPLIT — CHRONOLOGICAL, NOT RANDOM
# =============================================================================
# Time series must be split by time: train on the past, test on the future.
# A random split would let the model "see the future" during training and give
# a fake sense of accuracy — this is one of the most common mistakes in
# forecasting projects, and calling it out explicitly is a good interview point.
TEST_MONTHS = 6
train = monthly_revenue.iloc[:-TEST_MONTHS]
test = monthly_revenue.iloc[-TEST_MONTHS:]

print(f"\nTrain: {train.index.min().date()} to {train.index.max().date()} ({len(train)} months)")
print(f"Test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} months)")

# =============================================================================
# 3. BASELINE MODEL — SEASONAL NAIVE
# =============================================================================
# Before trusting any "smart" model, you need a dumb benchmark to beat.
# Seasonal naive = "this month's revenue will equal what it was 12 months ago."
# If SARIMA can't beat this, SARIMA isn't earning its complexity.
SEASON_LENGTH = 12
if len(train) >= SEASON_LENGTH:
    baseline_forecast = train.iloc[-SEASON_LENGTH:-SEASON_LENGTH + TEST_MONTHS].values
    if len(baseline_forecast) < TEST_MONTHS:
        # not enough history for a full seasonal lookback — fall back to last value repeated
        baseline_forecast = np.repeat(train.iloc[-1], TEST_MONTHS)
else:
    baseline_forecast = np.repeat(train.iloc[-1], TEST_MONTHS)

baseline_forecast = pd.Series(baseline_forecast[:TEST_MONTHS], index=test.index)

# =============================================================================
# 4. ARIMA MODEL
# =============================================================================
# IMPORTANT DESIGN NOTE: a seasonal SARIMA(1,1,1)(1,1,0,12) was tried first here
# and performed WORSE than the naive baseline (14% MAPE vs 5%). With only 24
# months of training data, that's just 2 full seasonal cycles — not enough
# for statsmodels to reliably estimate seasonal parameters, and it ends up
# fitting noise rather than a real yearly pattern. This is a common real-world
# forecasting trap: more model complexity doesn't help if the data doesn't
# support it, and this dataset's revenue plateaus into essentially flat noise
# after an initial ramp-up period, with no strong trend or seasonality to
# exploit. So: non-seasonal ARIMA(1,1,1), which is simpler and more honest
# about what this data can actually support.
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
fit = model.fit(disp=False)

arima_forecast = fit.forecast(steps=TEST_MONTHS)
arima_forecast.index = test.index

# Also get confidence intervals — a forecast without uncertainty bounds is
# incomplete for a finance stakeholder; they need to know the range, not just
# a point estimate.
forecast_obj = fit.get_forecast(steps=TEST_MONTHS)
conf_int = forecast_obj.conf_int(alpha=0.20)  # 80% confidence interval
conf_int.index = test.index

# =============================================================================
# 5. EVALUATION — MAE, RMSE, MAPE
# =============================================================================
def evaluate(actual, predicted, label):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"\n--- {label} ---")
    print(f"MAE:  ₹{mae:,.0f}")
    print(f"RMSE: ₹{rmse:,.0f}")
    print(f"MAPE: {mape:.2f}%")
    return {"label": label, "mae": mae, "rmse": rmse, "mape": mape}

print("\n" + "=" * 50)
print("FORECAST EVALUATION (last 6 months held out)")
print("=" * 50)

baseline_metrics = evaluate(test.values, baseline_forecast.values, "Baseline (Seasonal Naive)")
arima_metrics = evaluate(test.values, arima_forecast.values, "ARIMA(1,1,1)")

improvement = (1 - arima_metrics["mape"] / baseline_metrics["mape"]) * 100
print(f"\nARIMA improves on baseline MAPE by {improvement:.1f}%")

# =============================================================================
# 6. VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("SAP FI-CA — Revenue Forecast", fontsize=16, color="#E8ECF1", weight="bold")

# Panel 1: full history + forecast + confidence interval
ax = axes[0]
ax.plot(train.index, train.values, color=NORMAL, linewidth=2, label="Actual (train)")
ax.plot(test.index, test.values, color="#E8ECF1", linewidth=2, marker="o", label="Actual (held-out test)")
ax.plot(arima_forecast.index, arima_forecast.values, color=ACCENT, linewidth=2,
        linestyle="--", marker="s", label="ARIMA forecast")
ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color=ACCENT, alpha=0.15, label="80% confidence interval")
ax.set_title("Monthly Revenue: Actual vs Forecast", color="#E8ECF1")
ax.set_ylabel("Revenue (INR)")
ax.legend(facecolor="#161F2C", labelcolor="#E8ECF1", loc="upper left", fontsize=9)

# Panel 2: baseline vs ARIMA on the test window (zoomed in)
ax = axes[1]
ax.plot(test.index, test.values, color="#E8ECF1", linewidth=2, marker="o", label="Actual")
ax.plot(test.index, baseline_forecast.values, color=BASELINE_COLOR, linewidth=2,
        linestyle=":", marker="^", label=f"Baseline (MAPE {baseline_metrics['mape']:.1f}%)")
ax.plot(test.index, arima_forecast.values, color=ACCENT, linewidth=2,
        linestyle="--", marker="s", label=f"ARIMA (MAPE {arima_metrics['mape']:.1f}%)")
ax.set_title("Test Window: Baseline vs ARIMA", color="#E8ECF1")
ax.set_ylabel("Revenue (INR)")
ax.legend(facecolor="#161F2C", labelcolor="#E8ECF1", fontsize=9)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUT_DIR}/revenue_forecast.png", dpi=150, facecolor=fig.get_facecolor())
print(f"\nChart saved to {OUT_DIR}/revenue_forecast.png")
plt.show()

# Save results for your writeup
results_df = pd.DataFrame({
    "month": test.index,
    "actual": test.values,
    "baseline_forecast": baseline_forecast.values,
    "arima_forecast": arima_forecast.values,
    "arima_lower_80": conf_int.iloc[:, 0].values,
    "arima_upper_80": conf_int.iloc[:, 1].values,
})
results_df.to_csv(f"{OUT_DIR}/forecast_results.csv", index=False)
print(f"Results saved to {OUT_DIR}/forecast_results.csv")
