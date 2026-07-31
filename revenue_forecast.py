
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

billing = pd.read_csv(f"{DATA_DIR}/billing_documents.csv")

monthly_revenue = (
    billing.groupby("BILLING_PERIOD")["INVOICE_AMOUNT"].sum().sort_index()
)
monthly_revenue.index = pd.to_datetime(monthly_revenue.index)
monthly_revenue = monthly_revenue.asfreq("MS")  # MS = month start frequency

print(f"Loaded {len(monthly_revenue)} months of revenue data.")
print(monthly_revenue.tail())

TEST_MONTHS = 6
train = monthly_revenue.iloc[:-TEST_MONTHS]
test = monthly_revenue.iloc[-TEST_MONTHS:]

print(f"\nTrain: {train.index.min().date()} to {train.index.max().date()} ({len(train)} months)")
print(f"Test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} months)")

SEASON_LENGTH = 12
if len(train) >= SEASON_LENGTH:
    baseline_forecast = train.iloc[-SEASON_LENGTH:-SEASON_LENGTH + TEST_MONTHS].values
    if len(baseline_forecast) < TEST_MONTHS:
        # not enough history for a full seasonal lookback — fall back to last value repeated
        baseline_forecast = np.repeat(train.iloc[-1], TEST_MONTHS)
else:
    baseline_forecast = np.repeat(train.iloc[-1], TEST_MONTHS)

baseline_forecast = pd.Series(baseline_forecast[:TEST_MONTHS], index=test.index)

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


fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle("SAP FI-CA — Revenue Forecast", fontsize=16, color="#E8ECF1", weight="bold")

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
