
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import shap

sns.set_theme(style="darkgrid")
plt.rcParams["figure.facecolor"] = "#0F1620"
plt.rcParams["axes.facecolor"] = "#161F2C"
plt.rcParams["axes.edgecolor"] = "#3A4657"
plt.rcParams["text.color"] = "#E8ECF1"
plt.rcParams["axes.labelcolor"] = "#E8ECF1"
plt.rcParams["xtick.color"] = "#8A95A5"
plt.rcParams["ytick.color"] = "#8A95A5"
plt.rcParams["grid.color"] = "#232E3D"

DATA_DIR = "./data"
OUT_DIR = "./outputs"
os.makedirs(OUT_DIR, exist_ok=True)

ACCENT = "#E8A33D"
NORMAL = "#4FA88F"

billing = pd.read_csv(f"{DATA_DIR}/billing_documents.csv")
contracts = pd.read_csv(f"{DATA_DIR}/contracts.csv")
bp = pd.read_csv(f"{DATA_DIR}/business_partners.csv")
truth = pd.read_csv(f"{DATA_DIR}/GROUND_TRUTH_anomalies.csv")

df = billing.merge(
    contracts[["CONTRACT_ACCOUNT", "BASE_FEE", "USAGE_RATE", "STATUS"]],
    on="CONTRACT_ACCOUNT", how="left"
)
df = df.merge(bp[["GPART", "REGION", "INDUSTRY"]], on="GPART", how="left")

print(f"Loaded {len(df):,} billing documents | {df['GPART'].nunique()} customers | "
      f"{len(truth)} seeded ground-truth anomalies")

df["EXPECTED_AMOUNT"] = df["BASE_FEE"] + df["USAGE_UNITS"] * df["USAGE_RATE"]
df["AMOUNT_DEVIATION_PCT"] = (
    (df["INVOICE_AMOUNT"] - df["EXPECTED_AMOUNT"]) / df["EXPECTED_AMOUNT"].replace(0, np.nan)
)

df = df.sort_values(["CONTRACT_ACCOUNT", "BILLING_PERIOD"])
df["CUST_ROLLING_MEAN"] = df.groupby("CONTRACT_ACCOUNT")["INVOICE_AMOUNT"].transform(
    lambda x: x.expanding().mean().shift(1)
)
df["CUST_ROLLING_MEAN"] = df["CUST_ROLLING_MEAN"].fillna(df["INVOICE_AMOUNT"])
df["DEV_FROM_OWN_HISTORY"] = df["INVOICE_AMOUNT"] - df["CUST_ROLLING_MEAN"]

dup_counts = df.groupby(["CONTRACT_ACCOUNT", "BILLING_PERIOD"]).size().rename("DOC_COUNT_SAME_PERIOD")
df = df.merge(dup_counts, on=["CONTRACT_ACCOUNT", "BILLING_PERIOD"])
df["IS_DUPLICATE_PERIOD"] = (df["DOC_COUNT_SAME_PERIOD"] > 1).astype(int)
df["AMOUNT_TO_BASE_RATIO"] = df["INVOICE_AMOUNT"] / df["BASE_FEE"].replace(0, np.nan)

FEATURES = [
    "INVOICE_AMOUNT", "AMOUNT_DEVIATION_PCT", "DEV_FROM_OWN_HISTORY",
    "IS_DUPLICATE_PERIOD", "AMOUNT_TO_BASE_RATIO", "DUNNING_LEVEL",
]

model_df = df.copy()
model_df[FEATURES] = model_df[FEATURES].replace([np.inf, -np.inf], np.nan)
model_df[FEATURES] = model_df[FEATURES].fillna(model_df[FEATURES].median())

X = StandardScaler().fit_transform(model_df[FEATURES])
iso_forest = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
model_df["ANOMALY_SCORE_RAW"] = iso_forest.fit_predict(X)
model_df["ANOMALY_DECISION_SCORE"] = iso_forest.decision_function(X)
model_df["PREDICTED_ANOMALY"] = (model_df["ANOMALY_SCORE_RAW"] == -1).astype(int)

model_df["RULE_FLAG_DUPLICATE"] = model_df["IS_DUPLICATE_PERIOD"]
model_df["FINAL_ANOMALY_FLAG"] = (
    (model_df["PREDICTED_ANOMALY"] == 1) | (model_df["RULE_FLAG_DUPLICATE"] == 1)
).astype(int)


truth_ids = set(truth["BILLING_DOC"])
model_df["IS_TRUE_ANOMALY"] = model_df["BILLING_DOC"].isin(truth_ids).astype(int)

def eval_flag(col, label):
    p = precision_score(model_df.IS_TRUE_ANOMALY, model_df[col])
    r = recall_score(model_df.IS_TRUE_ANOMALY, model_df[col])
    f1 = f1_score(model_df.IS_TRUE_ANOMALY, model_df[col])
    print(f"{label:28s} Precision {p:.3f}  Recall {r:.3f}  F1 {f1:.3f}")
    return p, r, f1

print("\n--- Detection performance ---")
eval_flag("PREDICTED_ANOMALY", "ML only")
p, r, f1 = eval_flag("FINAL_ANOMALY_FLAG", "Hybrid (ML + rule)")

missed = model_df[(model_df.IS_TRUE_ANOMALY == 1) & (model_df.FINAL_ANOMALY_FLAG == 0)]
caught = model_df[(model_df.IS_TRUE_ANOMALY == 1) & (model_df.FINAL_ANOMALY_FLAG == 1)]
total_true_value = model_df.loc[model_df.IS_TRUE_ANOMALY == 1, "INVOICE_AMOUNT"].abs().sum()
caught_value = caught["INVOICE_AMOUNT"].abs().sum()
missed_value = missed["INVOICE_AMOUNT"].abs().sum()
print(f"\n--- Dollar-weighted recall ---")
print(f"Total value at risk (seeded anomalies): INR {total_true_value:,.0f}")
print(f"Value CAUGHT by model:                  INR {caught_value:,.0f} ({caught_value/total_true_value:.1%})")
print(f"Value MISSED by model:                  INR {missed_value:,.0f} ({missed_value/total_true_value:.1%})")

recall_by_type = (
    truth.merge(model_df[["BILLING_DOC", "FINAL_ANOMALY_FLAG"]], on="BILLING_DOC", how="left")
    .groupby("ANOMALY_TYPE")["FINAL_ANOMALY_FLAG"].mean()
)
print("\n--- Recall by anomaly type (hybrid) ---")
print(recall_by_type.round(3))

X_train, X_test, y_train, y_test = train_test_split(
    model_df[FEATURES], model_df["FINAL_ANOMALY_FLAG"], test_size=0.3,
    random_state=42, stratify=model_df["FINAL_ANOMALY_FLAG"]
)
surrogate = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
surrogate.fit(X_train, y_train)

surrogate_f1 = f1_score(y_test, surrogate.predict(X_test))
print(f"\nSurrogate model fidelity (F1 vs hybrid flag on held-out set): {surrogate_f1:.3f}")

explainer = shap.TreeExplainer(surrogate)
shap_values = explainer.shap_values(X_test)
# shap_values for binary classifiers: take the positive class
sv = shap_values[1] if isinstance(shap_values, list) else shap_values
if sv.ndim == 3:
    # Newer SHAP versions return (n_samples, n_features, n_classes)
    sv = sv[:, :, 1]

plt.figure(figsize=(9, 6))
shap.summary_plot(sv, X_test, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/shap_feature_importance.png", dpi=150, facecolor="#0F1620")
plt.close()
print(f"SHAP summary saved to {OUT_DIR}/shap_feature_importance.png")

# =============================================================================
# 7. DASHBOARD
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("SAP Revenue Assurance — Anomaly Detection Dashboard", fontsize=16, color="#E8ECF1", weight="bold")

ax = axes[0, 0]
monthly = model_df.groupby("BILLING_PERIOD").agg(
    revenue=("INVOICE_AMOUNT", "sum"), flagged=("FINAL_ANOMALY_FLAG", "sum")
).reset_index()
ax.plot(monthly["BILLING_PERIOD"], monthly["revenue"], color=NORMAL, linewidth=2)
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(monthly["BILLING_PERIOD"], rotation=90, fontsize=6)
ax.set_title("Monthly Revenue Trend", color="#E8ECF1")
ax.set_ylabel("Revenue (INR)")
ax2 = ax.twinx()
ax2.bar(monthly["BILLING_PERIOD"], monthly["flagged"], color=ACCENT, alpha=0.4)
ax2.set_ylabel("Flagged docs", color=ACCENT)

ax = axes[0, 1]
recall_by_type.sort_values().plot(kind="barh", ax=ax, color=ACCENT)
ax.set_title("Detection Recall by Anomaly Type (Hybrid)", color="#E8ECF1")
ax.set_xlabel("Recall")
ax.set_xlim(0, 1)

ax = axes[1, 0]
sample = model_df.sample(min(2000, len(model_df)), random_state=42)
normal_pts = sample[sample["FINAL_ANOMALY_FLAG"] == 0]
anomaly_pts = sample[sample["FINAL_ANOMALY_FLAG"] == 1]
ax.scatter(normal_pts["INVOICE_AMOUNT"], normal_pts["AMOUNT_DEVIATION_PCT"].clip(-2, 5),
           s=8, color=NORMAL, alpha=0.4, label="Normal")
ax.scatter(anomaly_pts["INVOICE_AMOUNT"], anomaly_pts["AMOUNT_DEVIATION_PCT"].clip(-2, 5),
           s=14, color=ACCENT, alpha=0.8, label="Flagged")
ax.set_title("Invoice Amount vs Deviation from Expected", color="#E8ECF1")
ax.set_xlabel("Invoice amount (INR)")
ax.set_ylabel("Deviation from expected (%)")
ax.legend(facecolor="#161F2C", labelcolor="#E8ECF1")

ax = axes[1, 1]
ax.axis("off")
cm_text = (
    f"HYBRID MODEL RESULTS\n\n"
    f"Precision: {p:.1%}\nRecall: {r:.1%}\nF1 Score: {f1:.3f}\n\n"
    f"Value at risk caught: {caught_value/total_true_value:.1%}\n"
    f"Value at risk missed: {missed_value/total_true_value:.1%}\n\n"
    f"Surrogate SHAP fidelity: {surrogate_f1:.3f}"
)
ax.text(0.1, 0.5, cm_text, fontsize=12, color="#E8ECF1", family="monospace", va="center")
ax.set_title("Model Evaluation Summary", color="#E8ECF1")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUT_DIR}/anomaly_dashboard.png", dpi=150, facecolor=fig.get_facecolor())
plt.close()
print(f"Dashboard saved to {OUT_DIR}/anomaly_dashboard.png")

model_df.to_csv(f"{OUT_DIR}/scored_billing_documents_full.csv", index=False)
print(f"Scored dataset saved to {OUT_DIR}/scored_billing_documents_full.csv")
print("\nDone. Run revenue_forecast.py next for the forecasting component.")
