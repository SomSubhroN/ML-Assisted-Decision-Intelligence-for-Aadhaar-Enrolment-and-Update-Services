import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import LabelEncoder

# =========================================================
# 1. LOAD MASTER DATASET
# =========================================================

INPUT_PATH = r"C:\Users\User\Desktop\UIDAI Hackathon\output\aadhaar_master_dataset.csv"

df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

print("Loaded master dataset:", df.shape)

# =========================================================
# 2. MONTHLY AGGREGATION
# =========================================================

df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

monthly = (
    df.groupby([
        "month",
        "state",
        "district",
        "source_type",
        "age_group"
    ])["count"]
    .sum()
    .reset_index()
)

monthly.to_csv("aadhaar_monthly_aggregated.csv", index=False)
print("Monthly aggregation done")

# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================

monthly["month_num"] = monthly["month"].dt.month
monthly["year"] = monthly["month"].dt.year

le_state = LabelEncoder()
le_district = LabelEncoder()

monthly["state_code"] = le_state.fit_transform(monthly["state"])
monthly["district_code"] = le_district.fit_transform(monthly["district"])

# =========================================================
# 4. ML FORECASTING (ENROLMENT DEMAND)
# =========================================================

enrol = monthly[monthly["source_type"] == "enrolment"]

X = enrol[[
    "month_num",
    "year",
    "state_code",
    "district_code"
]]

y = enrol["count"]

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

# Prepare future frame (next 6 months)
future = enrol.groupby(
    ["state", "district", "state_code", "district_code"]
).tail(1).copy()

future = pd.concat([future] * 6, ignore_index=True)
# Generate future months safely (pandas 2.x compatible)
future_months = []
for i in range(1, 7):
    temp = future.copy()
    temp["month"] = temp["month"] + pd.DateOffset(months=i)
    future_months.append(temp)

future = pd.concat(future_months, ignore_index=True)


future["month_num"] = future["month"].dt.month
future["year"] = future["month"].dt.year

future["predicted_enrolment"] = rf.predict(
    future[["month_num", "year", "state_code", "district_code"]]
)

# =========================================================
# 5. CONFIDENCE INTERVALS (BOOTSTRAP)
# =========================================================

bootstrap_preds = []

for _ in range(50):
    sample = enrol.sample(frac=0.8, replace=True)
    rf.fit(
        sample[["month_num", "year", "state_code", "district_code"]],
        sample["count"]
    )
    bootstrap_preds.append(
        rf.predict(
            future[["month_num", "year", "state_code", "district_code"]]
        )
    )

bootstrap_preds = np.array(bootstrap_preds)

future["lower_ci"] = np.percentile(bootstrap_preds, 10, axis=0)
future["upper_ci"] = np.percentile(bootstrap_preds, 90, axis=0)

future.to_csv("enrolment_forecast_with_confidence.csv", index=False)
print("Forecasting with confidence intervals done")

# =========================================================
# 6. STALENESS DETECTION (ML + RULES)
# =========================================================

enrol_m = monthly[monthly["source_type"] == "enrolment"]
updates_m = monthly[monthly["source_type"].isin(["biometric", "demographic"])]

ratio = (
    updates_m.groupby(["district", "month"])["count"].sum()
    /
    enrol_m.groupby(["district", "month"])["count"].sum()
).reset_index(name="update_ratio")

ratio["update_ratio"] = ratio["update_ratio"].fillna(0)

iso = IsolationForest(contamination=0.1, random_state=42)
ratio["anomaly"] = iso.fit_predict(ratio[["update_ratio"]])

stale_regions = ratio[
    (ratio["update_ratio"] < 0.15) | (ratio["anomaly"] == -1)
]

stale_regions.to_csv("aadhaar_stale_update_regions.csv", index=False)
print("Staleness detection completed")

# =========================================================
# 7. ASRI (AADHAAR SERVICE READINESS INDEX)
# =========================================================

enrol_score = enrol_m.groupby("district")["count"].sum()
update_score = updates_m.groupby("district")["count"].sum()
stale_penalty = stale_regions.groupby("district")["update_ratio"].mean().fillna(0)

asri = (
    0.4 * enrol_score +
    0.4 * update_score -
    0.2 * stale_penalty * enrol_score.mean()
)

asri = (
    (asri - asri.min()) /
    (asri.max() - asri.min())
) * 100

asri = asri.sort_values(ascending=False)

asri.to_csv("asri_index.csv")
print("ASRI computed")

# =========================================================
# 8. WHAT UIDAI CAN DO NEXT (DECISION INTELLIGENCE)
# =========================================================

recommendations = []

for district in asri.head(5).index:
    recommendations.append(
        f"{district}: High Aadhaar service pressure detected. "
        f"Recommend additional enrolment and update capacity."
    )

for district in stale_regions["district"].unique()[:5]:
    recommendations.append(
        f"{district}: Aadhaar updates lagging significantly. "
        f"Recommend mobile update units and awareness campaigns."
    )

pd.DataFrame(
    {"uidai_actionable_recommendation": recommendations}
).to_csv("uidai_next_steps.csv", index=False)

print("Decision intelligence generated")

# =========================================================
# 9. PIPELINE COMPLETE
# =========================================================

print("\n✅ UIDAI ML Intelligence Pipeline Completed Successfully")
print("Generated files:")
print("- aadhaar_monthly_aggregated.csv")
print("- enrolment_forecast_with_confidence.csv")
print("- aadhaar_stale_update_regions.csv")
print("- asri_index.csv")
print("- uidai_next_steps.csv")
