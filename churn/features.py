"""Raw Telco export -> model-ready frame. Shared by training and serving."""
import pandas as pd

CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
DERIVED = ["avg_charge_ratio", "n_addons"]
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
FEATURES = CATEGORICAL + NUMERIC + DERIVED


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # TotalCharges is blank for brand-new customers (tenure 0): nothing billed yet.
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    # >1 means the customer pays more now than their lifetime average: a recent price rise.
    lifetime_avg = out["TotalCharges"] / out["tenure"].where(out["tenure"] > 0)
    out["avg_charge_ratio"] = (out["MonthlyCharges"] / lifetime_avg).fillna(1.0)
    out["n_addons"] = (out[ADDONS] == "Yes").sum(axis=1)
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")
    return out[FEATURES]


def target(df: pd.DataFrame):
    return (df["Churn"] == "Yes").astype(int).to_numpy()
