import json

import pandas as pd

from churn.features import prepare

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0, "TotalCharges": 190.0,
}
LOW_RISK = {
    **HIGH_RISK, "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 70,
    "InternetService": "DSL", "OnlineSecurity": "Yes", "TechSupport": "Yes",
    "Contract": "Two year", "PaymentMethod": "Bank transfer (automatic)", "PaperlessBilling": "No",
    "MonthlyCharges": 60.0, "TotalCharges": 4200.0,
}


def test_quality_floor_and_policy(artifacts):
    m = json.loads((artifacts / "metrics.json").read_text())
    assert m["test"]["roc_auc"] > 0.80
    assert m["test"]["brier"] < 0.16
    profit = m["test_profit_by_policy"]
    assert profit["model_break_even"] > max(profit["threshold_0.5"], profit["contact_everyone"])


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_risk_ordering_and_recommendation(client):
    hi = client.post("/predict", json=HIGH_RISK).json()
    lo = client.post("/predict", json=LOW_RISK).json()
    assert hi["churn_probability"] > 0.5 > lo["churn_probability"]
    assert hi["recommend_contact"] and not lo["recommend_contact"]
    assert len(hi["top_drivers"]) == 3
    assert "Contract" in {d["feature"] for d in hi["top_drivers"] + lo["top_drivers"]}


def test_batch_matches_single(client):
    batch = client.post("/predict/batch", json={"customers": [HIGH_RISK, LOW_RISK]}).json()
    single = [client.post("/predict", json=c).json() for c in (HIGH_RISK, LOW_RISK)]
    assert batch == single


def test_invalid_input_rejected(client):
    for bad in ({**HIGH_RISK, "Contract": "Weekly"}, {**HIGH_RISK, "tenure": -1},
                {k: v for k, v in HIGH_RISK.items() if k != "MonthlyCharges"}):
        assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_new_customer_blank_total_charges():
    raw = pd.DataFrame([{**HIGH_RISK, "tenure": 0, "TotalCharges": " "}])
    X = prepare(raw)
    assert X["TotalCharges"].iloc[0] == 0.0 and X["avg_charge_ratio"].iloc[0] == 1.0
