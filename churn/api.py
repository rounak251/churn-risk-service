"""HTTP API: score customers and recommend who to contact.

uvicorn churn.api:app --port 8000
"""
import os
from enum import Enum
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .explain import top_drivers
from .features import prepare

ARTIFACT = os.environ.get("CHURN_MODEL", "artifacts/model.joblib")
bundle = joblib.load(ARTIFACT)
app = FastAPI(title="Churn risk service", version="1.0.0")


class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class Customer(BaseModel):
    gender: str = Field(pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: str = Field(pattern="^(Yes|No|No phone service)$")
    InternetService: str = Field(pattern="^(DSL|Fiber optic|No)$")
    OnlineSecurity: str = Field(pattern="^(Yes|No|No internet service)$")
    OnlineBackup: str = Field(pattern="^(Yes|No|No internet service)$")
    DeviceProtection: str = Field(pattern="^(Yes|No|No internet service)$")
    TechSupport: str = Field(pattern="^(Yes|No|No internet service)$")
    StreamingTV: str = Field(pattern="^(Yes|No|No internet service)$")
    StreamingMovies: str = Field(pattern="^(Yes|No|No internet service)$")
    Contract: str = Field(pattern="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: YesNo
    PaymentMethod: str = Field(pattern="^(Electronic check|Mailed check|Bank transfer \\(automatic\\)|Credit card \\(automatic\\))$")
    MonthlyCharges: float = Field(gt=0, lt=1000)
    TotalCharges: float = Field(ge=0)


class Scored(BaseModel):
    churn_probability: float
    expected_profit_if_contacted: float
    recommend_contact: bool
    top_drivers: list


class Batch(BaseModel):
    customers: List[Customer] = Field(min_length=1, max_length=5000)


def _score(customers):
    raw = pd.DataFrame([c.model_dump(mode="json") for c in customers])
    X = prepare(raw)
    p = bundle["model"].predict_proba(X)[:, 1]
    econ = bundle["economics"]
    profit = econ.expected_profit(p)
    drivers = top_drivers(bundle["explainer"], bundle["reference"], X)
    return [Scored(churn_probability=round(float(pi), 4), expected_profit_if_contacted=round(float(gi), 2),
                   recommend_contact=bool(gi > 0), top_drivers=d)
            for pi, gi, d in zip(p, profit, drivers)]


@app.get("/health")
def health():
    m = bundle["metrics"]
    return {"status": "ok", "model": m["selected_model"], "test_roc_auc": m["test"]["roc_auc"]}


@app.post("/predict", response_model=Scored)
def predict(customer: Customer):
    return _score([customer])[0]


@app.post("/predict/batch", response_model=List[Scored])
def predict_batch(batch: Batch):
    return _score(batch.customers)
