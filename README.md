# churn-risk-service

An end-to-end churn model for a subscription business. It covers validated training,
calibrated probabilities, a profit-based contact policy, per-customer explanations, and a
FastAPI service in Docker.

The question a retention team actually asks is not "what is the AUC?". It is **"who
should we call this month, and is it worth it?"** This project answers that.

## Results (held-out test set, 1,409 customers)

| Policy | Customers contacted | Profit |
|---|---:|---:|
| Contact nobody | 0 | $0 |
| Contact everyone | 1,409 | $19,310 |
| Model, standard 0.5 cut-off | 286 | $31,300 |
| **Model, profit break-even cut-off** | **662** | **$42,500** |

Profit uses explicit, configurable assumptions in [`churn/economics.py`](churn/economics.py):
an offer costs $50, 30% of contacted churners stay, and a saved customer is worth $800.
Under these assumptions contacting someone pays off once their churn probability is above
50 / (0.3 × 800) ≈ 0.21, not 0.5. That gives **36% more profit** than the default cut-off.
This only works because the probabilities are calibrated.

| Model (5-fold CV on train) | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|
| **Logistic regression** (selected) | **0.844** | **0.656** | **0.136** |
| LightGBM | 0.842 | 0.646 | 0.137 |

Held-out test: ROC-AUC 0.842, PR-AUC 0.630, Brier 0.139.

Gradient boosting did *not* beat a well-specified linear model on this data, so the
simpler, more explainable model ships. Model selection is decided by validation, not by
preference.

## API

```
$ curl -s localhost:8000/predict -H 'content-type: application/json' -d @customer.json
{
  "churn_probability": 0.9673,
  "expected_profit_if_contacted": 182.16,
  "recommend_contact": true,
  "top_drivers": [
    {"feature": "tenure", "effect": "raises risk", "log_odds": 1.495},
    {"feature": "InternetService", "effect": "raises risk", "log_odds": 0.519},
    {"feature": "Contract", "effect": "raises risk", "log_odds": 0.467}
  ]
}
```

Endpoints: `GET /health`, `POST /predict`, `POST /predict/batch` (up to 5,000 customers).
Inputs are validated (allowed category values, ranges), so bad rows get a 422 instead of a
silent garbage score.

## How it is built

- **Leak-free evaluation.** A stratified 20% test split is touched once, at the end. The
  model is chosen and calibrated with 5-fold CV inside the training split.
- **Calibration.** Isotonic calibration (`CalibratedClassifierCV`), measured by Brier score,
  because the business decision depends on the probability value, not just the ranking.
- **Explanations from the served model.** For logistic regression, each feature's
  contribution is coefficient × (value − training mean), summed over its one-hot columns.
  For LightGBM it uses native SHAP values.
- **Features.** Blank `TotalCharges` for new customers is handled (nothing billed yet).
  There is also a recent-price-rise ratio (current vs lifetime-average charge) and an
  add-on count.
- **Reproducible.** Seeds are fixed and training is deterministic (two runs give identical
  metrics), dependencies are pinned, and the Docker image trains the model at build time.
  CI runs tests and builds and smoke-tests the container.

## Run

```
make train     # downloads the data, trains, writes artifacts/metrics.json
make test      # 6 tests: quality floor, policy beats baselines, API behaviour, validation
make serve     # http://localhost:8000/docs
docker build -t churn . && docker run -p 8000:8000 churn
```

Data: IBM Telco Customer Churn sample dataset (7,043 customers), downloaded from
[IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d).

MIT licensed.
