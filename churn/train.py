"""Train, validate and package the churn model.

python -m churn.train --data data/telco_churn.csv --out artifacts
"""
import argparse
import json
import platform
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .economics import Economics, realised_profit
from .explain import fit_reference
from .features import CATEGORICAL, DERIVED, NUMERIC, prepare, target

SEED = 42


def make_logreg():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC + DERIVED),
    ])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))


def make_lgbm():
    return lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=SEED, deterministic=True, n_jobs=1, verbose=-1,
    )


def calibrated(model):
    return CalibratedClassifierCV(model, method="isotonic", cv=5)


def scores(y, p):
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/telco_churn.csv")
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data)
    X, y = prepare(raw), target(raw)
    # The test split is touched exactly once, at the end.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    candidates = {"logistic_regression": make_logreg(), "lightgbm": make_lgbm()}
    cv_report, oof = {}, {}
    for name, model in candidates.items():
        oof[name] = cross_val_predict(calibrated(model), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        cv_report[name] = scores(y_tr, oof[name])
    best = max(cv_report, key=lambda n: cv_report[n]["roc_auc"])

    econ = Economics()
    final = calibrated(candidates[best]).fit(X_tr, y_tr)
    p_te = final.predict_proba(X_te)[:, 1]

    # Contact policy: anyone whose expected profit is positive (p > break-even).
    contact_model = p_te > econ.break_even
    policies = {
        "contact_nobody": realised_profit(y_te, np.zeros_like(y_te, bool), econ),
        "contact_everyone": realised_profit(y_te, np.ones_like(y_te, bool), econ),
        "threshold_0.5": realised_profit(y_te, p_te > 0.5, econ),
        "model_break_even": realised_profit(y_te, contact_model, econ),
    }

    # Uncalibrated twin of the selected model, used only to explain predictions.
    explainer = candidates[best].fit(X_tr, y_tr)
    reference = fit_reference(explainer, X_tr)

    metrics = {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)), "churn_rate": round(float(y.mean()), 4),
        "cv_5fold_train": cv_report, "selected_model": best,
        "test": scores(y_te, p_te),
        "economics": {**econ.__dict__, "break_even_probability": round(econ.break_even, 4)},
        "test_profit_by_policy": {k: round(v, 2) for k, v in policies.items()},
        "test_customers_contacted": int(contact_model.sum()),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "lightgbm": lgb.__version__},
    }
    joblib.dump({"model": final, "explainer": explainer, "reference": reference, "economics": econ, "metrics": metrics}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
