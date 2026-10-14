"""Per-customer drivers: which features push this customer's risk up or down.

Logistic regression: contribution = coefficient x (value - training mean) in the
model's transformed space, summed back onto the original feature.
LightGBM: native SHAP values (pred_contrib).
"""
import numpy as np

from .features import FEATURES


def fit_reference(twin, X_train):
    """Store what explanations are measured against (training mean)."""
    if hasattr(twin, "named_steps"):
        pre = twin.named_steps["columntransformer"]
        return {"mean": np.asarray(pre.transform(X_train).mean(axis=0)).ravel()}
    return {}


def contributions(twin, reference, X):
    """Array (n_rows, n_features) of log-odds contributions aligned with FEATURES."""
    if hasattr(twin, "named_steps"):
        pre = twin.named_steps["columntransformer"]
        coef = twin.named_steps["logisticregression"].coef_.ravel()
        Z = pre.transform(X)
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
        per_col = (Z - reference["mean"]) * coef
        names = pre.get_feature_names_out()
        out = np.zeros((len(X), len(FEATURES)))
        for j, name in enumerate(names):
            raw = name.split("__", 1)[1]
            feat = next(f for f in FEATURES if raw == f or raw.startswith(f + "_"))
            out[:, FEATURES.index(feat)] += per_col[:, j]
        return out
    return twin.predict_proba(X, pred_contrib=True)[:, : len(FEATURES)]


def top_drivers(twin, reference, X, k=3):
    C = contributions(twin, reference, X)
    result = []
    for row in C:
        order = np.argsort(-np.abs(row))[:k]
        result.append([{"feature": FEATURES[j], "effect": "raises risk" if row[j] > 0 else "lowers risk",
                        "log_odds": round(float(row[j]), 3)} for j in order])
    return result
