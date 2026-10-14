import importlib
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    out = tmp_path_factory.mktemp("artifacts")
    from churn import train
    train.main(["--data", str(ROOT / "data" / "telco_churn.csv"), "--out", str(out)])
    return out


@pytest.fixture(scope="session")
def client(artifacts):
    os.environ["CHURN_MODEL"] = str(artifacts / "model.joblib")
    from fastapi.testclient import TestClient
    import churn.api
    importlib.reload(churn.api)
    return TestClient(churn.api.app)
