DATA_URL = https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

data/telco_churn.csv:
	mkdir -p data && curl -sL -o $@ $(DATA_URL)

train: data/telco_churn.csv
	python -m churn.train --data data/telco_churn.csv --out artifacts

serve: train
	uvicorn churn.api:app --port 8000

test: data/telco_churn.csv
	pytest -q tests

.PHONY: train serve test
