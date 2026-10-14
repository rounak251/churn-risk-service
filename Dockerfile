FROM python:3.12.7-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl make && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY churn/ churn/
COPY Makefile .
# Model is trained at build time so the image is self-contained and reproducible.
RUN make train
EXPOSE 8000
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
