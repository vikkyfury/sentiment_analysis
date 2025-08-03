# Sentiment Analysis Pipeline

This repository provides a complete end-to-end sentiment analysis pipeline, from data ingestion and preprocessing to model training, evaluation, API serving, and containerization. It leverages DVC for data versioning, MLflow for experiment tracking, GitHub Actions for CI/CD, and Docker Compose for local development.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Data & Model Versioning](#data--model-versioning)
- [Pipeline Steps](#pipeline-steps)
- [Testing](#testing)
- [API Service](#api-service)
- [Docker & Docker Compose](#docker--docker-compose)
- [CI/CD](#cicd)
- [Quickstart](#quickstart)

## Features

- **DVC-driven pipeline**: Ingest, preprocess, train, and evaluate data with reproducible stages.  
- **MLflow integration**: Track experiments, parameters, metrics, and artifacts.  
- **Quality gating**: CI pipeline automatically fails if `f1_macro < 0.95`.  
- **Production model promotion**: Separate `models/prod` snapshot for approved models.  
- **Flask API**: Serve predictions via `/health` and `/predict` endpoints.  
- **Docker containerization**: Portable image built with Gunicorn.  
- **Docker Compose**: One-command local launch.  
- **GitHub Actions CI**: Automates DVC repro, tests, image build, smoke tests, and Docker registry publishing.

## Prerequisites

- Git  
- Python 3.9  
- [DVC](https://dvc.org/) (with S3 support)  
- [Docker](https://www.docker.com/) and Docker Compose  
- (Optional) AWS credentials for DVC remote storage  

## Installation

```bash
git clone https://github.com/<your-org>/sentiment_analysis.git
cd sentiment_analysis

# Setup virtual environment
python -m venv .venv && source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt "dvc[s3]" pytest
```
## Data & Model Versioning

This project uses DVC to version both raw & processed data and trained models.

- Data is stored in the `data/` directory and tracked with `.dvc` files.  
- Models are stored under `models/latest` for experiments and `models/prod` for production snapshots.  
- Remote storage (e.g., S3) is configured in `.dvc/config` under the `storage` remote.  

Configure AWS CLI or DVC local config to enable `dvc pull` and `dvc push`:

```bash
# AWS CLI (persisted)
aws configure

# Or DVC local config (not committed)
dvc remote modify --local storage access_key_id     YOUR_KEY_ID
dvc remote modify --local storage secret_access_key YOUR_SECRET_KEY
dvc remote modify --local storage region            us-east-2
```
## Pipeline Steps

Run the full pipeline with DVC:

```bash
dvc pull -v          # fetch data & model artifacts
dvc repro -v         # execute ingest -> train -> evaluate
```
This generates:

- `data/processed/cleaned_data.csv` (cleaned dataset)  
- `models/latest` (trained model & vectorizer)  
- `metrics.json` (evaluation metrics)  

To promote the current model to production:

```bash
dvc repro promote   # copies models/latest -> models/prod
```
## Testing
Run the test suite with:
```bash
pytest -q
```
## API Service
A Flask API serves predictions:
```bash
# Start the API (default port 8000)
python src/serve.py

# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts":["great movie","terrible plot"]}'
```
## Docker & Docker Compose
Docker:
```bash
docker build -t sentiment-api:local .
docker run --rm -p 8000:8000 sentiment-api:local
```
Docker Compose: create a docker-compose.yml file:
```bash
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
```
Launch with:
```bash
docker compose up --build
```
Stop with:
```bash
docker compose down
```
## CI/CD
The GitHub Actions workflow (.github/workflows/ci.yml) automates:
- DVC pull, checkout, repro

- Quality gate (F1 ≥ 0.95)

- Pytest suite

- Docker image build & smoke test

- Publish to GitHub Container Registry (GHCR)

- Promote model to models/prod on gate pass

## Quickstart
Follow these steps to get the repo running end-to-end:
```bash# 
1. Clone and enter
git clone https://github.com/<your-org>/sentiment_analysis.git
cd sentiment_analysis

# 2. Set up Python & DVC
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt "dvc[s3]" pytest

# 3. Fetch data & models
dvc pull -v

# 4. Run pipeline & tests
dvc repro -v
pytest -q

# 5. Start the API locally
docker compose up --build
# In another terminal:
#   curl http://localhost:8000/health
#   curl -X POST http://localhost:8000/predict \
#     -H "Content-Type: application/json" \
#     -d '{"texts":["great movie","terrible plot"]}'

# 6. Tear down
docker compose down
deactivate
```

That’s it 🎉