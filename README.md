# ⚙️ MLOps Pipeline with Model Registry & Monitoring

End-to-end **MLOps pipeline** for classification model development, experiment tracking, model registry, production serving, and automated monitoring with retraining triggers.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.15-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.112-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![CI](https://github.com/yourgithub/mlops-pipeline/actions/workflows/ci.yml/badge.svg)

---

## Architecture

```
Raw Data
   │
   ▼
Data Loading & Splitting (80/20 stratified)
   │
   ▼
Model Training (3 candidates: LR, RF, GBM)
   │  ├── Cross-validation (5-fold, ROC-AUC)
   │  ├── MLflow Experiment Tracking
   │  └── MLflow Model Registry
   ▼
Best Model Selection (by ROC-AUC)
   │
   ▼
FastAPI Inference Server
   │  ├── /predict  — real-time predictions
   │  ├── /health   — model load status
   │  └── /metrics  — prediction monitoring
   ▼
Monitoring Layer
   ├── Performance tracking (accuracy, latency)
   ├── Data drift detection (mean-shift)
   └── Automated retraining trigger (accuracy < 0.80)
```

## Features

- **Multi-model comparison** — Logistic Regression, Random Forest, Gradient Boosting
- **MLflow tracking** — every experiment logged with params, metrics, artifacts
- **Model registry** — best model registered and versioned in MLflow
- **FastAPI serving** — `/predict`, `/health`, `/metrics` endpoints
- **Drift detection** — normalized mean-shift per feature with configurable threshold
- **Retraining trigger** — automatic flag when accuracy drops below threshold
- **Docker ready** — containerized for production deployment
- **GitHub Actions CI** — automated testing on every push

## Quickstart

```bash
git clone https://github.com/yourgithub/mlops-pipeline.git
cd mlops-pipeline
pip install -r requirements.txt

# Train all models and register best to MLflow
python -m src.train

# View MLflow UI
mlflow ui --backend-store-uri mlruns

# Start inference server
uvicorn src.serve:app --reload

# Run tests
pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Real-time prediction with confidence score |
| `/health` | GET | Model load status and request count |
| `/metrics` | GET | Churn rate, avg latency, recent predictions |

## Docker

```bash
docker build -t mlops-pipeline .
docker run -p 8000:8000 mlops-pipeline
```

## Project Structure

```
mlops-pipeline/
├── src/
│   ├── train.py       # Training pipeline + MLflow tracking + model registry
│   ├── serve.py       # FastAPI inference server with monitoring
│   └── monitor.py     # Drift detection + retraining trigger
├── tests/
│   └── test_pipeline.py  # Self-contained unit tests
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

## Author

**Mayur Sangle** — M.S. Data Science, University of Maryland
[LinkedIn](https://linkedin.com/in/mayur-sangle-04m) • [GitHub](https://github.com/Mayur074)
