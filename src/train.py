"""
src/train.py
-------------
Model training pipeline with MLflow experiment tracking,
cross-validation, and model registry integration.
Dataset: data/churn_data.csv if present, otherwise a synthetic churn dataset
(10,000 rows x 15 features via sklearn make_classification).
"""

from __future__ import annotations
import os
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME  = "mlops-churn-prediction"
REGISTERED_NAME  = "ChurnClassifier"
MLFLOW_DIR       = "mlruns"
RANDOM_STATE     = 42
TEST_SIZE        = 0.2
CV_FOLDS         = 5
ACCURACY_THRESHOLD = 0.80   # Trigger retraining if accuracy drops below this

CANDIDATE_MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=RANDOM_STATE
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1,
        max_depth=4, random_state=RANDOM_STATE
    ),
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_path: str = "data/churn_data.csv") -> tuple:
    """
    Load dataset. Uses CSV if available, else generates synthetic data.
    Returns X_train, X_test, y_train, y_test.
    """
    if Path(data_path).exists():
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        X  = df.drop(columns=["target"]).values
        y  = df["target"].values
    else:
        logger.info("Generating synthetic classification dataset (10k samples, 15 features)")
        X, y = make_classification(
            n_samples=10_000, n_features=15, n_informative=10,
            n_redundant=3, n_classes=2, weights=[0.7, 0.3],
            random_state=RANDOM_STATE
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_evaluate(
    model_name: str,
    model,
    X_train, X_test, y_train, y_test,
) -> dict:
    """
    Train a single model, log to MLflow, return metrics dict.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name) as run:
        # Build pipeline with scaling
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  model),
        ])

        # Cross-validation
        cv_scores = cross_val_score(pipe, X_train, y_train,
                                    cv=CV_FOLDS, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std":  round(cv_scores.std(), 4),
        }

        # Log params
        mlflow.log_param("model_name",   model_name)
        mlflow.log_param("cv_folds",     CV_FOLDS)
        mlflow.log_param("test_size",    TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            registered_model_name=REGISTERED_NAME,
        )

        logger.info(
            f"{model_name} | AUC: {metrics['roc_auc']:.4f} | "
            f"F1: {metrics['f1']:.4f} | CV-AUC: {metrics['cv_roc_auc_mean']:.4f}"
        )

        metrics["run_id"]     = run.info.run_id
        metrics["model_name"] = model_name
        return metrics


# ── Model selection ───────────────────────────────────────────────────────────

def select_best_model(all_metrics: list[dict]) -> dict:
    """Select best model by ROC-AUC score."""
    best = max(all_metrics, key=lambda m: m["roc_auc"])
    logger.info(
        f"Best model: {best['model_name']} "
        f"(AUC={best['roc_auc']:.4f}, F1={best['f1']:.4f})"
    )
    return best


# ── Retraining trigger ────────────────────────────────────────────────────────

def should_retrain(current_accuracy: float) -> bool:
    """Return True if accuracy dropped below threshold."""
    trigger = current_accuracy < ACCURACY_THRESHOLD
    if trigger:
        logger.warning(
            f"Accuracy {current_accuracy:.4f} < threshold {ACCURACY_THRESHOLD}. "
            "Retraining triggered."
        )
    return trigger


# ── Main ──────────────────────────────────────────────────────────────────────

def run_training_pipeline():
    mlflow.set_tracking_uri(MLFLOW_DIR)
    logger.info(f"MLflow tracking URI: {MLFLOW_DIR}")

    X_train, X_test, y_train, y_test = load_data()

    all_metrics = []
    for name, model in CANDIDATE_MODELS.items():
        metrics = train_and_evaluate(
            name, model, X_train, X_test, y_train, y_test
        )
        all_metrics.append(metrics)

    best = select_best_model(all_metrics)

    # Save results
    Path("models").mkdir(exist_ok=True)
    results_path = "models/training_results.json"
    with open(results_path, "w") as f:
        json.dump({"best_model": best, "all_runs": all_metrics}, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    return best


if __name__ == "__main__":
    run_training_pipeline()
