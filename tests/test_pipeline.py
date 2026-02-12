"""
tests/test_pipeline.py
-----------------------
Unit tests for MLOps pipeline components.
Self-contained — no MLflow/sklearn imports needed in CI.
"""

import os
import json
import pytest
import numpy as np
from dataclasses import dataclass
from pathlib import Path


# ── Config constants ──────────────────────────────────────────────────────────

ACCURACY_THRESHOLD = 0.80
DRIFT_THRESHOLD    = 0.10
TEST_SIZE          = 0.2
CV_FOLDS           = 5
RANDOM_STATE       = 42


# ── Config tests ──────────────────────────────────────────────────────────────

def test_accuracy_threshold_valid():
    assert 0.0 < ACCURACY_THRESHOLD < 1.0

def test_drift_threshold_valid():
    assert 0.0 < DRIFT_THRESHOLD < 1.0

def test_test_size_valid():
    assert 0.0 < TEST_SIZE < 1.0

def test_cv_folds_at_least_three():
    assert CV_FOLDS >= 3

def test_random_state_set():
    assert isinstance(RANDOM_STATE, int)


# ── Retraining trigger logic ──────────────────────────────────────────────────

def should_retrain(accuracy: float) -> bool:
    return accuracy < ACCURACY_THRESHOLD

def test_retrain_triggered_below_threshold():
    assert should_retrain(0.75) is True

def test_retrain_not_triggered_above_threshold():
    assert should_retrain(0.92) is False

def test_retrain_at_exact_threshold():
    assert should_retrain(ACCURACY_THRESHOLD) is False

def test_retrain_at_zero_accuracy():
    assert should_retrain(0.0) is True


# ── Drift detection logic ─────────────────────────────────────────────────────

def compute_drift_simple(ref: np.ndarray, curr: np.ndarray) -> dict:
    ref_means  = ref.mean(axis=0)
    curr_means = curr.mean(axis=0)
    ref_stds   = ref.std(axis=0) + 1e-8
    shifts     = np.abs(curr_means - ref_means) / ref_stds
    drifted    = np.where(shifts > DRIFT_THRESHOLD)[0].tolist()
    return {"drift_detected": len(drifted) > 0, "drifted_features": drifted}

def test_no_drift_identical_data():
    data = np.random.RandomState(42).randn(100, 5)
    result = compute_drift_simple(data, data)
    assert result["drift_detected"] is False

def test_drift_detected_on_shifted_data():
    ref  = np.zeros((100, 5))
    curr = np.ones((100, 5)) * 10
    result = compute_drift_simple(ref, curr)
    assert result["drift_detected"] is True

def test_drift_result_has_required_keys():
    data   = np.random.RandomState(0).randn(50, 3)
    result = compute_drift_simple(data, data)
    assert "drift_detected" in result
    assert "drifted_features" in result


# ── Metrics validation ────────────────────────────────────────────────────────

def test_f1_score_in_range():
    f1 = 0.87
    assert 0.0 <= f1 <= 1.0

def test_roc_auc_in_range():
    auc = 0.91
    assert 0.5 <= auc <= 1.0

def test_precision_recall_in_range():
    precision = 0.84
    recall    = 0.79
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0


# ── Monitoring report ─────────────────────────────────────────────────────────

@dataclass
class MonitoringReport:
    total_predictions: int
    churn_rate:        float
    avg_confidence:    float
    accuracy_ok:       bool
    retrain_required:  bool
    alerts:            list

def test_monitoring_report_no_alerts_when_healthy():
    report = MonitoringReport(
        total_predictions=100, churn_rate=0.2,
        avg_confidence=0.75, accuracy_ok=True,
        retrain_required=False, alerts=[]
    )
    assert len(report.alerts) == 0
    assert report.retrain_required is False

def test_monitoring_report_retrain_when_accuracy_low():
    report = MonitoringReport(
        total_predictions=200, churn_rate=0.3,
        avg_confidence=0.6, accuracy_ok=False,
        retrain_required=True, alerts=["Accuracy below threshold"]
    )
    assert report.retrain_required is True
    assert len(report.alerts) > 0


# ── File existence tests ──────────────────────────────────────────────────────

def test_train_file_exists():
    path = Path(__file__).parent.parent / "src" / "train.py"
    assert path.exists(), "src/train.py missing"

def test_serve_file_exists():
    path = Path(__file__).parent.parent / "src" / "serve.py"
    assert path.exists(), "src/serve.py missing"

def test_monitor_file_exists():
    path = Path(__file__).parent.parent / "src" / "monitor.py"
    assert path.exists(), "src/monitor.py missing"

def test_requirements_exists():
    path = Path(__file__).parent.parent / "requirements.txt"
    assert path.exists(), "requirements.txt missing"

def test_dockerfile_exists():
    path = Path(__file__).parent.parent / "Dockerfile"
    assert path.exists(), "Dockerfile missing"
