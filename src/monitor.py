"""
src/monitor.py
---------------
Production model monitoring: performance tracking, data drift
detection, and automated retraining triggers.
"""

from __future__ import annotations
import json
import logging
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACCURACY_THRESHOLD  = 0.80
DRIFT_THRESHOLD     = 0.10   # Max allowed mean shift per feature
MONITOR_LOG         = "models/monitoring_log.json"


@dataclass
class MonitoringReport:
    timestamp:          str
    total_predictions:  int
    churn_rate:         float
    avg_confidence:     float
    avg_latency_ms:     float
    accuracy_ok:        bool
    drift_detected:     bool
    retrain_required:   bool
    alerts:             list


def compute_drift(reference: np.ndarray, current: np.ndarray) -> dict:
    """
    Simple mean-shift drift detection.
    Compares feature means between reference (training) and current (serving) data.
    Returns per-feature drift and overall drift flag.
    """
    if reference.shape[1] != current.shape[1]:
        return {"error": "Feature dimension mismatch"}

    ref_means  = reference.mean(axis=0)
    curr_means = current.mean(axis=0)
    ref_stds   = reference.std(axis=0) + 1e-8

    normalized_shifts = np.abs(curr_means - ref_means) / ref_stds
    drifted_features  = np.where(normalized_shifts > DRIFT_THRESHOLD)[0].tolist()

    return {
        "drifted_features":   drifted_features,
        "drift_detected":     len(drifted_features) > 0,
        "max_shift":          round(float(normalized_shifts.max()), 4),
        "mean_shift":         round(float(normalized_shifts.mean()), 4),
        "normalized_shifts":  normalized_shifts.round(4).tolist(),
    }


def monitor_predictions(prediction_log: list, accuracy: float | None = None) -> MonitoringReport:
    """
    Analyze prediction logs and generate monitoring report.
    prediction_log: list of {prediction, probability, latency_ms, timestamp}
    """
    alerts = []

    if not prediction_log:
        return MonitoringReport(
            timestamp=datetime.utcnow().isoformat(),
            total_predictions=0, churn_rate=0.0,
            avg_confidence=0.0, avg_latency_ms=0.0,
            accuracy_ok=True, drift_detected=False,
            retrain_required=False, alerts=["No predictions to monitor"]
        )

    preds      = [p["prediction"]  for p in prediction_log]
    probs      = [p["probability"] for p in prediction_log]
    latencies  = [p["latency_ms"]  for p in prediction_log]

    churn_rate    = round(sum(preds) / len(preds), 4)
    avg_conf      = round(sum(probs) / len(probs), 4)
    avg_latency   = round(sum(latencies) / len(latencies), 2)
    max_latency   = max(latencies)

    # Accuracy check
    accuracy_ok = True
    if accuracy is not None and accuracy < ACCURACY_THRESHOLD:
        accuracy_ok = False
        alerts.append(
            f"Accuracy {accuracy:.4f} below threshold {ACCURACY_THRESHOLD}"
        )

    # Latency alert
    if avg_latency > 200:
        alerts.append(f"High avg latency: {avg_latency}ms")

    # Confidence drift alert
    if avg_conf < 0.55 or avg_conf > 0.95:
        alerts.append(f"Unusual avg confidence: {avg_conf:.4f}")

    retrain_required = not accuracy_ok

    report = MonitoringReport(
        timestamp         = datetime.utcnow().isoformat(),
        total_predictions = len(prediction_log),
        churn_rate        = churn_rate,
        avg_confidence    = avg_conf,
        avg_latency_ms    = avg_latency,
        accuracy_ok       = accuracy_ok,
        drift_detected    = False,
        retrain_required  = retrain_required,
        alerts            = alerts,
    )

    # Persist report
    Path("models").mkdir(exist_ok=True)
    log_path = Path(MONITOR_LOG)
    history  = json.loads(log_path.read_text()) if log_path.exists() else []
    history.append(asdict(report))
    log_path.write_text(json.dumps(history, indent=2))

    if alerts:
        logger.warning(f"Monitoring alerts: {alerts}")
    else:
        logger.info("Monitoring: all metrics within normal range.")

    return report
