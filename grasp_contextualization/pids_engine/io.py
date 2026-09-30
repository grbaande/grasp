"""Reading an experiment's report/classification files into engine types."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch

from .models import Anomaly, ClassMetric, Config
from .paths import cls_metrics_path, cls_predictions_path, report_path, unknown_exec_path

LOGGER = logging.getLogger(__name__)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def load_anomalies(c: Config) -> list[Anomaly]:
    rows = load_json(report_path(c)).get("detailed_detection", {}).get("anomalies", [])
    if not isinstance(rows, list):
        raise ValueError("Expected detailed_detection.anomalies to be a list")  # noqa: TRY004
    if c.exclude_unknown_exec and unknown_exec_path(c).exists():
        unknown = pd.read_csv(unknown_exec_path(c), header=None, names=["uuid", "exec", "id"])
        unknown_uuids = set(unknown["uuid"].dropna().astype(str))
        rows = [r for r in rows if str(r.get("uuid")) not in unknown_uuids]
    out: list[Anomaly] = []
    for r in rows:
        try:
            out.append(
                Anomaly(
                    str(r["id"]),
                    str(r.get("uuid")) if r.get("uuid") is not None else None,
                    str(r["time_window"]),
                    str(r["true_label"]),
                    str(r["pred_label"]),
                )
            )
        except Exception:  # noqa: BLE001
            LOGGER.warning("Skipping malformed row: %s", r)
    return out[: c.max_anomalies] if c.max_anomalies is not None else out


def parse_classification_report(report: str) -> dict[int, ClassMetric]:
    pattern = re.compile(
        r"^\s*(\d+)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+)\s*$"
    )
    out: dict[int, ClassMetric] = {}
    for line in report.splitlines():
        m = pattern.match(line)
        if m:
            cid = int(m.group(1))
            out[cid] = {
                "class_id": cid,
                "precision": float(m.group(2)),
                "recall": float(m.group(3)),
                "f1": float(m.group(4)),
                "support": int(m.group(5)),
            }
    return out


def load_learning_metrics(c: Config) -> tuple[dict[str, float], dict[int, ClassMetric]]:
    p = cls_metrics_path(c)
    if not p.exists():
        return {}, {}
    payload = load_json(p)
    global_metrics = {
        k: float(payload.get(k, float("nan")))
        for k in ["accuracy", "precision", "recall", "f1_score", "macro_f1_score"]
    }
    return global_metrics, parse_classification_report(
        str(payload.get("classification_report", ""))
    )


def load_prediction_confidence(c: Config) -> dict[tuple[str, str], list[tuple[int, float]]]:
    """Map (node_id, window_path) -> ranked (class_id, probability) pairs.

    node_id (== process_id) alone is NOT unique: the same node can appear as a
    subject in multiple overlapping time-window snapshots, each getting its
    own separate classification with a potentially different result. window_path
    (== an anomaly's time_window/data_path) disambiguates which occurrence a
    given anomaly report entry actually refers to.
    """
    path = cls_predictions_path(c)
    if not path.exists():
        return {}
    data = torch.load(str(path), weights_only=True)
    node_ids = data.get("node_ids", [])
    window_paths = data.get("window_path", [])
    y_hat_proba = data.get("y_hat_proba", [])
    return {
        (str(node_id), str(window_path)): [(int(cid), float(prob)) for cid, prob in row]
        for node_id, window_path, row in zip(node_ids, window_paths, y_hat_proba)
    }
