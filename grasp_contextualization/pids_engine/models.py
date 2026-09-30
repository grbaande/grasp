"""Config/data types and metric-view constants shared across the engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

Role = Literal["anomaly", "true_sample", "pred_sample"]
HopMode = Literal["one", "two"]

BASE_VIEWS = ["pyg_x_active", "pyg_edge_attr_active", "pyg_size_profile", "pyg_cmd_label_profile"]
FINAL_VIEW = "pyg_final_mean"
ABLATION_VIEWS = {
    "pyg_final_minus_x": "pyg_x_active",
    "pyg_final_minus_edge_attr": "pyg_edge_attr_active",
    "pyg_final_minus_size": "pyg_size_profile",
    "pyg_final_minus_cmd_label": "pyg_cmd_label_profile",
}
ABLATION_LABELS = {
    "pyg_final_minus_x": r"d_PyG^{-x}",
    "pyg_final_minus_edge_attr": r"d_PyG^{-e}",
    "pyg_final_minus_size": r"d_PyG^{-z}",
    "pyg_final_minus_cmd_label": r"d_PyG^{-c}",
}
ALL_VIEWS = [FINAL_VIEW] + list(ABLATION_VIEWS.keys()) + BASE_VIEWS
TOP_DIFFS = 15


class ExecWindowStats(TypedDict):
    train_windows: list[str]
    counts: list[int]
    ids: list[list[str]]


class ClassMetric(TypedDict):
    class_id: int
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class Config:
    data_dir: Path
    dataset: str = "atlasv2_edr"
    run_id: int = 1
    experiment_prefix: str | None = None
    context_size: int = 120
    step_size: int = 120
    output_dir: Path = Path("pids_simple_metric_visual_report")
    candidate_pool_per_label: int = 50
    max_anomalies: int | None = None
    seed: int = 7
    hop_mode: HopMode = "two"
    exclude_unknown_exec: bool = True
    fail_fast: bool = False


@dataclass(frozen=True, slots=True)
class Anomaly:
    process_id: str
    uuid: str | None
    data_path: str
    true_label: str
    pred_label: str


@dataclass(frozen=True, slots=True)
class SampleRef:
    anomaly_id: str
    sample_name: str
    process_id: str
    data_path: str
    role: Role
    label: str
    anomaly_time_window: str
    """The anomaly occurrence's own time_window. Same anomaly_id (process_id)
    can recur across multiple overlapping windows with separate detections;
    this disambiguates which occurrence a row belongs to. Distinct from
    `data_path`, which for pred_sample/true_sample roles is the *candidate's*
    training window instead."""


def experiment_prefix(c: Config) -> str:
    """Filename prefix before `_dataset-...`.

    Supports numeric GRASP runs such as `atlasv2_edr_1` and named experiments
    such as `cadets_e5_default_experiment_classic`.
    """
    return c.experiment_prefix.strip() if c.experiment_prefix else f"{c.dataset}_{c.run_id}"
