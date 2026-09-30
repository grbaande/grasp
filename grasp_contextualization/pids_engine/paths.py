"""Locating an experiment's report/graph-storage/classification files on disk."""

from __future__ import annotations

from pathlib import Path

from grasp_contextualization.shared import experiment_artifact_path

from .models import Config, experiment_prefix


def _resolve_artifact(c: Config, subdir: str, suffix: str) -> Path:
    prefix = experiment_prefix(c)
    exact = experiment_artifact_path(
        c.data_dir, c.dataset, prefix, c.context_size, c.step_size, subdir, suffix
    )
    pattern = (
        f"*_dataset-{c.dataset}_context_size-{c.context_size}_step_size-{c.step_size}_{suffix}"
    )
    return resolve_experiment_file(exact.parent, exact.name, pattern, prefix)


def report_path(c: Config) -> Path:
    return _resolve_artifact(c, "reports", "detailed_report.json")


def graph_storage_path(c: Config) -> Path:
    return _resolve_artifact(c, "graph_storage", "graph_storage.pt")


def cls_metrics_path(c: Config) -> Path:
    return _resolve_artifact(c, "classification_storage", "cls_storage_metrics.json")


def cls_predictions_path(c: Config) -> Path:
    return _resolve_artifact(c, "classification_storage", "test_cls_storage.pt")


def unknown_exec_path(c: Config) -> Path:
    return c.data_dir / "ground_truth" / c.dataset / "unknown_exec_processes.csv"


def resolve_experiment_file(
    directory: Path, exact_name: str, glob_pattern: str, experiment_prefix_value: str | None = None
) -> Path:
    """Resolve exact experiment file, with a readable fallback search.

    The exact filename is used first. If it is missing, the function scans the
    expected directory for files whose name still starts with the requested
    experiment prefix (numeric runs such as `atlasv2_edr_1`, or named ones
    such as `cadets_e5_default_experiment_classic`) in case only the exact
    suffix formatting differs. It does NOT fall back to a different
    experiment's file just because it happens to match the same
    dataset/context/step — that would silently analyze the wrong run.
    """
    exact = directory / exact_name
    if exact.exists():
        return exact
    candidates = sorted(directory.glob(glob_pattern))
    if experiment_prefix_value:
        prefixed = [
            p for p in candidates if p.name.startswith(f"{experiment_prefix_value}_dataset-")
        ]
        if prefixed:
            return prefixed[0]
    other_runs = (
        f" Found {len(candidates)} file(s) for this dataset/context/step from other experiment prefixes: {[p.name for p in candidates]}."
        if candidates
        else ""
    )
    raise FileNotFoundError(
        f"No file found for experiment prefix {experiment_prefix_value!r} in {directory} (expected {exact_name!r}).{other_runs}"
    )
