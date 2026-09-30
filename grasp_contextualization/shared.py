"""Shared helpers for pids_analysis_engine.py and pids_workbench_app.py.

Deliberately has no dependency on the `grasp` package itself — this stays
local to grasp_contextualization so the two PIDS tools can be developed and
run independently of GRASP's core internals.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def neighbor_fanout(name_hint: str, hop_mode: str) -> list[int]:
    """NeighborLoader fanout for a dataset/run, keyed off a dataset name or path.

    Large EDR datasets (name/path containing e5, optc, or carbanak) can explode
    with full `-1` fanout. For these, the two-hop neighborhood is capped to
    [10000, 20]. Other datasets keep the exact full neighborhood behavior.
    """
    name = str(name_hint).lower()
    if any(token in name for token in ["e5", "optc", "carbanak"]):
        return [10000] if hop_mode == "one" else [10000, 20]
    return [-1] if hop_mode == "one" else [-1, -1]


def experiment_artifact_path(
    data_dir: Path,
    dataset: str,
    experiment_prefix: str,
    context_size: int,
    step_size: int,
    subdir: str,
    suffix: str,
) -> Path:
    """Build the expected path for a GRASP experiment output artifact.

    e.g. subdir="reports", suffix="detailed_report.json" ->
    <data_dir>/data/<dataset>/reports/<prefix>_dataset-<dataset>_context_size-<c>_step_size-<s>_detailed_report.json
    """
    return (
        data_dir
        / "data"
        / dataset
        / subdir
        / (
            f"{experiment_prefix}_dataset-{dataset}_context_size-{context_size}"
            f"_step_size-{step_size}_{suffix}"
        )
    )
