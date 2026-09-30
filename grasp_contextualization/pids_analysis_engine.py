"""
pids_analysis_engine.py

Single-metric PIDS engine: adaptive PyG Data-object distance + existing v9 visualization schema.

This is intentionally NOT a top-k/mean/max aggregation system.
For every anomaly and every candidate sample, the engine computes one direct PyG
neighborhood distance. The main metric is `pyg_final_mean`.

The engine's internals live in pids_engine/ (config/data types, path resolution,
file loading, sampling, the distance metric, and report writing, one concern per
module) — this file is just the CLI entry point that wires them together.

Outputs are compatible with the visual workbench:
- conclusion_summary.csv
- view_summary.csv
- pairwise_similarities.csv
- feature_reasons.csv
- sample_index.csv
- frequency_prior.csv
- learning_quality.csv
- prediction_confidence.csv
- summary_statistics_view.csv
- failed_anomalies.csv
- run_config.json
- pyg_distance_metric.json
- ablation_summary.csv
- ablation_component_importance.csv
- ablation_winner_summary.csv
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, cast

# Ensure the repo root is importable regardless of how this script is invoked
# (direct path, `python -m`, or shelled out to from pids_workbench_app.py).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import torch

from grasp.graph import graph_storage
from grasp_contextualization.pids_engine.io import (
    load_anomalies,
    load_learning_metrics,
    load_prediction_confidence,
)
from grasp_contextualization.pids_engine.models import Config, HopMode
from grasp_contextualization.pids_engine.paths import graph_storage_path
from grasp_contextualization.pids_engine.reporting import (
    analyze_one,
    concat_outputs,
    prediction_confidence_frame,
    write_outputs,
)
from grasp_contextualization.pids_engine.sampling import executable_window_stats

LOGGER = logging.getLogger(__name__)


def run(c: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    random.seed(c.seed)
    torch.manual_seed(c.seed)
    rng = random.Random(c.seed)
    gs: graph_storage.GraphStorage = graph_storage.load_graph_storage(str(graph_storage_path(c)))
    anomalies = load_anomalies(c)
    global_metrics, class_metrics = load_learning_metrics(c)
    stats = executable_window_stats(gs)
    confidence_by_node = load_prediction_confidence(c)
    parts: list[dict[str, pd.DataFrame]] = []
    failed: list[dict[str, Any]] = []
    for anom in anomalies:
        try:
            LOGGER.info("Analyzing anomaly %s", anom.process_id)
            parts.append(analyze_one(anom, gs, stats, class_metrics, global_metrics, c, rng))
        except Exception as exc:
            LOGGER.exception("Failed anomaly %s", anom.process_id)
            failed.append(
                {
                    "anomaly_id": anom.process_id,
                    "uuid": anom.uuid,
                    "true_label": anom.true_label,
                    "pred_label": anom.pred_label,
                    "error": repr(exc),
                }
            )
            if c.fail_fast:
                raise
    prediction_confidence = prediction_confidence_frame(anomalies, confidence_by_node, gs)
    write_outputs(c, concat_outputs(parts), failed, prediction_confidence)
    LOGGER.info("Wrote output to %s", c.output_dir)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Single adaptive PyG metric with visual report schema"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="atlasv2_edr")
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument(
        "--experiment-prefix",
        default=None,
        help="Filename prefix before _dataset-, e.g. cadets_e5_default_experiment_classic",
    )
    parser.add_argument("--context-size", type=int, default=120)
    parser.add_argument("--step-size", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=Path("pids_simple_metric_visual_report"))
    parser.add_argument("--candidate-pool-per-label", type=int, default=50)
    parser.add_argument("--max-anomalies", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hop-mode", choices=["one", "two"], default="two")
    parser.add_argument("--include-unknown-exec", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    return Config(
        args.data_dir,
        args.dataset,
        args.run_id,
        args.experiment_prefix,
        args.context_size,
        args.step_size,
        args.output_dir,
        args.candidate_pool_per_label,
        args.max_anomalies,
        args.seed,
        cast(HopMode, args.hop_mode),
        not args.include_unknown_exec,
        args.fail_fast,
    )


if __name__ == "__main__":
    run(parse_args())
