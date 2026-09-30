"""Internals of pids_analysis_engine.py, split by concern.

- models: config/data types shared across the engine (Config, Anomaly,
  SampleRef, view/ablation constants).
- paths: locating an experiment's report/graph-storage/classification files.
- io: reading those files into the engine's types.
- sampling: turning an anomaly into a target + candidate-sample pool drawn
  from training data, and loading each sample's PyG neighborhood.
- metric: the adaptive PyG neighborhood-distance metric and its
  leave-one-view-out ablations.
- reporting: turning per-anomaly analysis into the report's CSV/JSON output.

pids_analysis_engine.py itself stays the CLI entry point: argument parsing
and the top-level run() loop that wires these together.
"""

from __future__ import annotations
