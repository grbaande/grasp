"""
pids_workbench_app.py

Streamlit UI for pids_analysis_engine.py. Contextualizes a reported GRASP
anomaly against sampled training executables for its predicted and true
labels, visualizes the neighborhoods and their distance-metric comparison,
and gives an analyst a first-pass view to decide whether the anomaly is a
false positive or a real one.

The app's internals live in pids_workbench/ (report IO, per-anomaly data
helpers, graph loading, graph/diff visualization, and the three pages, one
concern per module) — this file is just the bootstrap + page dispatch.

Pages:
- Create report: run pids_analysis_engine.py against a GRASP experiment.
- Overview: aggregate metrics/ablation across all anomalies in a report.
- Context View: the analyst's decision view for one anomaly at a time —
  TARGET vs. its closest predicted-label and true-label training samples.

Run (from the repo root):
  streamlit run grasp_contextualization/pids_workbench_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# -----------------------------------------------------------------------------
# Bootstrap imports
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

ROOTS = [
    Path(os.environ.get("GRASP_PROJECT_ROOT", ""))
    if os.environ.get("GRASP_PROJECT_ROOT")
    else None,
    REPO_ROOT,
    Path.cwd(),
    *Path(__file__).resolve().parents,
]
for root in ROOTS:
    if root is not None and root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))

st.set_page_config(page_title="PIDS Context Workbench", page_icon="PIDS", layout="wide")

from grasp_contextualization.pids_workbench.page_context_view import page_context_view
from grasp_contextualization.pids_workbench.page_create import page_create
from grasp_contextualization.pids_workbench.page_overview import page_overview

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

page = st.sidebar.radio(
    "Page",
    [
        "Create report",
        "Overview",
        "Context View",
    ],
)

if page == "Create report":
    page_create()
elif page == "Overview":
    page_overview()
else:
    page_context_view()
