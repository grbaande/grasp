"""Internals of pids_workbench_app.py, split by concern.

- report_io: locating and loading report directories written by the engine.
- anomaly_data: per-anomaly/occurrence lookups and rankings over report data.
- graph_loading: edge-vocabulary mapping and loading a raw PyG neighborhood
  for one process.
- graph_viz: rendering a single neighborhood (graph, event sequence, tables).
- diff_viz: rendering a target-vs-sample diff (graph, node/edge tables).
- page_create / page_overview / page_context_view: the three Streamlit pages.

pids_workbench_app.py itself stays the entry point: bootstrap imports,
st.set_page_config, and the sidebar page dispatch.
"""

from __future__ import annotations
