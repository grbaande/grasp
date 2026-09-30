"""'Create report' page: discover raw experiment runs and shell out to
pids_analysis_engine.py."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from grasp_contextualization.shared import REPO_ROOT, experiment_artifact_path

from .report_io import discover_reports_cached, load_report, read_json_if_exists

RUN_RE = re.compile(
    r"^(?P<prefix>.+?)_dataset-(?P<dataset>.+?)_context_size-(?P<context_size>\d+)_step_size-(?P<step_size>\d+)_detailed_report\.json$"
)


def graph_storage_file_for_run(
    data_dir: Path, dataset: str, experiment_prefix: str, context_size: int, step_size: int
) -> Path:
    return experiment_artifact_path(
        data_dir,
        dataset,
        experiment_prefix,
        context_size,
        step_size,
        "graph_storage",
        "graph_storage.pt",
    )


def cls_metrics_file_for_run(
    data_dir: Path, dataset: str, experiment_prefix: str, context_size: int, step_size: int
) -> Path:
    return experiment_artifact_path(
        data_dir,
        dataset,
        experiment_prefix,
        context_size,
        step_size,
        "classification_storage",
        "cls_storage_metrics.json",
    )


@st.cache_data(show_spinner=False)
def discover_experiment_runs_cached(data_dir_str: str) -> pd.DataFrame:
    user_path = Path(data_dir_str).expanduser()
    scan_roots: list[Path] = []
    if (user_path / "data").exists():
        scan_roots.append(user_path / "data")
    if user_path.name == "data":
        scan_roots.append(user_path)
    if not scan_roots:
        scan_roots.extend([user_path / "data", user_path])
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in scan_roots:
        for report in root.glob("*/reports/*_detailed_report.json"):
            if str(report.resolve()) in seen:
                continue
            seen.add(str(report.resolve()))
            match = RUN_RE.match(report.name)
            if not match:
                continue
            dataset = match.group("dataset")
            experiment_prefix = match.group("prefix")
            context_size = int(match.group("context_size"))
            step_size = int(match.group("step_size"))
            run_match = re.search(r"_(\d+)$", experiment_prefix)
            run_id = int(run_match.group(1)) if run_match else 1
            data_root = root.parent if root.name == "data" else root
            graph_file = graph_storage_file_for_run(
                data_root, dataset, experiment_prefix, context_size, step_size
            )
            cls_file = cls_metrics_file_for_run(
                data_root, dataset, experiment_prefix, context_size, step_size
            )
            anomalies = (
                read_json_if_exists(report).get("detailed_detection", {}).get("anomalies", [])
            )
            rows.append(
                {
                    "dataset": dataset,
                    "experiment_prefix": experiment_prefix,
                    "run_id": run_id,
                    "context_size": context_size,
                    "step_size": step_size,
                    "anomalies": len(anomalies) if isinstance(anomalies, list) else 0,
                    "report_present": report.exists(),
                    "graph_storage_present": graph_file.exists(),
                    "cls_metrics_present": cls_file.exists(),
                    "all_required_present": report.exists() and graph_file.exists(),
                    "data_dir": str(data_root),
                    "report": str(report),
                    "graph_storage": str(graph_file),
                    "classification_metrics": str(cls_file),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["label"])
    df = (
        pd.DataFrame(rows)
        .sort_values(["dataset", "experiment_prefix", "context_size", "step_size"])
        .reset_index(drop=True)
    )
    df["label"] = df.apply(
        lambda r: (
            f"{r['dataset']} | prefix={r['experiment_prefix']} | ctx={r['context_size']} | step={r['step_size']} | anomalies={r['anomalies']} | report={'yes' if r['report_present'] else 'NO'} | graph={'yes' if r['graph_storage_present'] else 'NO'} | cls={'yes' if r['cls_metrics_present'] else 'NO'}"
        ),
        axis=1,
    )
    return df


def page_create() -> None:
    st.header("Create Contextualization Report")
    st.caption("Runs pids_analysis_engine.py and writes its CSV/JSON report schema.")
    data_dir = Path(st.text_input("Data dir", str(REPO_ROOT)))
    if st.button("Refresh raw runs"):
        discover_experiment_runs_cached.clear()
    runs = discover_experiment_runs_cached(str(data_dir))
    selected = None
    if not runs.empty:
        label = st.selectbox("Choose raw experiment run", runs["label"].tolist())
        selected = runs[runs["label"] == label].iloc[0].to_dict()
    dataset_default = str(selected["dataset"]) if selected else "atlasv2_edr"
    experiment_prefix_default = (
        str(selected.get("experiment_prefix", f"{dataset_default}_1"))
        if selected
        else f"{dataset_default}_1"
    )
    run_id_default = int(selected["run_id"]) if selected else 1
    context_default = int(selected["context_size"]) if selected else 120
    step_default = int(selected["step_size"]) if selected else 120
    selected_data_dir_default = (
        str(selected.get("data_dir", data_dir)) if selected else str(data_dir)
    )
    with st.form("create_report"):
        c1, c2, c3 = st.columns(3)
        with c1:
            dataset = st.text_input("Dataset", dataset_default)
            experiment_prefix = st.text_input("Experiment prefix", experiment_prefix_default)
            run_id = st.number_input("Run ID fallback", min_value=1, value=run_id_default)
            context_size = st.number_input("Context size", min_value=1, value=context_default)
            step_size = st.number_input("Step size", min_value=1, value=step_default)
        with c2:
            pool = st.number_input("Candidate pool per label", min_value=1, value=5)
            max_anoms = st.text_input("Max anomalies empty=all", "")
            hop = st.selectbox("Hop", ["two", "one"])
        with c3:
            output_base = st.text_input("Output base dir", str(REPO_ROOT / "pids_runs"))
            output_dir = Path(
                st.text_input(
                    "Output dir",
                    str(
                        Path(output_base)
                        / f"{experiment_prefix}_pygdist_pool{int(pool)}_max{max_anoms or 'full'}"
                    ),
                )
            )
            engine = st.text_input(
                "Engine script",
                str(REPO_ROOT / "grasp_contextualization" / "pids_analysis_engine.py"),
            )
            include_unknown = st.checkbox("Include unknown exec", False)
        submitted = st.form_submit_button("Run analysis")
    if not submitted:
        if not runs.empty:
            st.subheader("Raw run availability")
            st.dataframe(runs.drop(columns=["label"], errors="ignore"), use_container_width=True)
        return
    effective_data_dir = Path(selected_data_dir_default)
    cmd = [
        sys.executable,
        engine,
        "--data-dir",
        str(effective_data_dir),
        "--dataset",
        dataset,
        "--run-id",
        str(int(run_id)),
        "--experiment-prefix",
        experiment_prefix,
        "--context-size",
        str(int(context_size)),
        "--step-size",
        str(int(step_size)),
        "--output-dir",
        str(output_dir),
        "--candidate-pool-per-label",
        str(int(pool)),
        "--hop-mode",
        hop,
    ]
    if max_anoms.strip():
        cmd += ["--max-anomalies", max_anoms.strip()]
    if include_unknown:
        cmd += ["--include-unknown-exec"]
    with st.status("Running analysis", expanded=True) as status:
        st.code(" ".join(cmd))
        try:
            subprocess.run(cmd, check=True, cwd=str(effective_data_dir))
            st.session_state["selected_report_dir"] = str(output_dir)
            discover_reports_cached.clear()
            load_report.clear()
            status.update(label="Done", state="complete")
            st.success(f"Wrote report to {output_dir}")
        except Exception as exc:  # noqa: BLE001
            status.update(label="Failed", state="error")
            st.exception(exc)
