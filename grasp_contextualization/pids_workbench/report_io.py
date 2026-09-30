"""Locating and loading report directories written by pids_analysis_engine.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from grasp_contextualization.shared import REPO_ROOT

from .graph_loading import load_raw_batch

REPORT_FILES = [
    "conclusion_summary.csv",
    "view_summary.csv",
    "pairwise_similarities.csv",
    "feature_reasons.csv",
    "sample_index.csv",
    "frequency_prior.csv",
    "learning_quality.csv",
    "prediction_confidence.csv",
    "summary_statistics_view.csv",
    "ablation_summary.csv",
    "ablation_component_importance.csv",
    "ablation_winner_summary.csv",
    "failed_anomalies.csv",
]
REPORT_REQUIRED_FILES = [
    "conclusion_summary.csv",
    "view_summary.csv",
    "pairwise_similarities.csv",
    "sample_index.csv",
]

# -----------------------------------------------------------------------------
# File IO
# -----------------------------------------------------------------------------


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # e.g. failed_anomalies.csv when nothing failed: pd.DataFrame([]).to_csv()
        # writes a single newline (1 byte, so the size check above doesn't catch
        # it), which has no header for read_csv to parse. That's a legitimate
        # "no rows" file, not a corrupt one, so don't warn about it.
        return pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not read {path.name}: {exc}")
        return pd.DataFrame()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


@st.cache_data(show_spinner=False)
def load_report(report_dir: str) -> dict[str, pd.DataFrame]:
    root = Path(report_dir).expanduser().resolve()
    return {name: safe_read_csv(root / name) for name in REPORT_FILES}


def safe_len_csv(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:  # noqa: BLE001
        return 0


# -----------------------------------------------------------------------------
# Report discovery
# -----------------------------------------------------------------------------


def report_dir_score(path: Path) -> int:
    return sum(1 for name in REPORT_REQUIRED_FILES if (path / name).exists())


def is_report_dir(path: Path) -> bool:
    return path.is_dir() and report_dir_score(path) >= 2 and (path / "sample_index.csv").exists()


def report_metadata(path: Path) -> dict[str, Any]:
    run_config = read_json_if_exists(path / "run_config.json")
    pyg_metric = read_json_if_exists(path / "pyg_distance_metric.json")
    return {
        "path": str(path),
        "name": path.name,
        "dataset": run_config.get("dataset", "?"),
        "run_id": run_config.get("run_id", "?"),
        "hop_mode": run_config.get("hop_mode", "?"),
        "candidate_pool": run_config.get("candidate_pool_per_label", "?"),
        "metric": pyg_metric.get("name", "legacy"),
        "anomalies": safe_len_csv(path / "conclusion_summary.csv"),
        "samples": safe_len_csv(path / "sample_index.csv"),
        "summary_rows": safe_len_csv(path / "summary_statistics_view.csv"),
        "score": report_dir_score(path),
    }


def default_search_roots() -> list[Path]:
    roots: list[Path] = []
    for item in os.environ.get("PIDS_REPORT_ROOTS", "").split(":"):
        if item.strip():
            roots.append(Path(item.strip()).expanduser())
    roots.extend(
        [
            Path.cwd(),
            REPO_ROOT,
            REPO_ROOT / "grasp_contextualization",
            REPO_ROOT / "pids_runs",
        ]
    )
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:  # noqa: BLE001
            key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


@st.cache_data(show_spinner=False)
def discover_reports_cached(root_strings: tuple[str, ...], max_depth: int = 6) -> pd.DataFrame:
    candidates: dict[str, dict[str, Any]] = {}
    for root_str in root_strings:
        root = Path(root_str).expanduser()
        if not root.exists():
            continue
        if is_report_dir(root):
            candidates[str(root.resolve())] = report_metadata(root.resolve())
        try:
            for sample_file in root.rglob("sample_index.csv"):
                parent = sample_file.parent.resolve()
                try:
                    rel_len = len(parent.relative_to(root.resolve()).parts)
                    if rel_len > max_depth:
                        continue
                except Exception:  # noqa: BLE001, S110
                    pass
                if is_report_dir(parent):
                    candidates[str(parent)] = report_metadata(parent)
        except Exception:  # noqa: BLE001, S112
            continue
    if not candidates:
        return pd.DataFrame(columns=["path", "label"])
    df = pd.DataFrame(list(candidates.values()))
    df["label"] = df.apply(
        lambda r: (
            f"{r['name']} | dataset={r['dataset']} | anomalies={r['anomalies']} | samples={r['samples']} | metric={r['metric']} | {r['path']}"
        ),
        axis=1,
    )
    return df.sort_values(["dataset", "name", "path"], ascending=[True, True, True]).reset_index(
        drop=True
    )


def choose_report_dir(key: str, default: str = str(REPO_ROOT / "pids_runs")) -> str:
    st.sidebar.markdown("### Experiment results")
    if "selected_report_dir" not in st.session_state:
        st.session_state["selected_report_dir"] = default
    roots_text = st.sidebar.text_area(
        "Report search roots",
        value="\n".join(str(p) for p in default_search_roots()),
        key=f"{key}_report_roots",
        height=100,
    )
    roots = tuple(line.strip() for line in roots_text.splitlines() if line.strip())
    c1, c2 = st.sidebar.columns(2)
    with c1:
        if st.button("Refresh", key=f"{key}_refresh_reports"):
            discover_reports_cached.clear()
            load_report.clear()
            load_raw_batch.clear()
    with c2:
        show_table = st.checkbox("Table", value=False, key=f"{key}_show_report_table")
    found = discover_reports_cached(roots)
    current = str(st.session_state.get("selected_report_dir", default))
    manual = st.sidebar.text_input("Manual report dir", current, key=f"{key}_manual_report")
    if found.empty:
        st.session_state["selected_report_dir"] = manual
        return manual
    options = ["<manual path>"] + found["label"].tolist()
    idx = 0
    matches = found.index[found["path"].astype(str) == current].tolist()
    if matches:
        idx = int(matches[0]) + 1
    selected = st.sidebar.selectbox(
        "Choose report", options, index=idx, key=f"{key}_selected_report"
    )
    if show_table:
        st.sidebar.dataframe(
            found.drop(columns=["label"], errors="ignore"), use_container_width=True
        )
    if selected == "<manual path>":
        st.session_state["selected_report_dir"] = manual
        return manual
    path = str(found.loc[found["label"] == selected, "path"].iloc[0])
    st.session_state["selected_report_dir"] = path
    return path
