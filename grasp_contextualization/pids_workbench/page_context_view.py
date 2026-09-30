"""'Context View' page: the analyst's decision view for one anomaly occurrence
at a time — TARGET vs. its closest predicted-label and true-label training
samples."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st

from grasp_contextualization.shared import REPO_ROOT

from .anomaly_data import (
    available_anomalies,
    available_occurrences,
    available_views,
    df_for_anomaly_occurrence,
    format_time_window,
    pick_ranked_sample,
    ranked_neighbors,
    target_row,
)
from .diff_viz import render_diff_pair
from .graph_loading import load_neighborhood_or_error, resolve_op_map_json
from .graph_viz import (
    filter_graph,
    render_graph_grid,
    render_node_edge_tables,
    render_sequence_tables,
)
from .report_io import choose_report_dir, load_report, read_json_if_exists


def visual_settings(prefix: str, default_dataset: str = "atlasv2_edr") -> tuple[str, str, str, str]:
    c1, c2, c3 = st.columns(3)
    with c1:
        dataset = st.text_input("Dataset", default_dataset, key=f"{prefix}_dataset")
        project_root = st.text_input("Project root", str(REPO_ROOT), key=f"{prefix}_project_root")
    with c2:
        vocab = st.selectbox(
            "Edge vocabulary",
            ["auto", "atlasv2", "carbanakv2", "spade_grasp", "sysdig", "tc", "optc", "none"],
            key=f"{prefix}_vocab",
        )
    with c3:
        hop = st.selectbox("Hop", ["two", "one"], key=f"{prefix}_hop")
    return dataset, project_root, vocab, hop


def shared_graph_filters(
    key_prefix: str, edge_frames: list[pd.DataFrame]
) -> tuple[str, str, list[int], list[str], list[str], bool, bool, int]:
    with st.expander("Shared display filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            label_mode = st.selectbox(
                "Node label mode",
                ["full executable wrapped", "kind + executable", "node id", "local index only"],
                key=f"{key_prefix}_label_mode",
            )
            exec_query = st.text_input(
                "Filter executable/node contains", "", key=f"{key_prefix}_exec_query"
            )
        with f2:
            hops = st.multiselect(
                "Show hops", [0, 1, 2, -1], default=[0, 1, 2], key=f"{key_prefix}_hops"
            )
            kinds = st.multiselect(
                "Node kinds",
                ["target", "subject", "file", "netflow", "memory", "other"],
                default=["target", "subject", "file", "netflow", "memory", "other"],
                key=f"{key_prefix}_kinds",
            )
        with f3:
            all_types: set[str] = set()
            for ef in edge_frames:
                if not ef.empty and "edge_type" in ef.columns:
                    all_types |= set(ef["edge_type"].astype(str))
            edge_types = st.multiselect(
                "Edge types",
                sorted(all_types),
                default=sorted(all_types),
                key=f"{key_prefix}_edge_types",
            )
            target_only = st.checkbox(
                "Only target-touching edges", value=False, key=f"{key_prefix}_target_only"
            )
            show_edge_labels = st.checkbox(
                "Show edge labels if <=150 edges", value=False, key=f"{key_prefix}_show_edge_labels"
            )
            max_nodes = st.slider("Max nodes", 10, 1500, 350, key=f"{key_prefix}_max_nodes")
    return label_mode, exec_query, hops, kinds, edge_types, target_only, show_edge_labels, max_nodes


def render_prediction_confidence(
    rep: dict[str, pd.DataFrame], aid: str, occurrence: str | None = None
) -> None:
    """Target node's raw top-3 classifier confidence, distinct from the neighborhood-
    distance evidence shown in the score metrics above."""
    conf = df_for_anomaly_occurrence(rep["prediction_confidence.csv"], aid, occurrence)
    if conf.empty:
        return
    conf = conf.sort_values("rank")
    has_true_label = "true_label" in conf.columns
    st.markdown("**Target's raw classifier confidence**")
    if has_true_label:
        st.caption(f"True label: **{conf.iloc[0]['true_label']}**")
    if "true_label_known_in_training" in conf.columns and not bool(
        conf["true_label_known_in_training"].iloc[0]
    ):
        st.warning(
            "This anomaly's true label was never seen during training — the classifier had "
            "no way to predict it correctly, so read its confidence here as evidence about a "
            "genuinely unfamiliar executable, not a classifier mistake."
        )
    cols = st.columns(len(conf))
    for col, (_, row) in zip(cols, conf.iterrows()):
        badges = []
        if bool(row.get("matches_reported_pred_label", False)):
            badges.append("reported")
        if has_true_label and bool(row.get("matches_true_label", False)):
            badges.append("true")
        badge_text = f" ✓ matches {' & '.join(badges)}" if badges else ""
        label = f"#{int(row['rank'])} {row['class_label']}{badge_text}"
        col.metric(label, f"{float(row['probability']):.1%}")
    missing = []
    if not conf["matches_reported_pred_label"].any():
        missing.append(f"reported prediction ({conf.iloc[0]['reported_pred_label']})")
    if has_true_label and not conf["matches_true_label"].any():
        missing.append(f"true label ({conf.iloc[0]['true_label']})")
    if missing:
        st.caption(
            f"Not in the raw top-3: {' and '.join(missing)}. Cluster-based correction and/or "
            "the raw model itself disagree with these."
        )


def page_context_view() -> None:
    st.header("Context View")
    st.caption(
        "The analyst's entry point for a reported anomaly: compare it against the closest "
        "training example for the model's predicted label and for the ground-truth true "
        "label, side by side, to help decide whether this is a false positive or a real "
        "anomaly."
    )
    report_dir = choose_report_dir("context")
    rep = load_report(report_dir)
    anomalies = available_anomalies(rep)
    if not anomalies:
        st.error("No anomalies found.")
        return
    cfg = read_json_if_exists(Path(report_dir) / "run_config.json")
    dataset, project_root, vocab, hop = visual_settings(
        "context", str(cfg.get("dataset", "atlasv2_edr"))
    )
    aid = st.selectbox("Anomaly", anomalies, key="context_aid")
    occurrences = available_occurrences(rep, aid)
    occurrence: str | None = occurrences[0] if occurrences else None
    if len(occurrences) > 1:
        st.info(
            f"This anomaly_id was detected in {len(occurrences)} overlapping time windows, "
            "each analyzed separately (same process, different sliding-window snapshots) — "
            "pick which occurrence to view below."
        )
        occurrence = st.selectbox(
            "Occurrence (time window)",
            occurrences,
            format_func=format_time_window,
            key="context_occurrence",
        )
    pair = df_for_anomaly_occurrence(rep["pairwise_similarities.csv"], aid, occurrence)
    views = ["ALL_VIEWS"] + available_views(pair)
    rank_view = st.selectbox(
        "Closest by view",
        views,
        index=views.index("pyg_final_mean") if "pyg_final_mean" in views else 0,
        key="context_rank_view",
    )
    target = target_row(rep, aid, occurrence)
    if target is None:
        st.error("No target row found for this anomaly/occurrence.")
        return
    pred_ranked = ranked_neighbors(rep, aid, rank_view, ["pred_sample"], occurrence)
    true_ranked = ranked_neighbors(rep, aid, rank_view, ["true_sample"], occurrence)
    p1, p2 = st.columns(2)
    with p1:
        if pred_ranked.empty:
            st.info("No prediction-label training samples available for this anomaly.")
            pred = None
        else:
            pred = pick_ranked_sample(pred_ranked, "Prediction sample", "context_pred_pick")
    with p2:
        if true_ranked.empty:
            st.info(
                "No true-label training samples available — most likely because this "
                "true class was never seen during training."
            )
            true = None
        else:
            true = pick_ranked_sample(true_ranked, "True sample", "context_true_pick")
    if pred is None and true is None:
        st.error(
            "Neither prediction-label nor true-label training samples are available for "
            "this anomaly — nothing to compare the target against."
        )
        return
    metrics: list[tuple[str, str, str | None]] = []
    if pred is not None:
        metrics.append(
            (
                f"Prediction sample score (#{int(pred['rank'])})",
                f"{float(pred['rank_score']):.4f}",
                str(pred.get("label", "")),
            )
        )
    if true is not None:
        metrics.append(
            (
                f"True sample score (#{int(true['rank'])})",
                f"{float(true['rank_score']):.4f}",
                str(true.get("label", "")),
            )
        )
    if pred is not None and true is not None:
        metrics.append(
            ("Pred - True", f"{float(pred['rank_score']) - float(true['rank_score']):.4f}", None)
        )
    for col, (label, value, delta) in zip(st.columns(len(metrics)), metrics):
        col.metric(label, value, delta)
    render_prediction_confidence(rep, aid, occurrence)
    op_name, op_map, op_json = resolve_op_map_json(vocab, dataset)
    st.caption(f"Report: {report_dir} | Operation map: {op_name} ({len(op_map)} entries)")
    target_batch = load_neighborhood_or_error(
        str(target["process_id"]), str(target["data_path"]), hop, op_json, project_root, "target"
    )
    if target_batch is None:
        return
    target_nodes, target_edges = target_batch

    pred_label = f"prediction sample #{int(pred['rank'])}" if pred is not None else None
    pred_batch = (
        load_neighborhood_or_error(
            str(pred["process_id"]),
            str(pred["data_path"]),
            hop,
            op_json,
            project_root,
            pred_label,  # type: ignore
        )
        if pred is not None
        else None
    )
    true_label = f"true sample #{int(true['rank'])}" if true is not None else None
    true_batch = (
        load_neighborhood_or_error(
            str(true["process_id"]),
            str(true["data_path"]),
            hop,
            op_json,
            project_root,
            true_label,  # type: ignore
        )
        if true is not None
        else None
    )

    label_mode, exec_query, hops, kinds, edge_types, target_only, show_edge_labels, max_nodes = (
        shared_graph_filters(
            "context",
            [target_edges]
            + ([pred_batch[1]] if pred_batch is not None else [])
            + ([true_batch[1]] if true_batch is not None else []),
        )
    )
    tn, te = filter_graph(
        target_nodes, target_edges, hops, kinds, edge_types, target_only, max_nodes, exec_query
    )
    graph_frames = [("TARGET", tn, te)]
    seq_frames = [("TARGET", te)]
    if pred_batch is not None:
        pn, pe = filter_graph(
            *pred_batch, hops, kinds, edge_types, target_only, max_nodes, exec_query
        )
        graph_frames.append((cast(str, pred_label).title(), pn, pe))
        seq_frames.append((cast(str, pred_label).title(), pe))
    if true_batch is not None:
        trn, tre = filter_graph(
            *true_batch, hops, kinds, edge_types, target_only, max_nodes, exec_query
        )
        graph_frames.append((cast(str, true_label).title(), trn, tre))
        seq_frames.append((cast(str, true_label).title(), tre))

    tab_names = ["Graphs", "Sequences", "Tables"]
    if pred_batch is not None:
        tab_names.append("Diff: Prediction")
    if true_batch is not None:
        tab_names.append("Diff: True")
    tabs = st.tabs(tab_names)
    tab_iter = iter(tabs)
    with next(tab_iter):
        render_graph_grid(graph_frames, label_mode, show_edge_labels)
    with next(tab_iter):
        max_seq = st.slider("Max sequence edges", 5, 300, 80, key="context_max_seq")
        render_sequence_tables(seq_frames, max_seq)
    with next(tab_iter):
        render_node_edge_tables(graph_frames)
    if pred_batch is not None:
        with next(tab_iter):
            render_diff_pair(tn, te, pn, pe, f"TARGET vs {pred_label}", "context_pred")  # type: ignore
    if true_batch is not None:
        with next(tab_iter):
            render_diff_pair(tn, te, trn, tre, f"TARGET vs {true_label}", "context_true")  # type: ignore
