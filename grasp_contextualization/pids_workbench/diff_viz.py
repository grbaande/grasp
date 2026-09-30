"""Rendering a target-vs-sample diff: graph, node diff, and edge diff."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .graph_viz import wrap_label


def compare_value_node(row: pd.Series, mode: str) -> str:
    if mode == "executable":
        return str(row.get("display_exec_full", ""))
    if mode == "kind":
        return str(row.get("kind", ""))
    if mode == "kind+executable":
        return f"{row.get('kind', '')}:{row.get('display_exec_full', '')}"
    return str(row.get("node_index_id", ""))


def compare_value_edge(row: pd.Series, mode: str) -> str:
    if mode == "edge_type":
        return str(row.get("edge_type", ""))
    if mode == "kind_signature":
        return str(row.get("edge_signature", ""))
    return str(row.get("exec_edge_signature", ""))


def diff_table(
    target_values: Iterable[Any], sample_values: Iterable[Any], key: str
) -> pd.DataFrame:
    tc, sc = Counter(map(str, target_values)), Counter(map(str, sample_values))
    rows = []
    for val in sorted(set(tc) | set(sc)):
        t, s = tc.get(val, 0), sc.get(val, 0)
        if t == s:
            status = "shared_equal_count"
        elif t > 0 and s == 0:
            status = "target_only"
        elif s > 0 and t == 0:
            status = "sample_only"
        else:
            status = "shared_count_diff"
        rows.append(
            {
                key: val,
                "target_count": t,
                "sample_count": s,
                "diff_sample_minus_target": s - t,
                "abs_diff": abs(s - t),
                "status": status,
            }
        )
    return (
        pd.DataFrame(rows).sort_values(["abs_diff", key], ascending=[False, True])
        if rows
        else pd.DataFrame()
    )


def status_color(status: str) -> str:
    return {
        "target_anchor": "#d62728",
        "sample_root": "#9467bd",
        "target_only": "#d62728",
        "sample_only": "#2ca02c",
        "shared": "#7f7f7f",
        "shared_equal_count": "#7f7f7f",
        "shared_count_diff": "#9467bd",
    }.get(status, "#777777")


def draw_diff_graph(
    tn: pd.DataFrame,
    te: pd.DataFrame,
    sn: pd.DataFrame,
    se: pd.DataFrame,
    title: str,
    diff_view: str,
    node_mode: str,
    edge_mode: str,
    show_target_only: bool = True,
    show_sample_only: bool = True,
    show_shared: bool = True,
    target_touching_only: bool = True,
    layout_mode: str = "side-by-side exact nodes",
) -> None:
    """Render target/sample diff graph without visible standalone T/S labels."""
    if nx is None or go is None:
        st.warning("networkx/plotly unavailable; showing tables instead.")
        return
    if diff_view == "nodes":
        st.info(
            "Node-only diff is shown as a table below. Graph diff focuses on edges so that neighborhood structure stays correct."
        )
        return
    graph = nx.MultiDiGraph()

    def target_ids(nodes: pd.DataFrame) -> set[int]:
        if nodes.empty or "local_idx" not in nodes.columns:
            return set()
        if "is_target" in nodes.columns:
            rows = nodes[nodes["is_target"].astype(bool)]
            if not rows.empty:
                return set(rows["local_idx"].astype(int).tolist())
        return {int(nodes.iloc[0]["local_idx"])} if not nodes.empty else set()

    target_local_ids = target_ids(tn)
    sample_root_ids = target_ids(sn)

    def label_value(row: pd.Series) -> str:
        if node_mode == "executable":
            return str(row.get("display_exec_full", ""))
        if node_mode == "kind":
            return str(row.get("kind", ""))
        if node_mode == "kind+executable":
            return f"{row.get('kind', '')}: {row.get('display_exec_full', '')}"
        return str(row.get("node_index_id", row.get("local_idx", "")))

    def node_label_from_row(row: pd.Series, side: str) -> str:
        local_idx = int(row.get("local_idx", -1))
        value = wrap_label(label_value(row), 30, 4)
        if side == "target" and local_idx in target_local_ids:
            return f"TARGET<br>{value}" if value else "TARGET"
        if side == "sample" and local_idx in sample_root_ids:
            return f"SAMPLE ROOT<br>{value}" if value else "SAMPLE ROOT"
        return value or str(local_idx)

    def exact_node_id(row: pd.Series, side: str) -> str:
        return f"{side}:{int(row.get('local_idx', -1))}"

    def identity_node_id(row: pd.Series) -> str:
        return f"I:{compare_value_node(row, node_mode)}"

    def get_node_row(nodes: pd.DataFrame, local_idx: int) -> pd.Series | None:
        if nodes.empty or "local_idx" not in nodes.columns:
            return None
        rows = nodes[nodes["local_idx"].astype(int) == int(local_idx)]
        return None if rows.empty else rows.iloc[0]

    def add_node(row: pd.Series | None, side: str, fallback: str = "unknown") -> str:
        if row is None:
            node_id = f"{side}:missing:{fallback}"
            if not graph.has_node(node_id):
                graph.add_node(node_id, label=wrap_label(fallback, 30, 4), status="missing")
            return node_id
        local_idx = int(row.get("local_idx", -1))
        is_target = side == "target" and local_idx in target_local_ids
        is_sample_root = side == "sample" and local_idx in sample_root_ids
        node_id = (
            identity_node_id(row)
            if layout_mode == "merged identity overlay" and not is_target and not is_sample_root
            else exact_node_id(row, side)
        )
        status = (
            "target_anchor"
            if is_target
            else "sample_root"
            if is_sample_root
            else "target_node"
            if side == "target"
            else "sample_node"
        )
        label = node_label_from_row(row, side)
        hover = "<br>".join([f"{k}: {v}" for k, v in row.to_dict().items()])
        if graph.has_node(node_id):
            old_status = graph.nodes[node_id].get("status", status)
            if old_status in {"target_anchor", "sample_root"}:
                status = old_status
                label = graph.nodes[node_id].get("label", label)
        graph.add_node(node_id, label=label, status=status, hover=hover)
        return node_id

    for tid in target_local_ids:
        add_node(get_node_row(tn, tid), "target", "TARGET")
    for sid in sample_root_ids:
        add_node(get_node_row(sn, sid), "sample", "SAMPLE ROOT")

    use_te, use_se = te.copy(), se.copy()
    if target_touching_only:
        if "touches_target" in use_te.columns:
            use_te = use_te[use_te["touches_target"].astype(bool)]
        if "touches_target" in use_se.columns:
            use_se = use_se[use_se["touches_target"].astype(bool)]

    t_edge_ids = {compare_value_edge(r, edge_mode) for _, r in use_te.iterrows()}
    s_edge_ids = {compare_value_edge(r, edge_mode) for _, r in use_se.iterrows()}
    allowed_status: set[str] = set()
    if show_shared:
        allowed_status.add("shared")
    if show_target_only:
        allowed_status.add("target_only")
    if show_sample_only:
        allowed_status.add("sample_only")

    def edge_status(edge_key: str, side: str) -> str:
        if edge_key in t_edge_ids and edge_key in s_edge_ids:
            return "shared"
        return "target_only" if side == "target" else "sample_only"

    def add_edges(df: pd.DataFrame, nodes: pd.DataFrame, side: str) -> None:
        for _, er in df.iterrows():
            key = compare_value_edge(er, edge_mode)
            status = edge_status(key, side)
            if status not in allowed_status:
                continue
            src_idx, dst_idx = int(er.get("src", -1)), int(er.get("dst", -1))
            src_id = add_node(get_node_row(nodes, src_idx), side, str(er.get("src_exec", src_idx)))
            dst_id = add_node(get_node_row(nodes, dst_idx), side, str(er.get("dst_exec", dst_idx)))
            graph.add_edge(
                src_id,
                dst_id,
                status=status,
                label=str(er.get("edge_type", "")),
                hover="<br>".join([f"{k}: {v}" for k, v in er.to_dict().items()]),
            )

    add_edges(use_te, tn, "target")
    add_edges(use_se, sn, "sample")

    if graph.number_of_edges() == 0:
        st.info(
            "No diff edges for the selected toggles/filters. TARGET and SAMPLE ROOT are still shown if available."
        )
    if graph.number_of_nodes() == 0:
        st.info("No diff graph elements")
        return

    fixed_pos: dict[str, tuple[float, float]] = {}
    for node, data in graph.nodes(data=True):
        if data.get("status") == "target_anchor":
            fixed_pos[node] = (-1.4, 0.0)
        elif data.get("status") == "sample_root":
            fixed_pos[node] = (1.4, 0.0)
    try:
        pos = nx.spring_layout(
            graph,
            seed=11,
            k=1.05,
            iterations=90,
            pos=fixed_pos if fixed_pos else None,
            fixed=list(fixed_pos.keys()) if fixed_pos else None,
        )
    except Exception:  # noqa: BLE001
        pos = nx.spring_layout(graph, seed=11, k=1.0, iterations=90)

    traces: list[Any] = []
    edge_names = {
        "shared": "shared edges",
        "target_only": "target-only edges",
        "sample_only": "sample-only edges",
    }
    for status in ["shared", "target_only", "sample_only"]:
        xs, ys, mx, my, mh = [], [], [], [], []
        for src, dst, data in graph.edges(data=True):
            if data.get("status") == status:
                xs += [pos[src][0], pos[dst][0], None]
                ys += [pos[src][1], pos[dst][1], None]
                mx.append((pos[src][0] + pos[dst][0]) / 2)
                my.append((pos[src][1] + pos[dst][1]) / 2)
                mh.append(data.get("hover", data.get("label", "")))
        if xs:
            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"width": 2.4, "color": status_color(status)},
                    name=edge_names.get(status, status),
                    hoverinfo="none",
                )
            )
            traces.append(
                go.Scatter(
                    x=mx,
                    y=my,
                    mode="markers",
                    marker={"size": 9, "color": "rgba(0,0,0,0)"},
                    hovertext=mh,
                    hoverinfo="text",
                    name=f"{edge_names.get(status, status)} details",
                    showlegend=False,
                )
            )

    node_groups = [
        ("target_anchor", "target root", 40, "#d62728"),
        ("sample_root", "sample root", 36, "#9467bd"),
        ("target_node", "target-side nodes", 24, "#6b6b6b"),
        ("sample_node", "sample-side nodes", 24, "#a0a0a0"),
        ("missing", "missing endpoint", 20, "#bbbbbb"),
    ]
    for status, name, size, color in node_groups:
        xs, ys, labels, hover = [], [], [], []
        for node, data in graph.nodes(data=True):
            if data.get("status") != status:
                continue
            xs.append(pos[node][0])
            ys.append(pos[node][1])
            labels.append(data.get("label", node))
            hover.append(data.get("hover", data.get("label", node)))
        if xs:
            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    hovertext=hover,
                    hoverinfo="text",
                    marker={"size": size, "color": color, "line": {"color": "white", "width": 1.7}},
                    name=name,
                    showlegend=True,
                )
            )

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        height=740,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 8, "r": 8, "t": 80, "b": 8},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_diff_pair(
    tn: pd.DataFrame,
    te: pd.DataFrame,
    sn: pd.DataFrame,
    se: pd.DataFrame,
    title: str,
    key_prefix: str,
) -> None:
    st.subheader(title)
    with st.expander("Diff graph controls", expanded=True):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            diff_view = st.selectbox(
                "Diff graph view", ["both", "edges", "nodes"], key=f"{key_prefix}_diff_view"
            )
            layout_mode = st.selectbox(
                "Graph layout",
                ["side-by-side exact nodes", "merged identity overlay"],
                key=f"{key_prefix}_layout_mode",
            )
        with d2:
            node_mode = st.selectbox(
                "Node identity",
                ["executable", "kind+executable", "kind", "node_id"],
                key=f"{key_prefix}_node_mode",
            )
            edge_mode = st.selectbox(
                "Edge identity",
                ["exec_signature", "kind_signature", "edge_type"],
                key=f"{key_prefix}_edge_mode",
            )
        with d3:
            show_target_only = st.toggle(
                "Show target-only", value=True, key=f"{key_prefix}_show_target_only"
            )
            show_sample_only = st.toggle(
                "Show sample-only", value=True, key=f"{key_prefix}_show_sample_only"
            )
            show_shared = st.toggle("Show shared", value=True, key=f"{key_prefix}_show_shared")
        with d4:
            target_touching_only = st.toggle(
                "Only target/root-touching edges",
                value=True,
                key=f"{key_prefix}_target_touching_only",
            )

    node_diff = diff_table(
        [compare_value_node(r, node_mode) for _, r in tn.iterrows()],
        [compare_value_node(r, node_mode) for _, r in sn.iterrows()],
        "node",
    )
    edge_diff = diff_table(
        [compare_value_edge(r, edge_mode) for _, r in te.iterrows()],
        [compare_value_edge(r, edge_mode) for _, r in se.iterrows()],
        "edge",
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Target-only nodes",
        int((node_diff["status"] == "target_only").sum()) if not node_diff.empty else 0,
    )
    m2.metric(
        "Sample-only nodes",
        int((node_diff["status"] == "sample_only").sum()) if not node_diff.empty else 0,
    )
    m3.metric(
        "Target-only edges",
        int((edge_diff["status"] == "target_only").sum()) if not edge_diff.empty else 0,
    )
    m4.metric(
        "Sample-only edges",
        int((edge_diff["status"] == "sample_only").sum()) if not edge_diff.empty else 0,
    )
    tabs = st.tabs(["Graph diff", "Node diff", "Edge diff"])
    with tabs[0]:
        draw_diff_graph(
            tn,
            te,
            sn,
            se,
            title,
            diff_view,
            node_mode,
            edge_mode,
            show_target_only,
            show_sample_only,
            show_shared,
            target_touching_only,
            layout_mode,
        )
    with tabs[1]:
        st.dataframe(node_diff, use_container_width=True, height=420)
    with tabs[2]:
        st.dataframe(edge_diff, use_container_width=True, height=420)
