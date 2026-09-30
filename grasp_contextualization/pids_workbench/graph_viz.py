"""Rendering a single neighborhood: graph, event sequence, and tables."""

from __future__ import annotations

import textwrap
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def wrap_label(text: Any, width: int = 24, max_lines: int = 4) -> str:
    value = str(text) if text is not None else ""
    lines = textwrap.wrap(value, width=width)
    shown = lines[:max_lines]
    if len(lines) > max_lines and shown:
        shown[-1] += "..."
    return "<br>".join(shown) if shown else ""


def filter_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    hops: list[int],
    kinds: list[str],
    edge_types: list[str],
    target_only: bool,
    max_nodes: int,
    exec_query: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = nodes[nodes["hop"].isin(hops) & nodes["kind"].isin(kinds)].copy()
    if exec_query:
        mask = (
            n["display_exec_full"].astype(str).str.contains(exec_query, case=False, na=False)
            | n["node_index_id"].astype(str).str.contains(exec_query, case=False, na=False)
            | n["is_target"]
        )
        n = n[mask]
    n = n.head(max_nodes)
    visible = set(n["local_idx"].astype(int))
    e = edges[edges["src"].isin(visible) & edges["dst"].isin(visible)].copy()
    if target_only:
        e = e[e["touches_target"]]
    if edge_types:
        e = e[e["edge_type"].astype(str).isin(edge_types)]
    return n, e


def draw_graph(
    nodes: pd.DataFrame, edges: pd.DataFrame, title: str, label_mode: str, show_edge_labels: bool
) -> None:
    """Improved single-neighborhood graph view."""
    if nx is None or go is None:
        st.subheader(title)
        st.dataframe(nodes, use_container_width=True)
        st.dataframe(edges, use_container_width=True)
        return
    graph = nx.DiGraph()
    colors = {
        "target": "#d62728",
        "subject": "#1f77b4",
        "file": "#2ca02c",
        "netflow": "#ff7f0e",
        "memory": "#9467bd",
        "other": "#777777",
    }
    for _, row in nodes.iterrows():
        idx = int(row["local_idx"])
        kind = str(row.get("kind", "other"))
        if label_mode == "full executable wrapped":
            label = wrap_label(row.get("display_exec_full", ""), 24, 4)
        elif label_mode == "kind + executable":
            label = f"{kind}<br>{wrap_label(row.get('display_exec_full', ''), 24, 3)}"
        elif label_mode == "node id":
            label = wrap_label(row.get("node_index_id", ""), 24, 2)
        else:
            label = str(idx)
        hover = "<br>".join([f"{k}: {v}" for k, v in row.to_dict().items()])
        graph.add_node(str(idx), label=label, kind=kind, hover=hover)
    for _, row in edges.iterrows():
        src, dst = str(int(row["src"])), str(int(row["dst"]))
        if src in graph and dst in graph:
            graph.add_edge(
                src,
                dst,
                label=str(row.get("edge_type", "")),
                hover="<br>".join([f"{k}: {v}" for k, v in row.to_dict().items()]),
            )
    if graph.number_of_nodes() == 0:
        st.info(f"{title}: no graph after filters")
        return
    try:
        pos = (
            nx.kamada_kawai_layout(graph)
            if graph.number_of_nodes() <= 120
            else nx.spring_layout(graph, seed=7, k=1.15, iterations=80)
        )
    except Exception:  # noqa: BLE001
        pos = nx.spring_layout(graph, seed=7, k=1.0, iterations=80)
    traces: list[Any] = []
    ex, ey = [], []
    for src, dst in graph.edges():
        ex += [pos[src][0], pos[dst][0], None]
        ey += [pos[src][1], pos[dst][1], None]
    if ex:
        traces.append(
            go.Scatter(
                x=ex,
                y=ey,
                mode="lines",
                line={"color": "rgba(90,90,90,0.38)", "width": 1.4},
                hoverinfo="none",
                name="edges",
                showlegend=False,
            )
        )
    mx, my, mh = [], [], []
    for src, dst, data in graph.edges(data=True):
        mx.append((pos[src][0] + pos[dst][0]) / 2)
        my.append((pos[src][1] + pos[dst][1]) / 2)
        mh.append(data.get("hover", data.get("label", "")))
    if mx:
        traces.append(
            go.Scatter(
                x=mx,
                y=my,
                mode="markers",
                marker={"size": 8, "color": "rgba(0,0,0,0)"},
                hovertext=mh,
                hoverinfo="text",
                name="edge details",
                showlegend=False,
            )
        )
    for kind in ["target", "subject", "file", "netflow", "memory", "other"]:
        xs, ys, labels, hover = [], [], [], []
        for node, data in graph.nodes(data=True):
            if str(data.get("kind", "other")) != kind:
                continue
            xs.append(pos[node][0])
            ys.append(pos[node][1])
            labels.append(data.get("label", node))
            hover.append(data.get("hover", node))
        if xs:
            size = 34 if kind == "target" else 24
            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    hovertext=hover,
                    hoverinfo="text",
                    marker={
                        "size": size,
                        "color": colors.get(kind, "#777777"),
                        "opacity": 0.96,
                        "line": {"color": "white", "width": 1.6},
                    },
                    name=kind,
                    showlegend=True,
                )
            )
    if show_edge_labels and len(edges) <= 150:
        lx, ly, lt = [], [], []
        for src, dst, data in graph.edges(data=True):
            label = str(data.get("label", ""))[:32]
            if not label:
                continue
            lx.append((pos[src][0] + pos[dst][0]) / 2)
            ly.append((pos[src][1] + pos[dst][1]) / 2)
            lt.append(label)
        if lx:
            traces.append(
                go.Scatter(
                    x=lx,
                    y=ly,
                    mode="text",
                    text=lt,
                    textfont={"size": 9, "color": "#444444"},
                    hoverinfo="none",
                    name="edge labels",
                    showlegend=False,
                )
            )
    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        height=700,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "right", "x": 1},
        xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
        yaxis={"visible": False},
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin={"l": 8, "r": 8, "t": 70, "b": 8},
    )
    st.plotly_chart(fig, use_container_width=True)


def sequence_table(edges: pd.DataFrame, max_edges: int) -> pd.DataFrame:
    cols = ["edge_id", "src_exec", "edge_type", "dst_exec", "touches_target", "edge_signature"]
    return (
        edges.sort_values("edge_id")[[c for c in cols if c in edges.columns]].head(max_edges)
        if not edges.empty
        else pd.DataFrame(columns=cols)
    )


def render_sequence_tables(named_edges: list[tuple[str, pd.DataFrame]], max_edges: int) -> None:
    """Lay out N event sequences as side-by-side columns."""
    cols = st.columns(len(named_edges))
    for col, (title, edges) in zip(cols, named_edges):
        with col:
            st.caption(title)
            st.dataframe(sequence_table(edges, max_edges), use_container_width=True, height=500)


def render_graph_grid(
    named_frames: list[tuple[str, pd.DataFrame, pd.DataFrame]],
    label_mode: str,
    show_edge_labels: bool,
) -> None:
    """Lay out N neighborhoods as side-by-side draw_graph columns."""
    cols = st.columns(len(named_frames))
    for col, (title, nodes, edges) in zip(cols, named_frames):
        with col:
            draw_graph(nodes, edges, title, label_mode, show_edge_labels)


def render_node_edge_tables(named_frames: list[tuple[str, pd.DataFrame, pd.DataFrame]]) -> None:
    """Lay out N neighborhoods as side-by-side node/edge dataframe columns."""
    cols = st.columns(len(named_frames))
    for col, (title, nodes, edges) in zip(cols, named_frames):
        with col:
            st.caption(title)
            st.dataframe(nodes, use_container_width=True, height=280)
            st.dataframe(edges, use_container_width=True, height=280)
