"""Per-anomaly/occurrence lookups and rankings over report data."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

PYG_VIEWS = [
    "pyg_final_mean",
    "pyg_final_minus_x",
    "pyg_final_minus_edge_attr",
    "pyg_final_minus_size",
    "pyg_final_minus_cmd_label",
    "pyg_x_active",
    "pyg_edge_attr_active",
    "pyg_size_profile",
    "pyg_cmd_label_profile",
]


def df_for_anomaly(df: pd.DataFrame, anomaly_id: str) -> pd.DataFrame:
    if df.empty or "anomaly_id" not in df.columns:
        return pd.DataFrame()
    return df[df["anomaly_id"].astype(str) == str(anomaly_id)].copy()


def df_for_anomaly_occurrence(
    df: pd.DataFrame, anomaly_id: str, occurrence: str | None
) -> pd.DataFrame:
    """df_for_anomaly, additionally scoped to one anomaly_time_window occurrence.

    The same anomaly_id (process_id) can recur across multiple overlapping
    time windows, each independently analyzed with its own samples/scores.
    Without this, rows from different occurrences would be mixed together.
    """
    sub = df_for_anomaly(df, anomaly_id)
    if occurrence and "anomaly_time_window" in sub.columns:
        sub = sub[sub["anomaly_time_window"].astype(str) == str(occurrence)]
    return sub


def available_occurrences(rep: dict[str, pd.DataFrame], anomaly_id: str) -> list[str]:
    samples = df_for_anomaly(rep["sample_index.csv"], anomaly_id)
    if samples.empty or "anomaly_time_window" not in samples.columns:
        return []
    return sorted(samples["anomaly_time_window"].dropna().astype(str).unique().tolist())


def format_time_window(path: str) -> str:
    """Render a graph-window filename's embedded nanosecond timestamps as UTC times."""
    name = Path(str(path)).stem
    match = re.search(r"(\d{10,})_to_(\d{10,})", name)
    if not match:
        return name

    def to_utc(ns: str) -> str:
        try:
            return datetime.fromtimestamp(int(ns) / 1e9, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        except (ValueError, OverflowError, OSError):
            return ns

    return f"{to_utc(match.group(1))} → {to_utc(match.group(2))}"


def available_anomalies(rep: dict[str, pd.DataFrame]) -> list[str]:
    for name in ["conclusion_summary.csv", "sample_index.csv", "pairwise_similarities.csv"]:
        df = rep.get(name, pd.DataFrame())
        if not df.empty and "anomaly_id" in df.columns:
            vals = df["anomaly_id"].astype(str)
            return (
                vals.tolist()
                if name == "conclusion_summary.csv"
                else sorted(vals.unique().tolist())
            )
    return []


def pair_score_column(pair: pd.DataFrame) -> str:
    if "similarity_score" in pair.columns:
        return "similarity_score"
    if "cosine_similarity" in pair.columns:
        return "cosine_similarity"
    return "rank_score"


def available_views(pair: pd.DataFrame) -> list[str]:
    if pair.empty or "view" not in pair.columns:
        return []
    existing = sorted(pair["view"].dropna().astype(str).unique().tolist())
    preferred = [v for v in PYG_VIEWS if v in existing]
    return preferred + [v for v in existing if v not in preferred]


def ranked_neighbors(
    rep: dict[str, pd.DataFrame],
    anomaly_id: str,
    view: str,
    roles: list[str],
    occurrence: str | None = None,
) -> pd.DataFrame:
    pair = df_for_anomaly_occurrence(
        rep.get("pairwise_similarities.csv", pd.DataFrame()), anomaly_id, occurrence
    )
    if pair.empty:
        return pd.DataFrame()
    score_col = pair_score_column(pair)
    if score_col not in pair.columns:
        return pd.DataFrame()
    pair = pair.copy()
    pair[score_col] = pd.to_numeric(pair[score_col], errors="coerce")
    if view != "ALL_VIEWS" and "view" in pair.columns:
        pair = pair[pair["view"].astype(str) == view]
    if roles and "ALL" not in roles and "role" in pair.columns:
        pair = pair[pair["role"].astype(str).isin(roles)]
    if pair.empty:
        return pd.DataFrame()
    group_cols = [
        c for c in ["sample_name", "role", "label", "process_id", "data_path"] if c in pair.columns
    ]
    ranked = (
        pair.groupby(group_cols, as_index=False)
        .agg({score_col: "first"})
        .rename(columns={score_col: "rank_score"})
    )
    ranked = ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def closest_overview_stats(rep: dict[str, pd.DataFrame]) -> pd.DataFrame:
    pair = rep.get("pairwise_similarities.csv", pd.DataFrame())
    if pair.empty or not {"anomaly_id", "view", "role"}.issubset(pair.columns):
        return pd.DataFrame()
    score_col = pair_score_column(pair)
    if score_col not in pair.columns:
        return pd.DataFrame()
    pair = pair.copy()
    pair[score_col] = pd.to_numeric(pair[score_col], errors="coerce")
    rows = []
    for (aid, view), sub in pair.groupby(["anomaly_id", "view"]):
        true_scores = sub[sub["role"].astype(str) == "true_sample"][score_col].dropna()
        pred_scores = sub[sub["role"].astype(str) == "pred_sample"][score_col].dropna()
        true_top = float(true_scores.max()) if not true_scores.empty else float("nan")
        pred_top = float(pred_scores.max()) if not pred_scores.empty else float("nan")
        if pd.isna(true_top) and pd.isna(pred_top):
            winner = "undecidable"
        elif pd.isna(true_top):
            winner = "prediction"
        elif pd.isna(pred_top):
            winner = "true_label"
        elif abs(pred_top - true_top) < 1e-12:
            winner = "tie"
        else:
            winner = "prediction" if pred_top > true_top else "true_label"
        rows.append(
            {
                "anomaly_id": aid,
                "view": view,
                "top_true_similarity": true_top,
                "top_pred_similarity": pred_top,
                "top_neighbor_winner": winner,
                "top_margin_pred_minus_true": pred_top - true_top
                if not (pd.isna(pred_top) or pd.isna(true_top))
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def metric_value(summary: pd.DataFrame, metric: str, default: str = "-") -> str:
    if summary.empty or "metric" not in summary.columns:
        return default
    row = summary[summary["metric"].astype(str) == metric]
    if row.empty:
        return default
    value = row["value"].iloc[0]
    try:
        return f"{float(value):.4f}"
    except Exception:  # noqa: BLE001
        return str(value)


def target_row(
    rep: dict[str, pd.DataFrame], aid: str, occurrence: str | None = None
) -> pd.Series | None:
    samples = df_for_anomaly_occurrence(rep["sample_index.csv"], aid, occurrence)
    target_rows = (
        samples[samples["role"].astype(str) == "anomaly"]
        if not samples.empty and "role" in samples.columns
        else pd.DataFrame()
    )
    return target_rows.iloc[0] if not target_rows.empty else None


def pick_ranked_sample(ranked: pd.DataFrame, label: str, key: str) -> pd.Series | None:
    """Selectbox over ranked candidates, defaulting to the closest (rank 1)."""
    if ranked.empty:
        return None
    choice = st.selectbox(
        label,
        ranked.index.tolist(),
        format_func=lambda i: (
            f"#{int(ranked.loc[i, 'rank'])} score={float(ranked.loc[i, 'rank_score']):.4f} "  # type: ignore
            f"| {ranked.loc[i, 'label']} | {ranked.loc[i, 'sample_name']}"
        ),
        key=key,
    )
    return ranked.loc[choice]
