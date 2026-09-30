"""The adaptive PyG neighborhood-distance metric and its leave-one-view-out
ablations."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pandas as pd
from torch import Tensor
from torch_geometric.data import Data

from .models import ABLATION_LABELS, ABLATION_VIEWS, BASE_VIEWS, FINAL_VIEW, TOP_DIFFS


def as_cpu_tensor(value: Any) -> Tensor | None:
    return value.detach().cpu() if value is not None and hasattr(value, "detach") else None


def active_columns(row: Tensor) -> tuple[int, ...]:
    row = row.detach().cpu().reshape(-1)
    return tuple(int(i) for i in row.nonzero(as_tuple=True)[0].tolist())


def normalize(counter: Counter[str]) -> dict[str, float]:
    total = float(sum(counter.values()))
    return {k: float(v) / total for k, v in counter.items() if v != 0} if total > 0 else {}


def profile_distance(a: dict[str, float], b: dict[str, float]) -> float:
    """Weighted-Jaccard distance for sparse non-negative profiles.

    distance = 1 - sum(min(a_i, b_i)) / sum(max(a_i, b_i)).
    Identical profiles return 0; fully disjoint profiles return 1.
    """
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    intersection = sum(
        min(max(0.0, float(a.get(k, 0.0))), max(0.0, float(b.get(k, 0.0)))) for k in keys
    )
    union = sum(max(max(0.0, float(a.get(k, 0.0))), max(0.0, float(b.get(k, 0.0)))) for k in keys)
    return 0.0 if union <= 0.0 else max(0.0, min(1.0, 1.0 - intersection / union))


def similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance))


def log_ratio_distance(a: float, b: float) -> float:
    return 1.0 - math.exp(-abs(math.log1p(max(0.0, a)) - math.log1p(max(0.0, b))))


def top_diffs(a: dict[str, float], b: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for k in set(a) | set(b):
        av, bv = float(a.get(k, 0.0)), float(b.get(k, 0.0))
        if av != bv:
            rows.append(
                {
                    "feature": k,
                    "target_value": av,
                    "sample_value": bv,
                    "sample_minus_target": bv - av,
                }
            )
    rows.sort(key=lambda r: abs(float(r["sample_minus_target"])), reverse=True)
    return rows[:TOP_DIFFS]


def x_active(batch: Data) -> dict[str, float]:
    x = as_cpu_tensor(getattr(batch, "x", None))
    if x is None or x.numel() == 0:
        return {"x:missing": 1.0}
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    counts: Counter[str] = Counter()
    for i in range(x.shape[0]):
        cols = active_columns(x[i])
        if not cols:
            continue
        for col in cols:
            counts[f"active_col:{col}"] += 1
    if x.shape[0] > 0:
        for col in active_columns(x[0]):
            counts[f"target_active_col:{col}"] += 1
    return normalize(counts)


def edge_attr_active(batch: Data) -> dict[str, float]:
    edge_attr = as_cpu_tensor(getattr(batch, "edge_attr", None))
    if edge_attr is None or edge_attr.numel() == 0:
        return {"edge_attr:missing": 1.0}
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.reshape(-1, 1)
    counts: Counter[str] = Counter()
    for i in range(edge_attr.shape[0]):
        cols = active_columns(edge_attr[i])
        if not cols:
            continue
        for col in cols:
            counts[f"active_col:{col}"] += 1
    return normalize(counts)


def size_profile(batch: Data) -> dict[str, float]:
    edge_index = as_cpu_tensor(getattr(batch, "edge_index", None))
    x = as_cpu_tensor(getattr(batch, "x", None))
    edge_attr = as_cpu_tensor(getattr(batch, "edge_attr", None))
    return {
        "num_nodes": float(int(getattr(batch, "num_nodes", 0) or 0)),
        "num_edges": float(
            int(edge_index.shape[1]) if edge_index is not None and edge_index.ndim == 2 else 0
        ),
        "x_active_entries": float(x.nonzero().shape[0]) if x is not None else 0.0,
        "edge_attr_active_entries": float(edge_attr.nonzero().shape[0])
        if edge_attr is not None
        else 0.0,
    }


def cmd_label_counts(batch: Data) -> Counter[str]:
    labels = getattr(batch, "cmd_label_list", []) or []
    return Counter(str(label) for label in labels if str(label))


def cmd_label_profile(batch: Data) -> dict[str, float]:
    counts = cmd_label_counts(batch)
    if not counts:
        return {"cmd_label:missing": 1.0}
    return normalize(Counter({f"cmd_label:{label}": count for label, count in counts.items()}))


def cmd_label_overlap_stats(
    target_counts: dict[str, float], sample_counts: dict[str, float]
) -> dict[str, float]:
    target = Counter({str(k): int(v) for k, v in target_counts.items()})
    sample = Counter({str(k): int(v) for k, v in sample_counts.items()})
    target_keys = set(target)
    sample_keys = set(sample)
    shared_keys = target_keys & sample_keys
    union_keys = target_keys | sample_keys
    shared_count = sum(min(target[k], sample[k]) for k in shared_keys)
    return {
        "cmd_label_shared_count": float(shared_count),
        "cmd_label_target_total": float(sum(target.values())),
        "cmd_label_sample_total": float(sum(sample.values())),
        "cmd_label_shared_distinct": float(len(shared_keys)),
        "cmd_label_union_distinct": float(len(union_keys)),
        "cmd_label_jaccard": float(len(shared_keys) / len(union_keys)) if union_keys else 1.0,
    }


def signature(batch: Data) -> dict[str, dict[str, float]]:
    return {
        "pyg_x_active": x_active(batch),
        "pyg_edge_attr_active": edge_attr_active(batch),
        "pyg_size_profile": size_profile(batch),
        "pyg_cmd_label_profile": cmd_label_profile(batch),
        "_cmd_label_counts": dict(cmd_label_counts(batch)),
    }


def compare_size(a: dict[str, float], b: dict[str, float]) -> tuple[float, list[dict[str, Any]]]:
    gaps = {k: log_ratio_distance(a.get(k, 0.0), b.get(k, 0.0)) for k in set(a) | set(b)}
    dist = min(1.0, sum(gaps.values()) / max(1, len(gaps)))
    diffs = [
        {"feature": k, "target_value": a.get(k, 0.0), "sample_value": b.get(k, 0.0), "distance": v}
        for k, v in sorted(gaps.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return dist, diffs[:TOP_DIFFS]


def compare_signatures(
    target: dict[str, dict[str, float]], sample: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    """Compare two PyG neighborhood signatures, including leave-one-view-out ablations.

    The four base views are computed exactly as in the original metric. The full
    metric is their arithmetic mean. Each ablation view recomputes the same final
    metric after removing one base view:

    - pyg_final_minus_x: removes pyg_x_active       (d_PyG^{-x})
    - pyg_final_minus_edge_attr: removes edge attrs (d_PyG^{-e})
    - pyg_final_minus_size: removes size profile    (d_PyG^{-z})
    - pyg_final_minus_cmd_label: removes labels     (d_PyG^{-c})
    """
    rows: list[dict[str, Any]] = []
    overlap = cmd_label_overlap_stats(
        target.get("_cmd_label_counts", {}), sample.get("_cmd_label_counts", {})
    )

    base: dict[str, dict[str, Any]] = {}
    for view in ["pyg_x_active", "pyg_edge_attr_active", "pyg_cmd_label_profile"]:
        dist = profile_distance(target.get(view, {}), sample.get(view, {}))
        base[view] = {
            "view": view,
            "distance": dist,
            "similarity_score": similarity(dist),
            "top_differences": top_diffs(target.get(view, {}), sample.get(view, {})),
            "metric_kind": "base_view",
            "removed_view": "",
            "ablation_label": "",
            **overlap,
        }

    size_dist, size_diffs = compare_size(
        target.get("pyg_size_profile", {}), sample.get("pyg_size_profile", {})
    )
    base["pyg_size_profile"] = {
        "view": "pyg_size_profile",
        "distance": size_dist,
        "similarity_score": similarity(size_dist),
        "top_differences": size_diffs,
        "metric_kind": "base_view",
        "removed_view": "",
        "ablation_label": "",
        **overlap,
    }

    # Emit the base views first for backwards-compatible visual diagnostics.
    rows.extend(base[v] for v in BASE_VIEWS)

    final_dist = sum(float(base[v]["distance"]) for v in BASE_VIEWS) / len(BASE_VIEWS)
    rows.append(
        {
            "view": FINAL_VIEW,
            "distance": final_dist,
            "similarity_score": similarity(final_dist),
            "top_differences": [],
            "metric_kind": "full_metric",
            "removed_view": "",
            "ablation_label": "",
            **overlap,
        }
    )

    # Leave-one-view-out ablations. These keep the same pairwise sample and only
    # change the final score definition, which makes margin shifts attributable to
    # the removed component.
    for ablation_view, removed_view in ABLATION_VIEWS.items():
        kept = [v for v in BASE_VIEWS if v != removed_view]
        ablated_dist = sum(float(base[v]["distance"]) for v in kept) / len(kept)
        rows.append(
            {
                "view": ablation_view,
                "distance": ablated_dist,
                "similarity_score": similarity(ablated_dist),
                "top_differences": [],
                "metric_kind": "ablation_metric",
                "removed_view": removed_view,
                "ablation_label": ABLATION_LABELS.get(ablation_view, ablation_view),
                **overlap,
            }
        )
    return rows


def nearest_score(df: pd.DataFrame, role: str, view: str) -> float:
    sub = df[(df["role"].astype(str) == role) & (df["view"].astype(str) == view)]
    scores = pd.to_numeric(sub["similarity_score"], errors="coerce").dropna()
    return float(scores.max()) if not scores.empty else float("nan")
