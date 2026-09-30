"""Turning per-anomaly analysis into the report's CSV/JSON output."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from grasp.graph import graph_storage

from .metric import compare_signatures, nearest_score, signature
from .models import (
    ABLATION_LABELS,
    ABLATION_VIEWS,
    ALL_VIEWS,
    BASE_VIEWS,
    FINAL_VIEW,
    Anomaly,
    ClassMetric,
    Config,
    ExecWindowStats,
    SampleRef,
)
from .sampling import build_sample_refs, label_count, load_neighbor_batch


def sample_row(ref: SampleRef) -> dict[str, Any]:
    return {
        "anomaly_id": ref.anomaly_id,
        "sample_name": ref.sample_name,
        "role": ref.role,
        "label": ref.label,
        "process_id": ref.process_id,
        "data_path": ref.data_path,
        "anomaly_time_window": ref.anomaly_time_window,
    }


def pairwise_row(anom: Anomaly, ref: SampleRef, comp: dict[str, Any]) -> dict[str, Any]:
    sim = float(comp["similarity_score"])
    return {
        "anomaly_id": anom.process_id,
        "anomaly_time_window": anom.data_path,
        "sample_name": ref.sample_name,
        "role": ref.role,
        "label": ref.label,
        "process_id": ref.process_id,
        "data_path": ref.data_path,
        "view": comp["view"],
        "cosine_similarity": sim,
        "similarity_score": sim,
        "distance": float(comp["distance"]),
        "top_differences_json": json.dumps(comp["top_differences"], ensure_ascii=False),
        "cmd_label_shared_count": comp.get("cmd_label_shared_count", float("nan")),
        "cmd_label_target_total": comp.get("cmd_label_target_total", float("nan")),
        "cmd_label_sample_total": comp.get("cmd_label_sample_total", float("nan")),
        "cmd_label_shared_distinct": comp.get("cmd_label_shared_distinct", float("nan")),
        "cmd_label_union_distinct": comp.get("cmd_label_union_distinct", float("nan")),
        "cmd_label_jaccard": comp.get("cmd_label_jaccard", float("nan")),
        "metric_kind": comp.get("metric_kind", ""),
        "removed_view": comp.get("removed_view", ""),
        "ablation_label": comp.get("ablation_label", ""),
    }


def reason_rows(anom: Anomaly, ref: SampleRef, comp: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "anomaly_id": anom.process_id,
            "anomaly_time_window": anom.data_path,
            "view": comp["view"],
            "category": comp["view"].replace("pyg_", ""),
            "feature": diff.get("feature", ""),
            "target_value": diff.get("target_value", float("nan")),
            "sample_value": diff.get("sample_value", float("nan")),
            "sample_name": ref.sample_name,
            "role": ref.role,
            "label": ref.label,
            "supports": "diagnostic",
            "evidence_strength": abs(
                float(diff.get("sample_minus_target", diff.get("distance", 0.0)))
            ),
        }
        for diff in comp["top_differences"]
    ]


def view_summary_rows(anom: Anomaly, pairwise: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view in ALL_VIEWS:
        true_score = nearest_score(pairwise, "true_sample", view)
        pred_score = nearest_score(pairwise, "pred_sample", view)
        winner = (
            "prediction"
            if (math.isnan(true_score) or (not math.isnan(pred_score) and pred_score >= true_score))
            else "true_label"
        )
        rows.append(
            {
                "anomaly_id": anom.process_id,
                "anomaly_time_window": anom.data_path,
                "view": view,
                "true_score": true_score,
                "pred_score": pred_score,
                "winner": winner,
                "margin_pred_minus_true": pred_score - true_score
                if not (math.isnan(pred_score) or math.isnan(true_score))
                else float("nan"),
                "true_label": anom.true_label,
                "pred_label": anom.pred_label,
            }
        )
    return rows


def conclusion_frame(anom: Anomaly, pairwise: pd.DataFrame) -> pd.DataFrame:
    true_score = nearest_score(pairwise, "true_sample", FINAL_VIEW)
    pred_score = nearest_score(pairwise, "pred_sample", FINAL_VIEW)
    winner = (
        "prediction"
        if (math.isnan(true_score) or (not math.isnan(pred_score) and pred_score >= true_score))
        else "true_label"
    )
    return pd.DataFrame(
        [
            {
                "anomaly_id": anom.process_id,
                "anomaly_time_window": anom.data_path,
                "uuid": anom.uuid,
                "true_label": anom.true_label,
                "pred_label": anom.pred_label,
                "decision_view": FINAL_VIEW,
                "true_score": true_score,
                "pred_score": pred_score,
                "winner": winner,
                "margin_pred_minus_true": pred_score - true_score
                if not (math.isnan(pred_score) or math.isnan(true_score))
                else float("nan"),
            }
        ]
    )


def learning_frame(
    anom: Anomaly,
    gs: graph_storage.GraphStorage,
    stats: defaultdict[str, ExecWindowStats],
    class_metrics: dict[int, ClassMetric],
    global_metrics: dict[str, float],
) -> pd.DataFrame:
    label_to_id = {str(k): int(v) for k, v in gs.train_subject_cmd_to_id.items()}
    empty: ClassMetric = {
        "class_id": -1,
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "support": 0,
    }
    true_id, pred_id = label_to_id.get(anom.true_label), label_to_id.get(anom.pred_label)
    true_metrics = class_metrics.get(true_id, empty) if true_id is not None else empty
    pred_metrics = class_metrics.get(pred_id, empty) if pred_id is not None else empty
    return pd.DataFrame(
        [
            {
                "anomaly_id": anom.process_id,
                "anomaly_time_window": anom.data_path,
                "uuid": anom.uuid,
                "true_label": anom.true_label,
                "pred_label": anom.pred_label,
                "true_class_id": true_id,
                "pred_class_id": pred_id,
                "true_label_known_in_training": true_id is not None,
                "true_precision": true_metrics["precision"],
                "true_recall": true_metrics["recall"],
                "true_f1": true_metrics["f1"],
                "true_support": true_metrics["support"],
                "pred_precision": pred_metrics["precision"],
                "pred_recall": pred_metrics["recall"],
                "pred_f1": pred_metrics["f1"],
                "pred_support": pred_metrics["support"],
                "true_label_train_count": label_count(stats, anom.true_label),
                "pred_label_train_count": label_count(stats, anom.pred_label),
                **{f"global_{k}": v for k, v in global_metrics.items()},
            }
        ]
    )


def frequency_frame(anom: Anomaly, stats: defaultdict[str, ExecWindowStats]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "anomaly_id": anom.process_id,
                "anomaly_time_window": anom.data_path,
                "true_label": anom.true_label,
                "pred_label": anom.pred_label,
                "true_label_train_count": label_count(stats, anom.true_label),
                "pred_label_train_count": label_count(stats, anom.pred_label),
            }
        ]
    )


def prediction_confidence_frame(
    anomalies: list[Anomaly],
    confidence_by_node: dict[tuple[str, str], list[tuple[int, float]]],
    gs: graph_storage.GraphStorage,
) -> pd.DataFrame:
    """Per-anomaly top-k predicted-class confidence for the target node itself.

    This is the classifier's own certainty on the anomaly's node, independent
    of the pyg_final_mean neighborhood-distance evidence computed elsewhere.

    IMPORTANT: these are the RAW softmax probabilities from the classifier
    (test_cls_storage.pt's y_hat_proba), before cluster-based misclassification
    correction. The anomaly's `pred_label` everywhere else in this report is
    the CLUSTER-CORRECTED prediction (see
    grasp/evaluation/evaluation_storage.py, which builds pred_cmd_labels from
    y_hat_cluster_corrected, not the raw y_hat). The two can legitimately
    disagree, so each row flags whether it matches the reported pred_label.
    Each row also carries the ground-truth true_label and whether it matches,
    so an analyst can tell "the raw model actually got it right, cluster
    correction just overrode it" apart from "the raw model was wrong too".
    """
    id_to_label = {int(v): str(k) for k, v in gs.train_subject_cmd_to_id.items()}
    known_true_labels = set(gs.train_subject_cmd_to_id.keys())
    rows: list[dict[str, Any]] = []
    for anom in anomalies:
        key = (anom.process_id, anom.data_path)
        for rank, (class_id, prob) in enumerate(confidence_by_node.get(key, []), start=1):
            class_label = id_to_label.get(class_id, f"UNKNOWN_{class_id}")
            rows.append(
                {
                    "anomaly_id": anom.process_id,
                    "anomaly_time_window": anom.data_path,
                    "rank": rank,
                    "class_id": class_id,
                    "class_label": class_label,
                    "probability": prob,
                    "reported_pred_label": anom.pred_label,
                    "matches_reported_pred_label": class_label == anom.pred_label,
                    "true_label": anom.true_label,
                    "matches_true_label": class_label == anom.true_label,
                    "true_label_known_in_training": anom.true_label in known_true_labels,
                }
            )
    return pd.DataFrame(rows)


def analyze_one(
    anom: Anomaly,
    gs: graph_storage.GraphStorage,
    stats: defaultdict[str, ExecWindowStats],
    class_metrics: dict[int, ClassMetric],
    global_metrics: dict[str, float],
    c: Config,
    rng: random.Random,
) -> dict[str, pd.DataFrame]:
    refs = build_sample_refs(anom, stats, c, rng)
    target_sig = signature(load_neighbor_batch(refs[0], c))
    pair_rows: list[dict[str, Any]] = []
    why_rows: list[dict[str, Any]] = []
    for ref in refs[1:]:
        sample_sig = signature(load_neighbor_batch(ref, c))
        for comp in compare_signatures(target_sig, sample_sig):
            pair_rows.append(pairwise_row(anom, ref, comp))
            why_rows.extend(reason_rows(anom, ref, comp))
    pairwise = pd.DataFrame(pair_rows)
    return {
        "conclusion": conclusion_frame(anom, pairwise),
        "view": pd.DataFrame(view_summary_rows(anom, pairwise)),
        "pairwise": pairwise,
        "reasons": pd.DataFrame(why_rows),
        "samples": pd.DataFrame([sample_row(ref) for ref in refs]),
        "frequency": frequency_frame(anom, stats),
        "learning": learning_frame(anom, gs, stats, class_metrics, global_metrics),
    }


def concat_outputs(parts: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    keys = ["conclusion", "view", "pairwise", "reasons", "samples", "frequency", "learning"]
    return {
        k: pd.concat([p[k] for p in parts if not p[k].empty], ignore_index=True)
        if any(not p[k].empty for p in parts)
        else pd.DataFrame()
        for k in keys
    }


def build_summary(
    outputs: dict[str, pd.DataFrame], failed: list[dict[str, Any]], c: Config
) -> pd.DataFrame:
    rows = [
        {"section": "run", "metric": "dataset", "value": c.dataset},
        {"section": "run", "metric": "run_id", "value": c.run_id},
        {"section": "run", "metric": "hop_mode", "value": c.hop_mode},
        {"section": "run", "metric": "failed_anomalies", "value": len(failed)},
    ]
    pairwise = outputs.get("pairwise", pd.DataFrame())
    if not pairwise.empty:
        rows.append({"section": "run", "metric": "pairwise_rows", "value": len(pairwise)})
        for view, g in pairwise.groupby("view"):
            scores = pd.to_numeric(g["similarity_score"], errors="coerce").dropna()
            dist = pd.to_numeric(g["distance"], errors="coerce").dropna()
            rows.append({"section": "view", "metric": f"{view}:count", "value": len(g)})
            if not scores.empty:
                rows.extend(
                    [
                        {
                            "section": "view",
                            "metric": f"{view}:similarity_mean",
                            "value": float(scores.mean()),
                        },
                        {
                            "section": "view",
                            "metric": f"{view}:similarity_median",
                            "value": float(scores.median()),
                        },
                        {
                            "section": "view",
                            "metric": f"{view}:similarity_min",
                            "value": float(scores.min()),
                        },
                        {
                            "section": "view",
                            "metric": f"{view}:similarity_max",
                            "value": float(scores.max()),
                        },
                    ]  # type: ignore
                )  # type: ignore
            if not dist.empty:
                rows.append(
                    {
                        "section": "view",
                        "metric": f"{view}:distance_mean",
                        "value": float(dist.mean()),
                    }  # type: ignore
                )  # type: ignore
    conclusion = outputs.get("conclusion", pd.DataFrame())
    if not conclusion.empty:
        for k, v in conclusion["winner"].value_counts().items():
            rows.append({"section": "decision", "metric": f"winner:{k}", "value": int(v)})
        margins = pd.to_numeric(conclusion["margin_pred_minus_true"], errors="coerce").dropna()
        if not margins.empty:
            rows.append(
                {
                    "section": "decision",
                    "metric": "margin_pred_minus_true_mean",
                    "value": float(margins.mean()),
                }  # type: ignore
            )  # type: ignore
            rows.append(
                {
                    "section": "decision",
                    "metric": "margin_pred_minus_true_median",
                    "value": float(margins.median()),
                }  # type: ignore
            )  # type: ignore
    return pd.DataFrame(rows)


def build_ablation_summary(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return per-anomaly leave-one-view-out sensitivity rows.

    Each row compares the full pyg_final_mean decision margin with one ablated
    final metric. The larger the absolute margin shift, the more influence the
    removed view had on the final nearest-neighbor decision evidence.
    """
    view = outputs.get("view", pd.DataFrame())
    if view.empty or "view" not in view.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for aid, g in view.groupby("anomaly_id"):
        full = g[g["view"].astype(str) == FINAL_VIEW]
        if full.empty:
            continue
        full_row = full.iloc[0]
        full_margin = pd.to_numeric(
            pd.Series([full_row.get("margin_pred_minus_true")]), errors="coerce"
        ).iloc[0]
        for ablation_view, removed_view in ABLATION_VIEWS.items():
            ab = g[g["view"].astype(str) == ablation_view]
            if ab.empty:
                continue
            ab_row = ab.iloc[0]
            ab_margin = pd.to_numeric(
                pd.Series([ab_row.get("margin_pred_minus_true")]), errors="coerce"
            ).iloc[0]
            margin_shift = (
                ab_margin - full_margin
                if not (pd.isna(ab_margin) or pd.isna(full_margin))
                else float("nan")
            )
            rows.append(
                {
                    "anomaly_id": aid,
                    "ablation_view": ablation_view,
                    "ablation_label": ABLATION_LABELS.get(ablation_view, ablation_view),
                    "removed_view": removed_view,
                    "true_label": full_row.get("true_label"),
                    "pred_label": full_row.get("pred_label"),
                    "full_true_score": full_row.get("true_score"),
                    "full_pred_score": full_row.get("pred_score"),
                    "full_winner": full_row.get("winner"),
                    "full_margin_pred_minus_true": full_margin,
                    "ablated_true_score": ab_row.get("true_score"),
                    "ablated_pred_score": ab_row.get("pred_score"),
                    "ablated_winner": ab_row.get("winner"),
                    "ablated_margin_pred_minus_true": ab_margin,
                    "margin_shift_ablated_minus_full": margin_shift,
                    "abs_margin_shift": abs(float(margin_shift))
                    if not pd.isna(margin_shift)
                    else float("nan"),
                    "decision_changed": bool(
                        str(ab_row.get("winner")) != str(full_row.get("winner"))
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_ablation_winner_summary(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count prediction and true-label wins for the full and ablated metrics.

    This gives the publication-style ablation summary requested by reviewers: for
    each full or leave-one-view-out metric, how many anomalies are supported by the
    predicted label versus the true label under the nearest-neighbor decision rule.
    """
    view = outputs.get("view", pd.DataFrame())
    if view.empty or not {"view", "winner"}.issubset(view.columns):
        return pd.DataFrame()

    tracked_views = [FINAL_VIEW] + list(ABLATION_VIEWS.keys())
    rows: list[dict[str, Any]] = []
    for metric_view in tracked_views:
        g = view[view["view"].astype(str) == metric_view].copy()
        if g.empty:
            continue
        winners = g["winner"].fillna("unknown").astype(str)
        margins = pd.to_numeric(
            g.get("margin_pred_minus_true", pd.Series(dtype=float)), errors="coerce"
        )
        pred_wins = int((winners == "prediction").sum())
        true_wins = int((winners == "true_label").sum())
        other_wins = int(len(g) - pred_wins - true_wins)
        rows.append(
            {
                "view": metric_view,
                "metric_kind": "full_metric" if metric_view == FINAL_VIEW else "ablation_metric",
                "removed_view": ABLATION_VIEWS.get(metric_view, ""),
                "ablation_label": ABLATION_LABELS.get(metric_view, ""),
                "n_anomalies": len(g),
                "prediction_wins": pred_wins,
                "true_label_wins": true_wins,
                "other_wins": other_wins,
                "prediction_win_rate": float(pred_wins / len(g)) if len(g) else float("nan"),
                "true_label_win_rate": float(true_wins / len(g)) if len(g) else float("nan"),
                "mean_margin_pred_minus_true": float(margins.mean())
                if not margins.dropna().empty
                else float("nan"),
                "median_margin_pred_minus_true": float(margins.median())
                if not margins.dropna().empty
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_ablation_importance(outputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate ablation sensitivity to rank component importance.

    In addition to margin-shift statistics, this includes winner-count summary
    columns so that each removed component directly reports how often the
    prediction wins and how often the true label wins after ablation.
    """
    ab = build_ablation_summary(outputs)
    if ab.empty:
        return pd.DataFrame()
    tmp = ab.copy()
    tmp["abs_margin_shift"] = pd.to_numeric(tmp["abs_margin_shift"], errors="coerce")
    tmp["margin_shift_ablated_minus_full"] = pd.to_numeric(
        tmp["margin_shift_ablated_minus_full"], errors="coerce"
    )
    tmp["decision_changed"] = tmp["decision_changed"].astype(bool)
    tmp["full_winner"] = tmp["full_winner"].fillna("unknown").astype(str)
    tmp["ablated_winner"] = tmp["ablated_winner"].fillna("unknown").astype(str)

    rows: list[dict[str, Any]] = []
    for removed_view, g in tmp.groupby("removed_view"):
        ablated_prediction_wins = int((g["ablated_winner"] == "prediction").sum())
        ablated_true_wins = int((g["ablated_winner"] == "true_label").sum())
        full_prediction_wins = int((g["full_winner"] == "prediction").sum())
        full_true_wins = int((g["full_winner"] == "true_label").sum())
        pred_to_true = int(
            ((g["full_winner"] == "prediction") & (g["ablated_winner"] == "true_label")).sum()
        )
        true_to_pred = int(
            ((g["full_winner"] == "true_label") & (g["ablated_winner"] == "prediction")).sum()
        )
        n = int(len(g))  # noqa: RUF046
        rows.append(
            {
                "removed_view": removed_view,
                "ablation_view": str(g["ablation_view"].iloc[0]),
                "ablation_label": str(g["ablation_label"].iloc[0]),
                "n_anomalies": n,
                "full_prediction_wins": full_prediction_wins,
                "full_true_label_wins": full_true_wins,
                "ablated_prediction_wins": ablated_prediction_wins,
                "ablated_true_label_wins": ablated_true_wins,
                "ablated_prediction_win_rate": float(ablated_prediction_wins / n)
                if n
                else float("nan"),
                "ablated_true_label_win_rate": float(ablated_true_wins / n) if n else float("nan"),
                "prediction_to_true_label_flips": pred_to_true,
                "true_label_to_prediction_flips": true_to_pred,
                "mean_abs_margin_shift": float(g["abs_margin_shift"].mean()),
                "median_abs_margin_shift": float(g["abs_margin_shift"].median()),
                "max_abs_margin_shift": float(g["abs_margin_shift"].max()),
                "mean_signed_margin_shift": float(g["margin_shift_ablated_minus_full"].mean()),
                "decision_change_count": int(g["decision_changed"].sum()),
                "decision_change_rate": float(g["decision_changed"].mean()),
            }
        )
    out = (
        pd.DataFrame(rows)
        .sort_values(["mean_abs_margin_shift", "decision_change_rate"], ascending=[False, False])
        .reset_index(drop=True)
    )
    out.insert(0, "importance_rank", range(1, len(out) + 1))
    return out


def write_outputs(
    c: Config,
    outputs: dict[str, pd.DataFrame],
    failed: list[dict[str, Any]],
    prediction_confidence: pd.DataFrame,
) -> None:
    c.output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "conclusion_summary.csv": outputs["conclusion"],
        "view_summary.csv": outputs["view"],
        "pairwise_similarities.csv": outputs["pairwise"],
        "feature_reasons.csv": outputs["reasons"],
        "sample_index.csv": outputs["samples"],
        "frequency_prior.csv": outputs["frequency"],
        "learning_quality.csv": outputs["learning"],
        "prediction_confidence.csv": prediction_confidence,
        "summary_statistics_view.csv": build_summary(outputs, failed, c),
        "ablation_summary.csv": build_ablation_summary(outputs),
        "ablation_component_importance.csv": build_ablation_importance(outputs),
        "ablation_winner_summary.csv": build_ablation_winner_summary(outputs),
        "failed_anomalies.csv": pd.DataFrame(failed),
    }
    for name, df in files.items():
        df.to_csv(c.output_dir / name, index=False)
    (c.output_dir / "run_config.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in asdict(c).items()}, indent=2
        ),
        encoding="utf-8",
    )
    (c.output_dir / "pyg_distance_metric.json").write_text(
        json.dumps(
            {
                "name": "single_adaptive_pyg_distance",
                "final_view": FINAL_VIEW,
                "views": ALL_VIEWS,
                "base_views": BASE_VIEWS,
                "ablation_views": ABLATION_VIEWS,
                "ablation_labels": ABLATION_LABELS,
                "distance": "weighted_jaccard_overlap plus log_ratio_size",
                "selection": "nearest sample by pyg_final_mean and leave-one-view-out variants; no top-k/mean/max aggregation controls",
                "ablation": "leave one base view out and measure margin shifts, decision changes, and prediction-vs-true winner counts",
                "neighbor_policy": "full [-1,-1] except dataset names containing e5/optc/carbanak use [10000,20] for two-hop",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
