"""'Overview' page: aggregate metrics/ablation across all anomalies in a report."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .anomaly_data import (
    available_anomalies,
    closest_overview_stats,
    metric_value,
    pair_score_column,
)
from .report_io import choose_report_dir, load_report


def page_overview() -> None:
    st.header("Overview")
    report_dir = choose_report_dir("overview")
    rep = load_report(report_dir)
    conc, view, pair = (
        rep["conclusion_summary.csv"],
        rep["view_summary.csv"],
        rep["pairwise_similarities.csv"],
    )
    reasons, samples, failed = (
        rep["feature_reasons.csv"],
        rep["sample_index.csv"],
        rep["failed_anomalies.csv"],
    )
    summary = rep["summary_statistics_view.csv"]
    ablation, ablation_importance, ablation_winners = (
        rep["ablation_summary.csv"],
        rep["ablation_component_importance.csv"],
        rep["ablation_winner_summary.csv"],
    )
    learning, frequency = rep["learning_quality.csv"], rep["frequency_prior.csv"]
    prediction_confidence = rep["prediction_confidence.csv"]
    if conc.empty and pair.empty:
        st.error("No valid report found. Create or select a report first.")
        return
    st.caption(f"Analyzing: {report_dir}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Successful anomalies", len(conc) if not conc.empty else len(available_anomalies(rep))
    )
    c2.metric("Failed anomalies", len(failed))
    c3.metric("Samples", len(samples))
    c4.metric("Pairwise rows", len(pair))
    c5.metric("Reason rows", len(reasons))
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("PyG final mean", metric_value(summary, "pyg_final_mean:similarity_mean"))
    s2.metric("Winner prediction", metric_value(summary, "winner:prediction", "0"))
    s3.metric("Winner true_label", metric_value(summary, "winner:true_label", "0"))
    s4.metric("Mean margin", metric_value(summary, "margin_pred_minus_true_mean"))
    if not ablation_importance.empty and "removed_view" in ablation_importance.columns:
        top = ablation_importance.iloc[0]
        st.info(
            f"Most influential removed view: {top.get('removed_view')} (mean |margin shift|={float(top.get('mean_abs_margin_shift', 0.0)):.4f}, decision-change rate={float(top.get('decision_change_rate', 0.0)):.2%})"
        )
    tabs = st.tabs(
        [
            "Conclusions",
            "View summary",
            "Pairwise distributions",
            "Summary statistics",
            "Closest overview",
            "Learning quality",
            "Prediction confidence",
            "Ablation",
            "Failures",
        ]
    )
    with tabs[0]:
        st.dataframe(conc, use_container_width=True, height=350)
    with tabs[1]:
        st.dataframe(view, use_container_width=True, height=350)
        if (
            px is not None
            and not view.empty
            and {"view", "true_score", "pred_score"}.issubset(view.columns)
        ):
            st.plotly_chart(
                px.box(
                    view.melt(
                        id_vars=["view"],
                        value_vars=["true_score", "pred_score"],
                        var_name="side",
                        value_name="score",
                    ),
                    x="view",
                    y="score",
                    color="side",
                    points="all",
                ),
                use_container_width=True,
            )
    with tabs[2]:
        st.dataframe(pair, use_container_width=True, height=350)
        score_col = pair_score_column(pair) if not pair.empty else "cosine_similarity"
        if px is not None and not pair.empty and {"view", "role", score_col}.issubset(pair.columns):
            st.plotly_chart(
                px.box(pair, x="view", y=score_col, color="role", points="all"),
                use_container_width=True,
            )
    with tabs[3]:
        st.dataframe(summary, use_container_width=True, height=450)
    with tabs[4]:
        st.dataframe(closest_overview_stats(rep), use_container_width=True, height=450)
    with tabs[5]:
        st.subheader("Per-anomaly classifier quality and training-label frequency")
        st.caption(
            "Precision/recall/F1 of the true and predicted class, plus how often each "
            "label appeared in training — low support or weak per-class scores are a "
            "sign the classifier had little to learn from for that label."
        )
        st.dataframe(learning, use_container_width=True, height=350)
        st.subheader("Frequency prior")
        st.dataframe(frequency, use_container_width=True, height=220)
    with tabs[6]:
        st.subheader("Prediction confidence (target node)")
        st.caption(
            "Raw classifier softmax top-3 for each anomaly's own node, **before** "
            "cluster-based misclassification correction. `reported_pred_label` is the "
            "corrected prediction shown everywhere else in this report (Conclusions, "
            "Context View) — the two can legitimately disagree; "
            "`matches_reported_pred_label` flags which raw row, if any, agrees with it. "
            "`true_label`/`matches_true_label` do the same against the ground truth, so "
            "you can tell 'the raw model was actually right, correction overrode it' apart "
            "from 'the raw model was wrong too'."
        )
        if prediction_confidence.empty:
            st.info(
                "No prediction_confidence.csv in this report. Regenerate the report with "
                "the current engine to see this data."
            )
        else:
            occurrence_cols = ["anomaly_id", "anomaly_time_window"]
            agree = prediction_confidence.groupby(occurrence_cols)[
                "matches_reported_pred_label"
            ].any()
            agree_rate = float(agree.mean()) if not agree.empty else float("nan")
            if "matches_true_label" in prediction_confidence.columns:
                true_agree = prediction_confidence.groupby(occurrence_cols)[
                    "matches_true_label"
                ].any()
                true_agree_rate = float(true_agree.mean()) if not true_agree.empty else float("nan")
            else:
                true_agree_rate = float("nan")
            per_occurrence = prediction_confidence.drop_duplicates(occurrence_cols)
            unseen_rate = (
                float((~per_occurrence["true_label_known_in_training"]).mean())
                if "true_label_known_in_training" in per_occurrence.columns
                and not per_occurrence.empty
                else float("nan")
            )
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric(
                "Reported prediction found in raw top-3",
                f"{agree_rate:.1%}" if not pd.isna(agree_rate) else "-",
            )
            pc2.metric(
                "True label found in raw top-3",
                f"{true_agree_rate:.1%}" if not pd.isna(true_agree_rate) else "-",
            )
            pc3.metric(
                "Anomalies with true class unseen in training",
                f"{unseen_rate:.1%}" if not pd.isna(unseen_rate) else "-",
            )
            st.dataframe(prediction_confidence, use_container_width=True, height=350)
    with tabs[7]:
        st.subheader("Leave-one-view-out ablation")
        st.dataframe(ablation_winners, use_container_width=True, height=220)
        st.dataframe(ablation_importance, use_container_width=True, height=260)
        st.dataframe(ablation, use_container_width=True, height=350)
    with tabs[8]:
        st.dataframe(failed, use_container_width=True, height=350)
