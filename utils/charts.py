"""
Plotly interactive chart builders for Multilingual Review Intelligence.
Constructs high-contrast, publication-grade interactive charts for:
- Model metric comparisons (Bar, Grouped Bar, Radar charts)
- Sentiment class distributions
- Language performance & Cross-lingual transfer gaps
- Interactive Confusion Matrix heatmaps
- Confidence distribution histograms and boxplots
"""
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from config import SENTIMENT_COLORS, MODEL_CONFIG, SENTIMENT_CLASSES


def create_model_comparison_bar(
    df_metrics: pd.DataFrame,
    metric_col: str = "macro_f1",
    metric_label: str = "Macro-F1",
    title: Optional[str] = None,
) -> Any:
    """Creates a stylized bar chart comparing models on a specific metric."""
    import plotly.graph_objects as go

    if title is None:
        title = f"Model Comparison: {metric_label}"

    # Sort descending
    df_sorted = df_metrics.sort_values(by=metric_col, ascending=False)

    colors = [MODEL_CONFIG.get(m, {}).get("color", "#3B82F6") for m in df_sorted["model"]]
    text_labels = [f"{val * 100:.2f}%" for val in df_sorted[metric_col]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=df_sorted["model"],
                y=df_sorted[metric_col],
                marker_color=colors,
                text=text_labels,
                textposition="outside",
                textfont=dict(size=13, family="Inter", weight="bold"),
                cliponaxis=False,
                hovertemplate="<b>%{x}</b><br>" + f"{metric_label}: " + "%{y:.4f}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=17, family="Inter")),
        xaxis=dict(title="Transformer Model", tickfont=dict(size=12, weight="bold")),
        yaxis=dict(
            title=metric_label,
            tickformat=".0%",
            range=[0, max(1.0, df_sorted[metric_col].max() * 1.15)],
            gridcolor="rgba(255,255,255,0.1)",
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


def create_grouped_metrics_bar(
    df_metrics: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    metric_names: Optional[Dict[str, str]] = None,
    title: str = "Comparative Evaluation Across All Standard Metrics",
) -> Any:
    """Grouped bar chart showing all metrics for each model."""
    import plotly.graph_objects as go

    if metrics is None:
        metrics = ["accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1"]
    if metric_names is None:
        metric_names = {
            "accuracy": "Accuracy",
            "precision_macro": "Macro Precision",
            "recall_macro": "Macro Recall",
            "macro_f1": "Macro-F1",
            "weighted_f1": "Weighted-F1",
        }

    fig = go.Figure()

    for _, row in df_metrics.iterrows():
        model = row["model"]
        color = MODEL_CONFIG.get(model, {}).get("color", "#3B82F6")
        values = [row.get(m, 0.0) for m in metrics]
        text_vals = [f"{v * 100:.1f}%" for v in values]

        fig.add_trace(
            go.Bar(
                name=model,
                x=[metric_names.get(m, m) for m in metrics],
                y=values,
                marker_color=color,
                text=text_vals,
                textposition="outside",
                textfont=dict(size=11, family="Inter"),
                hovertemplate=f"<b>{model}</b><br>%{{x}}: %{{y:.2%}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter")),
        barmode="group",
        yaxis=dict(
            title="Score",
            tickformat=".0%",
            range=[0, 1.05],
            gridcolor="rgba(255,255,255,0.1)",
        ),
        xaxis=dict(tickfont=dict(size=12, weight="bold")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=70, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
    )
    return fig


def create_radar_chart(
    df_metrics: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    metric_names: Optional[Dict[str, str]] = None,
    title: str = "Multidimensional Model Evaluation (Radar)",
) -> Any:
    """Radar (spider) chart comparing models across 5 metrics."""
    import plotly.graph_objects as go

    if metrics is None:
        metrics = ["accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1"]
    if metric_names is None:
        metric_names = {
            "accuracy": "Accuracy",
            "precision_macro": "Macro Precision",
            "recall_macro": "Macro Recall",
            "macro_f1": "Macro-F1",
            "weighted_f1": "Weighted-F1",
        }

    categories = [metric_names.get(m, m) for m in metrics]
    # Close the radar polygon
    categories_closed = categories + [categories[0]]

    fig = go.Figure()

    for _, row in df_metrics.iterrows():
        model = row["model"]
        color = MODEL_CONFIG.get(model, {}).get("color", "#3B82F6")
        values = [row.get(m, 0.0) for m in metrics]
        values_closed = values + [values[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                name=model,
                line=dict(color=color, width=2.5),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.5, 0.85],
                tickformat=".0%",
                tickfont=dict(size=10),
                gridcolor="rgba(255,255,255,0.15)",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, family="Inter", weight="bold"),
                gridcolor="rgba(255,255,255,0.15)",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        title=dict(text=title, font=dict(size=16, family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=60, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        height=420,
    )
    return fig


def create_sentiment_distribution_chart(
    df_preds: pd.DataFrame,
    title: str = "Sentiment Label Distribution (Ground Truth vs Model Predictions)",
) -> Any:
    """Grouped bar chart showing True vs Predicted distribution across sentiment classes."""
    import plotly.graph_objects as go

    true_counts = df_preds["true_label"].value_counts().reindex(SENTIMENT_CLASSES, fill_value=0)
    pred_counts = df_preds["predicted_label"].value_counts().reindex(SENTIMENT_CLASSES, fill_value=0)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Ground Truth",
            x=[c.capitalize() for c in SENTIMENT_CLASSES],
            y=true_counts.values,
            marker_color="#94A3B8",
            text=true_counts.values,
            textposition="outside",
            hovertemplate="<b>Ground Truth %{x}</b>: %{y:,} reviews<extra></extra>",
        )
    )

    pred_colors = [SENTIMENT_COLORS.get(c, "#3B82F6") for c in SENTIMENT_CLASSES]
    fig.add_trace(
        go.Bar(
            name="Predicted",
            x=[c.capitalize() for c in SENTIMENT_CLASSES],
            y=pred_counts.values,
            marker_color=pred_colors,
            text=pred_counts.values,
            textposition="outside",
            hovertemplate="<b>Predicted %{x}</b>: %{y:,} reviews<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter")),
        barmode="group",
        xaxis=dict(title="Sentiment Class", tickfont=dict(size=12, weight="bold")),
        yaxis=dict(title="Number of Reviews", gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


def create_language_comparison_chart(
    df_lang: pd.DataFrame,
    title: str = "English vs Spanish Performance (Macro-F1)",
) -> Any:
    """Bar chart comparing English and Spanish Macro-F1 across models in bilingual experiment."""
    import plotly.graph_objects as go

    bilingual_lang = df_lang[df_lang["experiment"] == "bilingual"]

    fig = go.Figure()

    for lang, lang_name, color in [("en", "English (en)", "#3B82F6"), ("es", "Spanish (es)", "#F97316")]:
        sub = bilingual_lang[bilingual_lang["language"] == lang]
        text_vals = [f"{v * 100:.2f}%" for v in sub["macro_f1"]]

        fig.add_trace(
            go.Bar(
                name=lang_name,
                x=sub["model"],
                y=sub["macro_f1"],
                marker_color=color,
                text=text_vals,
                textposition="outside",
                textfont=dict(size=12, family="Inter"),
                hovertemplate=f"<b>%{{x}} - {lang_name}</b><br>Macro-F1: %{{y:.2%}}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter")),
        barmode="group",
        xaxis=dict(title="Transformer Model", tickfont=dict(size=12, weight="bold")),
        yaxis=dict(
            title="Macro-F1",
            tickformat=".0%",
            range=[0, 0.95],
            gridcolor="rgba(255,255,255,0.1)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


def create_cross_lingual_transfer_chart(
    df_overall: pd.DataFrame,
    df_lang: pd.DataFrame,
    title: str = "Cross-Lingual Transfer Gap: Bilingual Spanish vs English-to-Spanish Transfer",
) -> Any:
    """Grouped bar chart showing the performance drop when training on English only."""
    import plotly.graph_objects as go

    # Extract bilingual Spanish Macro-F1 and Transfer Spanish Macro-F1
    bilingual_es = df_lang[(df_lang["experiment"] == "bilingual") & (df_lang["language"] == "es")].set_index("model")
    transfer_es = df_lang[(df_lang["experiment"] == "english_to_spanish") & (df_lang["language"] == "es")].set_index("model")

    models = ["mDeBERTa-v3", "XLM-R", "mBERT"]
    bilingual_vals = [bilingual_es.loc[m, "macro_f1"] for m in models]
    transfer_vals = [transfer_es.loc[m, "macro_f1"] for m in models]
    drops = [(b - t) * 100 for b, t in zip(bilingual_vals, transfer_vals)]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Bilingual Spanish Macro-F1",
            x=models,
            y=bilingual_vals,
            marker_color="#10B981",
            text=[f"{v * 100:.2f}%" for v in bilingual_vals],
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Transfer Spanish Macro-F1 (Zero-Shot)",
            x=models,
            y=transfer_vals,
            marker_color="#EF4444",
            text=[f"{v * 100:.2f}% (Drop: -{d:.2f}pp)" for v, d in zip(transfer_vals, drops)],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter")),
        barmode="group",
        xaxis=dict(title="Transformer Model", tickfont=dict(size=12, weight="bold")),
        yaxis=dict(
            title="Macro-F1",
            tickformat=".0%",
            range=[0, 0.95],
            gridcolor="rgba(255,255,255,0.1)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    return fig


def create_confusion_matrix_heatmap(
    cm_df: pd.DataFrame,
    title: str = "Confusion Matrix",
    normalize: Optional[str] = None,  # None, "true", "all"
) -> Any:
    """Interactive Plotly heatmap for confusion matrix with formatted text annotations."""
    import plotly.graph_objects as go

    z_vals = cm_df.values
    x_labels = [c.capitalize() for c in cm_df.columns]
    y_labels = [r.capitalize() for r in cm_df.index]

    # Calculate row-normalized percentages for annotations
    row_sums = z_vals.sum(axis=1, keepdims=True)
    row_pcts = np.divide(z_vals, row_sums, out=np.zeros_like(z_vals, dtype=float), where=row_sums != 0)

    text_annotations = []
    for i in range(len(y_labels)):
        row_ann = []
        for j in range(len(x_labels)):
            count = z_vals[i, j]
            pct = row_pcts[i, j] * 100
            row_ann.append(f"<b>{count}</b><br>({pct:.1f}%)")
        text_annotations.append(row_ann)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=x_labels,
            y=y_labels,
            text=text_annotations,
            texttemplate="%{text}",
            textfont=dict(size=14, family="Inter"),
            colorscale="Viridis",
            reversescale=False,
            hoverongaps=False,
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Count"),
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter")),
        xaxis=dict(title="<b>Predicted Sentiment</b>", side="bottom"),
        yaxis=dict(title="<b>True Sentiment</b>", autorange="reversed"),
        margin=dict(l=60, r=40, t=60, b=50),
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_confidence_distribution_plot(
    df_preds: pd.DataFrame,
    model: Optional[str] = None,
    title: str = "Prediction Confidence Distribution (Correct vs Misclassified)",
) -> Any:
    """Histogram / density plot comparing model confidence for correct vs incorrect predictions."""
    import plotly.graph_objects as go

    sub = df_preds.copy()
    if model and model != "All":
        sub = sub[sub["model"] == model]

    correct_conf = sub[sub["correct"] == True]["confidence"].dropna()
    incorrect_conf = sub[sub["correct"] == False]["confidence"].dropna()

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=correct_conf,
            name=f"Correct (n={len(correct_conf):,})",
            marker_color="#10B981",
            opacity=0.7,
            xbins=dict(start=0.3, end=1.0, size=0.035),
            hovertemplate="Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Histogram(
            x=incorrect_conf,
            name=f"Misclassified (n={len(incorrect_conf):,})",
            marker_color="#EF4444",
            opacity=0.75,
            xbins=dict(start=0.3, end=1.0, size=0.035),
            hovertemplate="Confidence: %{x:.2f}<br>Count: %{y}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter")),
        barmode="overlay",
        xaxis=dict(title="Prediction Confidence Score", tickformat=".2f", gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Number of Reviews", gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig
