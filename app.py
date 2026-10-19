"""
Multilingual Review Intelligence — Main Streamlit Application
A professional analytics and evaluation dashboard for multilingual transformer models
(mBERT, XLM-RoBERTa, mDeBERTa-v3) on English and Spanish product reviews.
"""
import sys
import os
from pathlib import Path
import importlib.util
import pandas as pd
import numpy as np

# Ensure root directory is in sys.path and register DirectPathFinder for robust import resolution
ROOT_DIR = Path(__file__).resolve().parent

class DirectPathFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname in sys.modules:
            return None
        parts = fullname.split(".")
        base = str(ROOT_DIR)
        candidate_dir = os.path.join(base, *parts)
        init_py = os.path.join(candidate_dir, "__init__.py")
        file_py = candidate_dir + ".py"
        if os.path.isdir(candidate_dir) and os.path.isfile(init_py):
            return importlib.util.spec_from_file_location(fullname, init_py, submodule_search_locations=[candidate_dir])
        elif os.path.isfile(file_py):
            return importlib.util.spec_from_file_location(fullname, file_py)
        return None

sys.meta_path.insert(0, DirectPathFinder)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import streamlit as st
except ImportError:
    print("Streamlit is not installed in this environment. Please run: pip install -r requirements.txt")
    sys.exit(1)

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_ICON,
    DATA_DIR,
    MODELS_DIR,
    SENTIMENT_CLASSES,
    SENTIMENT_COLORS,
    MODEL_CONFIG,
    EXPERIMENT_CONFIG,
    LANGUAGE_NAMES,
    CUSTOM_CSS,
)
from utils.data_loader import (
    DataLoader,
    filter_predictions,
    validate_uploaded_csv,
)
from utils.metrics import (
    compute_classification_metrics,
    format_metric,
)
from utils.charts import (
    create_model_comparison_bar,
    create_grouped_metrics_bar,
    create_radar_chart,
    create_sentiment_distribution_chart,
    create_language_comparison_chart,
    create_cross_lingual_transfer_chart,
    create_confusion_matrix_heatmap,
    create_confidence_distribution_plot,
)
from utils.inference import ModelInferencer
from utils.export_helpers import (
    dataframe_to_csv_bytes,
    generate_experiment_markdown_report,
)

# -----------------------------------------------------------------------------
# App Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Data Loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all_datasets():
    loader = DataLoader(DATA_DIR)
    data = loader.load_all()
    status = loader.get_load_status()
    return data, status


data_dict, load_status = load_all_datasets()
df_overall = data_dict.get("overall_metrics")
df_bilingual = data_dict.get("bilingual_comparison")
df_cross = data_dict.get("cross_lingual_comparison")
df_lang = data_dict.get("language_wise_metrics")
df_char = data_dict.get("model_characteristics")
df_err_sum = data_dict.get("error_summary")
df_preds = data_dict.get("predictions")
df_err = data_dict.get("error_analysis")

inferencer = ModelInferencer(MODELS_DIR)


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.caption("Multilingual Sentiment Analytics & Model Evaluation")
    st.divider()

    nav_selection = st.radio(
        "Navigation Menu",
        [
            "📊 Overview & Summary",
            "⚖️ Model Comparison",
            "🌍 Language & Transfer Analysis",
            "🔍 Review Explorer",
            "⚠️ Error Analysis & Confusion Matrices",
            "📤 CSV Upload & Validation",
            "⚡ Live Sentiment Analyzer",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### 🛠️ System Health")

    # Data files status
    available_files = sum(1 for v in load_status.values() if v["available"])
    total_files = len(load_status)
    if available_files == total_files:
        st.success(f"✓ Experimental Data: {available_files}/{total_files} CSVs loaded")
    else:
        st.warning(f"⚠️ Experimental Data: {available_files}/{total_files} CSVs loaded")

    # Inference engine status
    mdeberta_status = inferencer.get_status("mDeBERTa-v3")
    if mdeberta_status["ready_for_inference"]:
        st.success("✓ Live Inference: Weights Active")
    else:
        st.info("ℹ️ Live Inference: Checkpoint Weights Not Found (Evaluation-Only Mode)")

    st.divider()
    st.markdown(
        "<small style='color: #94A3B8;'>Built for comparative evaluation of multilingual "
        "transformers (mBERT, XLM-R, mDeBERTa-v3) on English & Spanish reviews.</small>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW & SUMMARY
# -----------------------------------------------------------------------------
if nav_selection == "📊 Overview & Summary":
    st.title("📊 Multilingual Review Intelligence: Overview")
    st.markdown(
        "Welcome to the **Multilingual Review Intelligence** platform. This system provides "
        "interactive analytics and comparative diagnostics for three multilingual Transformer encoders "
        "(**mBERT**, **XLM-RoBERTa**, and **mDeBERTa-v3**) fine-tuned for three-class sentiment analysis "
        "(*negative*, *neutral*, *positive*) on the Multilingual Amazon Reviews Corpus (English & Spanish)."
    )

    if df_preds is None or df_overall is None:
        st.error("Error loading required experimental data. Please ensure data files exist in the `data/` directory.")
        st.stop()

    # Dynamic KPI Calculations from actual data
    total_predictions = len(df_preds)
    bilingual_preds = df_preds[df_preds["experiment"] == "bilingual"]
    transfer_preds = df_preds[df_preds["experiment"] == "english_to_spanish"]

    bilingual_metrics = df_bilingual if df_bilingual is not None else df_overall[df_overall["experiment"] == "bilingual"]
    best_bilingual = bilingual_metrics.sort_values(by="macro_f1", ascending=False).iloc[0]

    transfer_metrics = df_cross if df_cross is not None else df_overall[df_overall["experiment"] == "english_to_spanish"]
    best_transfer = transfer_metrics.sort_values(by="macro_f1", ascending=False).iloc[0]

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Predictions</div>
                <div class="kpi-value">{total_predictions:,}</div>
                <div class="kpi-subtext">3,000 Bilingual + 1,500 Transfer</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Top Bilingual Model</div>
                <div class="kpi-value" style="color: #3B82F6;">{best_bilingual['model']}</div>
                <div class="kpi-subtext">Acc: {best_bilingual['accuracy']*100:.1f}% | Macro-F1: <b>{best_bilingual['macro_f1']*100:.2f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Top Zero-Shot Transfer</div>
                <div class="kpi-value" style="color: #10B981;">{best_transfer['model']}</div>
                <div class="kpi-subtext">Acc: {best_transfer['accuracy']*100:.1f}% | Macro-F1: <b>{best_transfer['macro_f1']*100:.2f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        lowest_err_rate = (1.0 - best_bilingual['accuracy']) * 100
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Lowest Error Rate</div>
                <div class="kpi-value" style="color: #F59E0B;">{lowest_err_rate:.1f}%</div>
                <div class="kpi-subtext">{best_bilingual['model']} (220 errors / 1,000)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Layout: Sentiment Distribution & Experimental Splits
    c_left, c_right = st.columns([3, 2])

    with c_left:
        st.subheader("📈 Sentiment Class Distribution")
        st.caption("Distribution of ground truth vs predicted sentiment labels across the dataset.")
        dist_fig = create_sentiment_distribution_chart(df_preds)
        st.plotly_chart(dist_fig, use_container_width=True)

    with c_right:
        st.subheader("📋 Dataset Partition & Configuration")
        st.caption("Controlled splits sampled from Amazon Reviews Multi (seed 42).")

        split_data = [
            {"Split": "Training Set", "English (en)": "1,200", "Spanish (es)": "1,200", "Total": "2,400", "Purpose": "Task Fine-Tuning"},
            {"Split": "Validation Set", "English (en)": "300", "Spanish (es)": "300", "Total": "600", "Purpose": "Checkpoint Selection"},
            {"Split": "Bilingual Test", "English (en)": "500", "Spanish (es)": "500", "Total": "1,000", "Purpose": "Bilingual Benchmarking"},
            {"Split": "Transfer Test", "English (en)": "—", "Spanish (es)": "500", "Total": "500", "Purpose": "Zero-Shot Cross-Lingual"},
        ]
        st.dataframe(pd.DataFrame(split_data), use_container_width=True, hide_index=True)

        st.markdown(
            """
            **Label Construction:**
            - **Negative (Class 0):** 1–2 Star reviews
            - **Neutral (Class 1):** 3 Star reviews *(Class imbalance: ~half the size)*
            - **Positive (Class 2):** 4–5 Star reviews
            """
        )

    # Executive Summary Highlights
    st.subheader("📌 Key Experimental Findings")
    st.markdown(
        """
        1. **mDeBERTa-v3 achieves the best overall performance:** Outperforms XLM-R and mBERT in both accuracy (78.00%) and Macro-F1 (73.27%).
        2. **Neutral reviews are the primary bottleneck:** Due to mixed sentiment signals and star-rating subjectivity, neutral reviews have the highest error rate across all models (neutral recall is 49.0% for mDeBERTa-v3, 34.8% for XLM-R, and 37.8% for mBERT).
        3. **Cross-lingual transfer causes performance degradation:** Training on English only and evaluating on Spanish results in Macro-F1 drops across the board (-6.17 pp for mDeBERTa-v3, -7.83 pp for XLM-R, and -11.75 pp for mBERT), with neutral class boundaries collapsing most severely.
        """
    )

    # Download summary report
    st.markdown("### 📥 Download Experimental Summary")
    report_md = generate_experiment_markdown_report(df_overall, df_lang, df_char)
    st.download_button(
        label="Download Full Metrics Summary (Markdown)",
        data=report_md,
        file_name="multilingual_review_intelligence_summary.md",
        mime="text/markdown",
    )


# -----------------------------------------------------------------------------
# PAGE 2: MODEL COMPARISON
# -----------------------------------------------------------------------------
elif nav_selection == "⚖️ Model Comparison":
    st.title("⚖️ Model Comparison: mBERT vs XLM-R vs mDeBERTa-v3")
    st.markdown(
        "Evaluate the three multilingual Transformer encoders on the shared **1,000-review bilingual test set** "
        "(500 English + 500 Spanish reviews). All models were fine-tuned using identical hyperparameters: "
        "`max_length=128`, `batch_size=8`, `epochs=2`, `lr=2e-5`, `seed=42`."
    )

    if df_bilingual is None:
        st.error("Bilingual comparison metrics file is missing.")
        st.stop()

    # Formatted display table
    display_df = df_bilingual.copy()
    display_df = display_df.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "precision_macro": "Macro Precision",
            "recall_macro": "Macro Recall",
            "macro_f1": "Macro-F1",
            "weighted_f1": "Weighted-F1",
            "test_examples": "N (Test Reviews)",
        }
    )

    for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro-F1", "Weighted-F1"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%")

    st.dataframe(
        display_df[["Model", "Accuracy", "Macro Precision", "Macro Recall", "Macro-F1", "Weighted-F1", "N (Test Reviews)"]],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        selected_metric = st.selectbox(
            "Select Metric to Compare:",
            options=["macro_f1", "accuracy", "weighted_f1", "precision_macro", "recall_macro"],
            format_func=lambda x: {
                "macro_f1": "Macro-F1 (Primary Benchmark)",
                "accuracy": "Accuracy",
                "weighted_f1": "Weighted-F1",
                "precision_macro": "Macro Precision",
                "recall_macro": "Macro Recall",
            }.get(x, x),
        )
        bar_fig = create_model_comparison_bar(
            df_bilingual,
            metric_col=selected_metric,
            metric_label=selected_metric.replace("_", " ").title(),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    with col_chart2:
        radar_fig = create_radar_chart(df_bilingual)
        st.plotly_chart(radar_fig, use_container_width=True)

    # Grouped bar chart comparing all metrics
    st.subheader("📊 Full Multi-Metric Profile")
    grouped_fig = create_grouped_metrics_bar(df_bilingual)
    st.plotly_chart(grouped_fig, use_container_width=True)

    st.markdown("---")

    # Architectural & Pretraining Characteristics
    st.subheader("🧬 Architectural & Pretraining Characteristics")
    st.caption("Key design differences from model cards and publications.")

    if df_char is not None:
        st.dataframe(df_char, use_container_width=True, hide_index=True)

    # Technical deep-dive accordion
    with st.expander("🔍 Senior ML Engineer Analysis: Why mDeBERTa-v3 Outperforms the Rest"):
        st.markdown(
            """
            - **Disentangled Attention:** Unlike standard BERT and RoBERTa, which sum content and positional embeddings into a single vector, DeBERTa represents each word with two distinct vectors (content and relative position) and calculates attention via disentangled matrices. This enables superior sensitivity to word order and syntax in complex reviews.
            - **ELECTRA-Style Pretraining:** mDeBERTa-v3 utilizes a replaced token detection (RTD) generator-discriminator objective rather than standard masked-language modeling (MLM). This trains the discriminator on every token position in the sequence, yielding much higher parameter efficiency and semantic representations.
            - **Vocabulary Scale & Subword Tokenization:** mDeBERTa-v3 features a 250,000-token SentencePiece vocabulary—more than double mBERT's 110K WordPiece vocabulary. This significantly minimizes out-of-vocabulary subword fragmentation in Spanish.
            - **Pretraining Domain Diversity:** Both XLM-R and mDeBERTa-v3 were trained on filtered CommonCrawl (CC-100) data encompassing 2.5 TB of web text, providing exposure to colloquial phrasing, slang, and informal review language that Wikipedia-trained mBERT rarely encounters.
            """
        )


# -----------------------------------------------------------------------------
# PAGE 3: LANGUAGE ANALYSIS & CROSS-LINGUAL TRANSFER
# -----------------------------------------------------------------------------
elif nav_selection == "🌍 Language & Transfer Analysis":
    st.title("🌍 Language Analysis & Cross-Lingual Transfer")
    st.markdown(
        "Compare model behavior across **English (en)** and **Spanish (es)**, and examine "
        "the **zero-shot cross-lingual transfer experiment** (trained on English only, tested on Spanish)."
    )

    if df_lang is None or df_cross is None:
        st.error("Language or cross-lingual metrics data is missing.")
        st.stop()

    tab_bilingual, tab_transfer = st.tabs([
        "🌐 Bilingual Language Breakdown (EN vs ES)",
        "🔄 Zero-Shot Cross-Lingual Transfer (EN -> ES)",
    ])

    with tab_bilingual:
        st.subheader("English vs Spanish Performance in Bilingual Training")
        st.caption("Models trained on 1,200 EN + 1,200 ES reviews and tested on 500 reviews per language.")

        bilingual_lang = df_lang[df_lang["experiment"] == "bilingual"].copy()
        
        # Format table
        disp_bl = bilingual_lang.copy()
        disp_bl["language"] = disp_bl["language"].map(LANGUAGE_NAMES)
        for col in ["accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1"]:
            disp_bl[col] = disp_bl[col].apply(lambda x: f"{x*100:.2f}%")
        disp_bl = disp_bl.rename(columns={
            "model": "Model", "language": "Language", "test_examples": "N",
            "accuracy": "Accuracy", "precision_macro": "Macro Precision",
            "recall_macro": "Macro Recall", "macro_f1": "Macro-F1", "weighted_f1": "Weighted-F1"
        })

        st.dataframe(disp_bl[["Model", "Language", "Accuracy", "Macro-F1", "Weighted-F1", "Macro Precision", "Macro Recall", "N"]], use_container_width=True, hide_index=True)

        lang_fig = create_language_comparison_chart(df_lang)
        st.plotly_chart(lang_fig, use_container_width=True)

        st.info(
            "💡 **Language Parity Observation:** mDeBERTa-v3 achieves the most consistent cross-language performance, "
            "with a small 2.77 pp Macro-F1 gap between English (74.60%) and Spanish (71.83%). XLM-R exhibits slightly higher "
            "Spanish accuracy (76.40%) than English (75.20%), while mBERT suffers the largest language discrepancy (74.20% EN vs 70.60% ES)."
        )

    with tab_transfer:
        st.subheader("Cross-Lingual Transfer: English Fine-Tuning -> Spanish Testing")
        st.markdown(
            """
            <div class="status-alert status-alert-blue">
                <div>
                    <b>Experimental Setup:</b> Models were fine-tuned <b>strictly on 1,200 English reviews</b> (with 300 English validation examples) 
                    and then evaluated on <b>500 Spanish reviews</b> without seeing a single Spanish label during task fine-tuning.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Cross-lingual metrics table
        disp_cross = df_cross.copy()
        disp_cross["Accuracy"] = disp_cross["accuracy"].apply(lambda x: f"{x*100:.2f}%")
        disp_cross["Macro-F1"] = disp_cross["macro_f1"].apply(lambda x: f"{x*100:.2f}%")
        disp_cross["Weighted-F1"] = disp_cross["weighted_f1"].apply(lambda x: f"{x*100:.2f}%")
        disp_cross["Macro Precision"] = disp_cross["precision_macro"].apply(lambda x: f"{x*100:.2f}%")
        disp_cross["Macro Recall"] = disp_cross["recall_macro"].apply(lambda x: f"{x*100:.2f}%")
        disp_cross["N"] = disp_cross["test_examples"]
        disp_cross["Model"] = disp_cross["model"]

        st.dataframe(disp_cross[["Model", "Accuracy", "Macro-F1", "Weighted-F1", "Macro Precision", "Macro Recall", "N"]], use_container_width=True, hide_index=True)

        st.subheader("📉 Transfer Degradation / Drop Analysis")
        st.caption("Quantifying the performance penalty of zero-shot transfer compared to bilingual supervision on Spanish test data.")

        # Compute drops dynamically
        bilingual_es = df_lang[(df_lang["experiment"] == "bilingual") & (df_lang["language"] == "es")].set_index("model")
        transfer_es = df_lang[(df_lang["experiment"] == "english_to_spanish") & (df_lang["language"] == "es")].set_index("model")

        drop_rows = []
        for m in ["mDeBERTa-v3", "XLM-R", "mBERT"]:
            b_acc = bilingual_es.loc[m, "accuracy"]
            t_acc = transfer_es.loc[m, "accuracy"]
            b_f1 = bilingual_es.loc[m, "macro_f1"]
            t_f1 = transfer_es.loc[m, "macro_f1"]

            drop_rows.append({
                "Model": m,
                "Bilingual Spanish Acc": f"{b_acc*100:.2f}%",
                "Transfer Spanish Acc": f"{t_acc*100:.2f}%",
                "Accuracy Drop (pp)": f"{(b_acc - t_acc)*100:.2f} pp",
                "Bilingual Spanish Macro-F1": f"{b_f1*100:.2f}%",
                "Transfer Spanish Macro-F1": f"{t_f1*100:.2f}%",
                "Macro-F1 Drop (pp)": f"{(b_f1 - t_f1)*100:.2f} pp",
            })

        st.dataframe(pd.DataFrame(drop_rows), use_container_width=True, hide_index=True)

        transfer_fig = create_cross_lingual_transfer_chart(df_overall, df_lang)
        st.plotly_chart(transfer_fig, use_container_width=True)

        st.markdown(
            """
            **Transfer Analysis Takeaways:**
            - **mDeBERTa-v3 is the most resilient to cross-lingual transfer:** Maintains 74.40% accuracy and 65.66% Macro-F1, suffering the smallest drop (-6.17 pp Macro-F1).
            - **mBERT suffers severe transfer failure:** Drops by 11.75 pp in Macro-F1 (down to 52.85%), reflecting weaker cross-lingual alignment in Wikipedia pretraining.
            - **Neutral class collapse:** In the zero-shot setting, models fail almost completely on neutral Spanish reviews (mBERT recalls only 8.8% of neutrals, XLM-R recalls 13.7%, and mDeBERTa-v3 recalls 25.5%).
            """
        )


# -----------------------------------------------------------------------------
# PAGE 4: INTERACTIVE REVIEW EXPLORER
# -----------------------------------------------------------------------------
elif nav_selection == "🔍 Review Explorer":
    st.title("🔍 Interactive Review Explorer")
    st.markdown(
        "Search, filter, and inspect prediction records from the experimental dataset. "
        "Filter by model, language, true label, predicted label, correctness, and confidence."
    )

    if df_preds is None:
        st.error("Predictions data is missing.")
        st.stop()

    # Filter controls in expandable container
    with st.expander("🛠️ Filter Controls & Search", expanded=True):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            exp_choice = st.selectbox(
                "Experiment:",
                options=["All", "bilingual", "english_to_spanish"],
                format_func=lambda x: {
                    "All": "All Experiments",
                    "bilingual": "Bilingual (EN + ES)",
                    "english_to_spanish": "Transfer (EN -> ES)",
                }.get(x, x),
            )

        with f_col2:
            model_choice = st.selectbox("Model:", options=["All", "mDeBERTa-v3", "XLM-R", "mBERT"])

        with f_col3:
            lang_choice = st.selectbox(
                "Language:",
                options=["All", "en", "es"],
                format_func=lambda x: {"All": "All Languages", "en": "English", "es": "Spanish"}.get(x, x),
            )

        with f_col4:
            correct_choice = st.selectbox(
                "Prediction Outcome:",
                options=["All", "Correct Only", "Misclassified Only"],
            )

        f_col5, f_col6, f_col7 = st.columns([1, 1, 2])

        with f_col5:
            true_label_choice = st.selectbox(
                "True Sentiment:",
                options=["All"] + SENTIMENT_CLASSES,
                format_func=lambda x: x.capitalize(),
            )

        with f_col6:
            pred_label_choice = st.selectbox(
                "Predicted Sentiment:",
                options=["All"] + SENTIMENT_CLASSES,
                format_func=lambda x: x.capitalize(),
            )

        with f_col7:
            conf_range = st.slider(
                "Confidence Range:",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05,
            )

        search_query = st.text_input("🔎 Search Review Text (Keyword or Phrase):", "")

    # Apply filters dynamically
    filtered_df = filter_predictions(
        df_preds,
        experiment=exp_choice,
        model=model_choice,
        language=lang_choice,
        true_label=true_label_choice,
        predicted_label=pred_label_choice,
        correctness=correct_choice,
        min_confidence=conf_range[0],
        max_confidence=conf_range[1],
        search_query=search_query,
    )

    # Result counts and quick stats
    total_matches = len(filtered_df)
    st.markdown(f"**Found {total_matches:,} matching reviews** (out of {len(df_preds):,} total records)")

    if total_matches == 0:
        st.warning("No reviews matched your selected filter criteria. Try broadening your filters.")
    else:
        # Quick stats on filtered subset
        acc_filtered = filtered_df["correct"].mean() * 100
        avg_conf = filtered_df["confidence"].mean() * 100

        st.caption(f"Subset Accuracy: **{acc_filtered:.1f}%** | Mean Model Confidence: **{avg_conf:.1f}%**")

        # Display table with formatting
        preview_cols = ["model", "language", "experiment", "true_label", "predicted_label", "confidence", "correct", "text"]
        st_df = filtered_df[preview_cols].copy()
        st_df["confidence"] = st_df["confidence"].apply(lambda x: f"{x:.3f}")
        st_df["language"] = st_df["language"].map({"en": "English", "es": "Spanish"})
        st_df["correct"] = st_df["correct"].map({True: "✓ Correct", False: "✗ Error"})

        st.dataframe(
            st_df,
            use_container_width=True,
            height=400,
        )

        # Review deep-dive inspect tool
        st.subheader("📖 Detailed Review Inspector")
        selected_idx = st.number_input(
            "Select review row number to view full text:",
            min_value=0,
            max_value=max(0, total_matches - 1),
            value=0,
            step=1,
        )

        if total_matches > 0:
            target_row = filtered_df.iloc[selected_idx]
            correct_badge = "✓ Correctly Classified" if target_row["correct"] else "✗ Misclassified"
            badge_color = "#10B981" if target_row["correct"] else "#EF4444"

            st.markdown(
                f"""
                <div class="review-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                        <span style="font-weight: 700; color: #F8FAFC;">Model: {target_row['model']} | Language: {LANGUAGE_NAMES.get(target_row['language'], target_row['language'])} ({target_row['experiment']})</span>
                        <span style="font-weight: 700; color: {badge_color};">{correct_badge}</span>
                    </div>
                    <div style="margin-bottom: 0.5rem;">
                        <b>True Sentiment:</b> <span class="badge-{target_row['true_label']}">{target_row['true_label'].upper()}</span> &nbsp;&nbsp;
                        <b>Predicted Sentiment:</b> <span class="badge-{target_row['predicted_label']}">{target_row['predicted_label'].upper()}</span> &nbsp;&nbsp;
                        <b>Confidence:</b> {target_row['confidence']:.4f}
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 0.8rem; border-radius: 6px; font-family: monospace; font-size: 0.9rem; white-space: pre-wrap; color: #E2E8F0;">
{target_row['text']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Download filtered results
        csv_bytes = dataframe_to_csv_bytes(filtered_df)
        st.download_button(
            label="📥 Download Filtered Reviews as CSV",
            data=csv_bytes,
            file_name="filtered_predictions.csv",
            mime="text/csv",
        )


# -----------------------------------------------------------------------------
# PAGE 5: ERROR ANALYSIS & CONFUSION MATRICES
# -----------------------------------------------------------------------------
elif nav_selection == "⚠️ Error Analysis & Confusion Matrices":
    st.title("⚠️ Error Analysis & Confusion Matrices")
    st.markdown(
        "Diagnose model misclassifications, analyze prediction confidence distributions, "
        "and inspect interactive confusion matrices for both the bilingual and cross-lingual experiments."
    )

    if df_preds is None or df_err is None:
        st.error("Prediction and error analysis data are missing.")
        st.stop()

    tab_cm, tab_confidence, tab_qualitative = st.tabs([
        "🔲 Confusion Matrices",
        "📊 Confidence & Calibration",
        "🔬 Qualitative Error Case Studies",
    ])

    with tab_cm:
        st.subheader("Interactive Confusion Matrix Visualizer")
        
        c_exp, c_model = st.columns(2)
        with c_exp:
            cm_exp = st.selectbox(
                "Experiment Setup:",
                options=["bilingual", "english_to_spanish"],
                format_func=lambda x: {
                    "bilingual": "Bilingual Experiment (1,000 Test Reviews)",
                    "english_to_spanish": "Cross-Lingual Transfer (500 Spanish Test Reviews)",
                }.get(x, x),
            )
        with c_model:
            cm_model = st.selectbox("Select Model:", options=["mDeBERTa-v3", "XLM-R", "mBERT"])

        # Dynamically compute confusion matrix from actual predictions
        subset = df_preds[(df_preds["experiment"] == cm_exp) & (df_preds["model"] == cm_model)]
        metrics_dict = compute_classification_metrics(subset["true_label"].values, subset["predicted_label"].values)
        cm_df = metrics_dict["confusion_matrix"]

        cm_fig = create_confusion_matrix_heatmap(
            cm_df,
            title=f"Confusion Matrix: {cm_model} ({cm_exp.replace('_', ' ').title()})",
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        # Per-class breakdown metrics table
        st.markdown(f"#### Per-Class Performance Breakdown: {cm_model}")
        per_class_data = []
        for l, stats in metrics_dict["per_class"].items():
            per_class_data.append({
                "Sentiment Class": l.capitalize(),
                "Support (N)": stats["support"],
                "Precision": f"{stats['precision']*100:.2f}%",
                "Recall": f"{stats['recall']*100:.2f}%",
                "F1-Score": f"{stats['f1']*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(per_class_data), use_container_width=True, hide_index=True)

        st.caption(
            "Note the diagonal dominance on Negative and Positive classes, and the severe dispersal on Neutral. "
            f"Neutral recall for {cm_model} in {cm_exp} is **{metrics_dict['per_class']['neutral']['recall']*100:.2f}%**."
        )

    with tab_confidence:
        st.subheader("Prediction Confidence: Correct vs Misclassified Reviews")
        st.markdown(
            "Models should ideally exhibit high confidence on correct predictions and low confidence "
            "on errors. Overconfident errors (incorrect predictions with confidence > 85%) reveal systematic "
            "semantic blind spots."
        )

        conf_model_filter = st.selectbox(
            "Filter Confidence Plot by Model:",
            options=["All", "mDeBERTa-v3", "XLM-R", "mBERT"],
        )

        conf_fig = create_confidence_distribution_plot(df_preds, model=conf_model_filter)
        st.plotly_chart(conf_fig, use_container_width=True)

        # Overconfident errors table
        st.subheader("🚨 High-Confidence Misclassifications (Confidence > 0.85)")
        st.caption("Reviews where the model was strongly convinced of an incorrect sentiment label.")

        overconfident = df_preds[(df_preds["correct"] == False) & (df_preds["confidence"] >= 0.85)].sort_values(
            by="confidence", ascending=False
        )

        st.markdown(f"**Found {len(overconfident):,} high-confidence error cases.**")
        st.dataframe(
            overconfident[["model", "language", "experiment", "true_label", "predicted_label", "confidence", "text"]],
            use_container_width=True,
            height=300,
        )

    with tab_qualitative:
        st.subheader("Qualitative Error Case Studies from the Project Report")
        st.markdown(
            "Detailed analysis of why reviews were misclassified by the models based on domain "
            "linguistic nuance and weakly-supervised rating labels."
        )

        qualitative_cases = [
            {
                "Experiment & Model": "Bilingual / mBERT",
                "Language & Transition": "Spanish: Neutral ➔ Negative",
                "Review Excerpt": "Envío demasiado lento. El colchón está bien; el envío tardó más de dos semanas.",
                "Error Mechanism": "Mixed Sentiment: Positive product evaluation ('colchón está bien') mixed with strong delivery complaint. Label compresses multi-faceted review into single rating.",
            },
            {
                "Experiment & Model": "Bilingual / mBERT",
                "Language & Transition": "Spanish: Neutral ➔ Positive",
                "Review Excerpt": "Bastante bien. Son tal y como aparece en la foto. Le doy tres estrellas…",
                "Error Mechanism": "Star Rating vs Wording: Lexical phrasing ('bastante bien') signals positive sentiment, while user explicitly gave 3 stars due to minor caveats.",
            },
            {
                "Experiment & Model": "Bilingual / XLM-R",
                "Language & Transition": "English: Neutral ➔ Positive",
                "Review Excerpt": "Three Stars. Attracts small ants… at least in S Florida.",
                "Error Mechanism": "Implicit Negation: Model missed the negative drawback ('attracts small ants') and treated mild language as neutral/positive.",
            },
            {
                "Experiment & Model": "Bilingual / mDeBERTa-v3",
                "Language & Transition": "English: Negative ➔ Positive",
                "Review Excerpt": "Great service. I was impressed by the speed with which it arrived. Thank you.",
                "Error Mechanism": "Ground-Truth Label Disagreement: Text is unequivocally positive ('Great service', 'impressed'), but labeled negative in corpus (likely user misclick on 1-star).",
            },
            {
                "Experiment & Model": "Bilingual / mDeBERTa-v3",
                "Language & Transition": "Spanish: Neutral ➔ Positive",
                "Review Excerpt": "Libro interesante. Interesante y bien ilustrado; el tamaño es adecuado.",
                "Error Mechanism": "Positive Lexical Bias: Enthusiastic adjectives ('interesante', 'bien ilustrado') overwhelm the 3-star neutral threshold.",
            },
            {
                "Experiment & Model": "Transfer / mDeBERTa-v3",
                "Language & Transition": "Spanish: Neutral ➔ Positive",
                "Review Excerpt": "Uso diario soportado. La uso casi todos los días y funciona bastante bien…",
                "Error Mechanism": "Cross-Lingual Boundary Shift: English-only fine-tuning causes decision boundary for neutral to shrink, shunting mild satisfaction into positive.",
            },
        ]

        st.dataframe(pd.DataFrame(qualitative_cases), use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 6: CSV UPLOAD & VALIDATION
# -----------------------------------------------------------------------------
elif nav_selection == "📤 CSV Upload & Validation":
    st.title("📤 CSV Upload, Validation & Diagnostics")
    st.markdown(
        "Upload your own CSV containing review text or external model predictions. "
        "The system validates column schemas, detects the file mode, computes dynamic metrics, "
        "and allows downloading filtered results."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file to upload",
        type=["csv"],
        help="Upload a Prediction CSV (with text, true_label, predicted_label) or Raw Reviews (with text).",
    )

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"✓ Successfully read file: `{uploaded_file.name}` ({len(user_df):,} rows, {len(user_df.columns)} columns)")

            # Validate uploaded CSV
            is_valid, mode, issues, clean_df = validate_uploaded_csv(user_df)

            if not is_valid:
                st.error("❌ Validation Failed. Please fix the following schema issues:")
                for issue in issues:
                    st.write(f"- {issue}")
            else:
                if mode == "predictions":
                    st.markdown(
                        """
                        <div class="status-alert status-alert-green">
                            <div>
                                <b>Mode Detected: Prediction Benchmark Evaluation</b><br>
                                Found columns for review text, true ground-truth labels, and model predictions.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Dynamic metrics computation on uploaded predictions
                    user_metrics = compute_classification_metrics(
                        clean_df["true_label"].values,
                        clean_df["predicted_label"].values,
                    )

                    # Display KPI cards
                    u_col1, u_col2, u_col3, u_col4 = st.columns(4)
                    with u_col1:
                        st.metric("Total Reviews", f"{user_metrics['test_examples']:,}")
                    with u_col2:
                        st.metric("Accuracy", f"{user_metrics['accuracy']*100:.2f}%")
                    with u_col3:
                        st.metric("Macro-F1", f"{user_metrics['macro_f1']*100:.2f}%")
                    with u_col4:
                        st.metric("Weighted-F1", f"{user_metrics['weighted_f1']*100:.2f}%")

                    st.markdown("---")

                    # Confusion Matrix on uploaded predictions
                    st.subheader("Confusion Matrix on Uploaded Predictions")
                    user_cm_fig = create_confusion_matrix_heatmap(
                        user_metrics["confusion_matrix"],
                        title="Uploaded Predictions: Confusion Matrix",
                    )
                    st.plotly_chart(user_cm_fig, use_container_width=True)

                elif mode == "raw_reviews":
                    st.markdown(
                        """
                        <div class="status-alert status-alert-blue">
                            <div>
                                <b>Mode Detected: Raw Review Dataset</b><br>
                                Found review text column. Ready for preview and inspection.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Data preview & filter download
                st.subheader("Dataset Preview")
                st.dataframe(clean_df.head(100), use_container_width=True)

                # Export validated CSV
                export_csv = dataframe_to_csv_bytes(clean_df)
                st.download_button(
                    label="📥 Download Validated & Cleaned CSV",
                    data=export_csv,
                    file_name=f"validated_{uploaded_file.name}",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error parsing uploaded CSV file: {str(e)}")

    else:
        st.info("💡 Please upload a CSV file using the file selector above.")

        with st.expander("ℹ️ Supported CSV Schemas & Format Requirements"):
            st.markdown(
                """
                **1. Prediction Evaluation CSV:**
                - `text` *(string, required)*: The review text.
                - `true_label` *(string, required)*: Ground truth sentiment (`negative`, `neutral`, `positive`).
                - `predicted_label` *(string, required)*: Model predicted sentiment (`negative`, `neutral`, `positive`).
                - *Optional columns:* `model`, `language`, `confidence`, `experiment`.

                **2. Raw Reviews CSV:**
                - `text` *(string, required)*: Review content.
                - *Optional columns:* `language` (e.g. `en`, `es`).
                """
            )


# -----------------------------------------------------------------------------
# PAGE 7: LIVE SENTIMENT ANALYZER
# -----------------------------------------------------------------------------
elif nav_selection == "⚡ Live Sentiment Analyzer":
    st.title("⚡ Live Sentiment Analyzer")
    st.markdown(
        "Interactive interface for entering a product review and executing real-time sentiment inference. "
        "Connected to actual fine-tuned Transformer model weights when provided."
    )

    col_select1, col_select2 = st.columns(2)

    with col_select1:
        selected_model_key = st.selectbox(
            "Select Multilingual Transformer:",
            options=["mDeBERTa-v3", "XLM-R", "mBERT"],
            format_func=lambda x: MODEL_CONFIG.get(x, {}).get("display_name", x),
        )

    with col_select2:
        custom_weights_path = st.text_input(
            "Custom Checkpoint Path (Optional):",
            value="",
            placeholder="e.g. models/mdeberta-v3 or /path/to/checkpoint",
            help="Specify a local directory containing 'pytorch_model.bin' or 'model.safetensors' and 'config.json'.",
        )

    # Check status of weights
    active_path = custom_weights_path.strip() if custom_weights_path.strip() else None
    model_status = inferencer.get_status(selected_model_key, active_path)

    if not model_status["ready_for_inference"]:
        st.markdown(
            f"""
            <div class="status-alert status-alert-amber">
                <div>
                    <b>⚠️ Live Inference Currently Unavailable for {selected_model_key}</b><br>
                    Trained model checkpoint weights were not included in the experimental results archive. 
                    In accordance with strict ML engineering guidelines, <b>this system NEVER fabricates predictions or mock scores</b>.<br><br>
                    <i>Status:</i> {model_status['status_message']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("📖 How to Enable Live Inference with Fine-Tuned Weights"):
            st.markdown(
                f"""
                To connect real fine-tuned weights for **{selected_model_key}**:
                1. Export your trained Hugging Face checkpoint directory containing:
                   - `config.json`
                   - `pytorch_model.bin` or `model.safetensors`
                   - Tokenizer files (`tokenizer.json`, `vocab.txt` or `spm.model`)
                2. Place the files into `{MODELS_DIR}/{inferencer.MODEL_DIR_MAP.get(selected_model_key, 'model_name')}/` or specify the path above.
                3. Ensure `torch` and `transformers` are installed (`pip install -r requirements.txt`).
                4. Refresh this page to immediately activate real-time neural inference.
                """
            )
    else:
        st.markdown(
            f"""
            <div class="status-alert status-alert-green">
                <div>
                    <b>✓ Model Weights Active for {selected_model_key}</b><br>
                    Loaded checkpoint from: <code>{model_status['checkpoint_path']}</code>. Real-time inference is enabled.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Review text input
    st.subheader("Enter Product Review")

    # Sample reviews dropdown for quick testing
    sample_reviews = {
        "— Select a sample review —": "",
        "[EN - Positive] Excellent build quality and fast shipping. Highly recommended!": (
            "Excellent build quality and fast shipping. Highly recommended!"
        ),
        "[EN - Neutral] The product works fine, but took three weeks to arrive and packaging was damaged.": (
            "The product works fine, but took three weeks to arrive and packaging was damaged."
        ),
        "[EN - Negative] Stopped working after two days. Extremely cheap plastic and terrible customer support.": (
            "Stopped working after two days. Extremely cheap plastic and terrible customer support."
        ),
        "[ES - Positive] Me ha encantado el libro. Excelente presentación y llegó a tiempo.": (
            "Me ha encantado el libro. Excelente presentación y llegó a tiempo."
        ),
        "[ES - Neutral] El colchón está bien para el precio, pero el envío tardó demasiado.": (
            "El colchón está bien para el precio, pero el envío tardó demasiado."
        ),
        "[ES - Negative] Pésima calidad, no lo compren. Se rompió en el primer uso.": (
            "Pésima calidad, no lo compren. Se rompió en el primer uso."
        ),
    }

    sample_selection = st.selectbox("Quick-fill with an Amazon review excerpt:", list(sample_reviews.keys()))
    default_text = sample_reviews[sample_selection]

    review_input = st.text_area(
        "Review Text:",
        value=default_text,
        height=130,
        placeholder="Type or paste an English or Spanish product review here...",
    )

    if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
        if not review_input.strip():
            st.warning("Please enter some review text to analyze.")
        else:
            result = inferencer.predict(review_input, selected_model_key, active_path)

            if not result["success"]:
                st.error(f"Cannot perform inference: {result['error']}")
                if "notice" in result:
                    st.info(result["notice"])
            else:
                st.success("✓ Inference Complete")
                pred = result["prediction"]
                conf = result["confidence"]
                probs = result["probabilities"]

                st.markdown(
                    f"""
                    ### Predicted Sentiment: <span class="badge-{pred}">{pred.upper()}</span>
                    **Winning Confidence:** {conf*100:.2f}%
                    """,
                    unsafe_allow_html=True,
                )

                # Probabilities bar chart
                prob_df = pd.DataFrame([
                    {"Sentiment": c.capitalize(), "Probability": probs[c]} for c in SENTIMENT_CLASSES
                ])
                st.bar_chart(prob_df.set_index("Sentiment"))
