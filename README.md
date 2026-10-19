# Multilingual Review Intelligence 🌐

**A Comparative Evaluation and Analytics Dashboard for Multilingual Transformers**  
*Evaluating mBERT, XLM-RoBERTa, and mDeBERTa-v3 on English and Spanish Product Reviews*

---

## 📌 Executive Summary

**Multilingual Review Intelligence** is an enterprise-grade analytics dashboard and diagnostic platform designed to benchmark and interpret multilingual Transformer models (**mBERT**, **XLM-RoBERTa**, and **mDeBERTa-v3**) fine-tuned for three-class sentiment classification (*negative*, *neutral*, *positive*) on product reviews from the **Multilingual Amazon Reviews Corpus**.

The platform is built on real experimental data across two distinct evaluation regimes:
1. **Bilingual Evaluation:** Models trained on 2,400 balanced English and Spanish reviews (1,200 EN + 1,200 ES) and evaluated on 1,000 held-out reviews (500 EN + 500 ES).
2. **Zero-Shot Cross-Lingual Transfer:** Models fine-tuned strictly on 1,200 English reviews and tested on 500 Spanish reviews with zero Spanish task supervision.

### Key Benchmark Findings
- **Overall Winner:** `mDeBERTa-v3` achieves the highest performance across all benchmarks: **78.00% accuracy** and **73.27% Macro-F1** in bilingual testing, and **74.40% accuracy** and **65.66% Macro-F1** in cross-lingual transfer.
- **Pretraining Scaling:** `XLM-R` (**75.80% Acc / 69.01% Macro-F1**) significantly outperforms Wikipedia-trained `mBERT` (**72.40% Acc / 66.72% Macro-F1**), demonstrating the value of diverse web-crawl pretraining (CC-100) for informal product review language.
- **The Neutral Bottleneck:** Across all architectures, neutral reviews (3 stars) represent the primary failure mode due to mixed sentiment signals and star-rating subjectivity. In cross-lingual transfer, neutral recall collapses down to **8.82%** for mBERT, **13.73%** for XLM-R, and **25.49%** for mDeBERTa-v3.

---

## 🚀 Features & Capabilities

| # | Feature | Status with Existing Results | Requirement for Saved Weights |
|---|---|---|---|
| **1** | **Overview & Summary** | ✅ **Fully Operational** | None (uses experimental data) |
| **2** | **Model Comparison** | ✅ **Fully Operational** | None (uses actual metric outputs) |
| **3** | **Language & Transfer Analysis** | ✅ **Fully Operational** | None (uses language-wise metrics) |
| **4** | **Review Explorer** | ✅ **Fully Operational** | None (searches 4,500 actual predictions) |
| **5** | **Error Analysis & Confusion Matrices** | ✅ **Fully Operational** | None (computes real confusion matrices & confidence) |
| **6** | **CSV Upload & Validation** | ✅ **Fully Operational** | None (validates schema & computes dynamic metrics) |
| **7** | **Live Sentiment Analyzer** | ⚠️ **Interface Ready & Protected** | **Requires fine-tuned model checkpoint** |

### Feature Details
1. **Overview & Summary:** Dynamic KPI cards, true vs predicted sentiment distributions, experimental split configurations, and downloadable executive summary reports.
2. **Model Comparison:** Interactive Bar, Grouped Multi-Metric, and Radar/Spider charts comparing Accuracy, Macro Precision, Macro Recall, Macro-F1, and Weighted-F1, paired with architectural and pretraining analysis.
3. **Language & Transfer Analysis:** English vs Spanish language breakdown, paired with an isolated analysis of zero-shot transfer degradation (-6.17 pp Macro-F1 drop for mDeBERTa-v3, -7.83 pp for XLM-R, -11.75 pp for mBERT).
4. **Review Explorer:** Real-time multi-criteria filtering across 4,500 predictions by experiment, model, language, true label, predicted label, correctness, confidence range, and full-text keyword search, with CSV export.
5. **Error Analysis:** Dynamic interactive confusion matrix heatmaps, prediction confidence distributions for correct vs misclassified items, high-confidence error audits, and qualitative case studies.
6. **CSV Upload & Validation:** Dual-mode CSV ingestion engine that validates schemas for either prediction evaluations or raw reviews, computes live classification metrics, and renders interactive confusion matrices on user data.
7. **Live Sentiment Analyzer:** Transparent neural inference interface connected to Hugging Face Transformers and PyTorch. **Strict ML Integrity Policy:** When weights are not supplied, the interface clearly marks inference as unavailable and provides setup instructions; it **never fabricates predictions or mock probabilities**.

---

## 🏗️ Architecture & Directory Structure

```
multilingual-review-intelligence/
├── app.py                      # Main Streamlit dashboard application
├── config.py                   # Central constants, paths, color palettes & CSS
├── requirements.txt            # Python dependencies (Streamlit, Pandas, Plotly, etc.)
├── README.md                   # Complete documentation and setup guide
├── data/                       # Experimental CSVs and reference figures
│   ├── overall_metrics.csv
│   ├── bilingual_comparison.csv
│   ├── cross_lingual_comparison.csv
│   ├── language_wise_metrics.csv
│   ├── model_characteristics.csv
│   ├── error_summary.csv
│   ├── predictions.csv
│   ├── error_analysis.csv
│   └── *.png                   # Reference evaluation plots
├── models/                     # Directory for fine-tuned Transformer weights
│   ├── README.md               # Instructions for adding model checkpoints
│   ├── mdeberta-v3/            # (Optional) Drop mDeBERTa-v3 weights here
│   ├── xlm-roberta/            # (Optional) Drop XLM-RoBERTa weights here
│   └── mbert/                  # (Optional) Drop mBERT weights here
├── utils/                      # Modular backend packages
│   ├── __init__.py
│   ├── data_loader.py          # Robust CSV loader, caching, prediction filters & schema validator
│   ├── metrics.py              # Pure-python / NumPy multi-class metric calculator
│   ├── charts.py               # Interactive Plotly chart builders (Radar, Heatmaps, Distributions)
│   ├── inference.py            # Hugging Face inference pipeline with transparent weight check
│   └── export_helpers.py       # CSV byte export and markdown report generation
└── tests/                      # Automated unit and integration test suite
    ├── __init__.py
    ├── test_data_loader.py     # Tests for loading, schemas, and multi-filter logic
    ├── test_metrics.py         # Tests verifying metrics match experimental numbers exactly
    ├── test_upload_validator.py# Tests for CSV upload schema validation
    ├── test_inference.py       # Tests verifying anti-fabrication policy
    └── run_all_tests.py        # Master test runner
```

---

## 💻 Installation & Local Setup

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- `pip` package manager

### 1. Clone or Extract the Repository
```bash
git clone <repo-url>
cd multilingual-review-intelligence
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Automated Test Suite
Verify that all data loaders, metric formulas, and validation rules pass:
```bash
python3 tests/run_all_tests.py
```
*(All 20 unit and integration tests should pass with `OK`)*

### 5. Launch the Streamlit Application
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 🧠 Connecting Live Model Weights (Feature 7)

By default, the application runs in **Evaluation & Analytics Mode** using the 4,500 real predictions and experiment outputs. The experimental results bundle does not include the ~1.5 GB PyTorch checkpoint weights for the models.

To enable **real-time live inference** on new reviews:
1. Export or copy your fine-tuned model checkpoint directory containing:
   - `pytorch_model.bin` (or `model.safetensors`)
   - `config.json`
   - Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `spm.model` or `vocab.txt`)
2. Place the checkpoint directory into the appropriate subfolder inside `models/`:
   - `models/mdeberta-v3/` for mDeBERTa-v3
   - `models/xlm-roberta/` for XLM-RoBERTa
   - `models/mbert/` for mBERT
   *(Alternatively, provide any custom path directly in the web UI)*
3. Refresh the Streamlit page. The system will detect the checkpoint, display an active status badge, and perform real-time forward passes using PyTorch.

---

## 📊 Data Dictionary & Experimental Specifications

| Dataset File | Rows | Columns | Purpose |
|---|---|---|---|
| `bilingual_comparison.csv` | 3 | 8 | Overall metrics on 1,000 bilingual test reviews across models |
| `cross_lingual_comparison.csv` | 3 | 8 | Overall metrics on 500 Spanish reviews for English-to-Spanish transfer |
| `overall_metrics.csv` | 6 | 8 | Unified metrics across both bilingual and cross-lingual experiments |
| `language_wise_metrics.csv` | 9 | 9 | Language-specific breakdowns (EN test vs ES test) |
| `model_characteristics.csv` | 3 | 6 | Architecture, pretraining corpus, vocab size, and analysis focus |
| `error_summary.csv` | 9 | 6 | Error rates and sample counts by model, experiment, and language |
| `predictions.csv` | 4,500 | 8 | Complete review texts, true labels, predictions, confidence, correctness |
| `error_analysis.csv` | 1,178 | 8 | Filtered subset of all 1,178 misclassified review records |

---

## 🛡️ Machine Learning Integrity Policy
This platform strictly adheres to ML engineering best practices:
- **No Hardcoded Scores:** All metrics, totals, and percentages are computed dynamically from data.
- **Zero Hallucination / Mocking:** If model weights are missing, the system refuses to generate synthetic probabilities or fake classifications.
- **Transparent Attribution:** Experimental configurations, label definitions, and error taxonomies are documented explicitly.
