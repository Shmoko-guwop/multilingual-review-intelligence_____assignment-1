"""
Configuration constants, color palettes, and custom styling for Multilingual Review Intelligence.
"""
from pathlib import Path

# Application metadata
APP_TITLE = "Multilingual Review Intelligence"
APP_SUBTITLE = "Comparative Evaluation of Multilingual Transformers for Sentiment Analysis"
APP_ICON = "🌐"

# Base paths (robust relative path resolution)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Sentiment classes and styling
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]

SENTIMENT_COLORS = {
    "negative": "#EF4444",  # Crimson Red
    "neutral": "#F59E0B",   # Warm Amber
    "positive": "#10B981",  # Emerald Green
}

# Model colors and display names
MODEL_CONFIG = {
    "mBERT": {
        "display_name": "mBERT (bert-base-multilingual-cased)",
        "short_name": "mBERT",
        "color": "#8B5CF6",  # Violet
        "checkpoint": "google-bert/bert-base-multilingual-cased",
    },
    "XLM-R": {
        "display_name": "XLM-RoBERTa (xlm-roberta-base)",
        "short_name": "XLM-R",
        "color": "#06B6D4",  # Cyan/Teal
        "checkpoint": "FacebookAI/xlm-roberta-base",
    },
    "mDeBERTa-v3": {
        "display_name": "mDeBERTa-v3 (mdeberta-v3-base)",
        "short_name": "mDeBERTa-v3",
        "color": "#3B82F6",  # Royal Blue
        "checkpoint": "microsoft/mdeberta-v3-base",
    },
}

EXPERIMENT_CONFIG = {
    "bilingual": {
        "title": "Bilingual Experiment",
        "description": "Trained and evaluated on combined English and Spanish review subsets (1,000 test reviews: 500 EN + 500 ES).",
        "short_label": "Bilingual (EN + ES)",
    },
    "english_to_spanish": {
        "title": "Cross-Lingual Transfer Experiment",
        "description": "Trained on English reviews only; evaluated zero-shot on 500 Spanish reviews (no Spanish training supervision).",
        "short_label": "Zero-Shot Transfer (EN -> ES)",
    },
}

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
}

# Custom CSS for polished, responsive analytics dashboard layout
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 0.8rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 0.4rem;
    }
    
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    
    .kpi-subtext {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.35rem;
    }
    
    .badge-negative { background-color: rgba(239, 68, 68, 0.2); color: #FCA5A5; border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 4px; padding: 2px 6px; font-weight: 600; }
    .badge-neutral { background-color: rgba(245, 158, 11, 0.2); color: #FCD34D; border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 4px; padding: 2px 6px; font-weight: 600; }
    .badge-positive { background-color: rgba(16, 185, 129, 0.2); color: #6EE7B7; border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 4px; padding: 2px 6px; font-weight: 600; }
    
    .status-alert {
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    
    .status-alert-amber {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #FDE68A;
    }
    
    .status-alert-blue {
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.35);
        color: #BFDBFE;
    }
    
    .status-alert-green {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #A7F3D0;
    }
    
    .review-box {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(71, 85, 105, 0.4);
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.75rem;
    }
</style>
"""
