"""
Inference engine for fine-tuned Hugging Face Transformers and PyTorch.
Enforces strict transparency:
- Connects to real fine-tuned model weights when available in models/ directory.
- Clearly marks live inference as UNAVAILABLE when weights are missing.
- NEVER fabricates predictions, scores, or mock confidence values.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import json

from config import MODELS_DIR, SENTIMENT_CLASSES, MODEL_CONFIG


class ModelInferencer:
    """Manages model checkpoint discovery, loading, and real inference."""

    MODEL_DIR_MAP = {
        "mBERT": "mbert",
        "XLM-R": "xlm-roberta",
        "mDeBERTa-v3": "mdeberta-v3",
    }

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self._loaded_models = {}
        self._loaded_tokenizers = {}

    def check_weights_available(
        self, model_key: str, custom_path: Optional[str] = None
    ) -> Tuple[bool, Optional[Path], str]:
        """
        Checks whether valid fine-tuned model weights exist on disk.
        Looks for standard Hugging Face files: (pytorch_model.bin or model.safetensors) + config.json.
        """
        if custom_path and Path(custom_path).exists():
            search_dir = Path(custom_path)
        else:
            dir_name = self.MODEL_DIR_MAP.get(model_key, model_key.lower().replace("-", "_"))
            search_dir = self.models_dir / dir_name

        if not search_dir.exists():
            return (
                False,
                None,
                f"Directory '{search_dir}' does not exist. The submitted experiment archive only included evaluation CSVs and figures, not the fine-tuned model weights.",
            )

        weight_files = list(search_dir.glob("*.bin")) + list(search_dir.glob("*.safetensors"))
        config_files = list(search_dir.glob("config.json"))

        if not weight_files:
            return (
                False,
                search_dir,
                f"No weight files (.bin or .safetensors) found in '{search_dir}'.",
            )

        if not config_files:
            return (
                False,
                search_dir,
                f"Model weights found, but 'config.json' is missing in '{search_dir}'.",
            )

        return True, search_dir, f"Valid weights found at '{search_dir}'."

    def get_status(self, model_key: str, custom_path: Optional[str] = None) -> Dict[str, Any]:
        """Returns comprehensive status dictionary for a model."""
        has_weights, path, message = self.check_weights_available(model_key, custom_path)

        # Check torch & transformers dependencies
        torch_ok = False
        transformers_ok = False
        try:
            import torch
            torch_ok = True
        except ImportError:
            pass

        try:
            import transformers
            transformers_ok = True
        except ImportError:
            pass

        is_ready = has_weights and torch_ok and transformers_ok

        return {
            "model_key": model_key,
            "display_name": MODEL_CONFIG.get(model_key, {}).get("display_name", model_key),
            "weights_available": has_weights,
            "checkpoint_path": str(path) if path else None,
            "torch_installed": torch_ok,
            "transformers_installed": transformers_ok,
            "ready_for_inference": is_ready,
            "status_message": message,
            "how_to_enable": (
                "To enable live inference for this model:\n"
                f"1. Save your fine-tuned checkpoint files ('pytorch_model.bin' or 'model.safetensors' and 'config.json') into '{self.models_dir}/{self.MODEL_DIR_MAP.get(model_key, 'model_name')}/'.\n"
                "2. Ensure PyTorch and Hugging Face Transformers are installed in your Python environment.\n"
                "3. Refresh this page to automatically detect the weights."
            ),
        }

    def predict(
        self,
        text: str,
        model_key: str,
        custom_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Executes real model inference if weights are present.
        STRICT POLICY: Never fabricates predictions or returns mock probabilities.
        """
        status = self.get_status(model_key, custom_path)

        if not status["ready_for_inference"]:
            return {
                "success": False,
                "error": status["status_message"],
                "model": model_key,
                "prediction": None,
                "confidence": None,
                "probabilities": None,
                "notice": (
                    "This system never fabricates predictions or model scores. "
                    "Live inference is strictly disabled because genuine trained model weights "
                    "were not provided in the experiment bundle. Synthetic or fabricated scores "
                    "are prohibited to maintain research integrity."
                ),
            }

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            checkpoint_path = status["checkpoint_path"]

            # Load or retrieve cached model
            if model_key not in self._loaded_models:
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
                model.eval()
                self._loaded_models[model_key] = model
                self._loaded_tokenizers[model_key] = tokenizer
            else:
                model = self._loaded_models[model_key]
                tokenizer = self._loaded_tokenizers[model_key]

            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            pred_idx = int(probs.argmax())
            pred_label = SENTIMENT_CLASSES[pred_idx]
            confidence = float(probs[pred_idx])

            prob_dict = {
                SENTIMENT_CLASSES[i]: float(probs[i]) for i in range(len(SENTIMENT_CLASSES))
            }

            return {
                "success": True,
                "model": model_key,
                "prediction": pred_label,
                "confidence": confidence,
                "probabilities": prob_dict,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error during model inference: {str(e)}",
                "model": model_key,
                "prediction": None,
                "confidence": None,
                "probabilities": None,
            }
