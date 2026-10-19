# Model Checkpoints Directory

This directory is reserved for fine-tuned Hugging Face Transformer checkpoints and PyTorch model weights for the **Multilingual Review Intelligence** application.

## Directory Structure
To enable **Feature 7: Live Sentiment Analyzer**, place the exported model checkpoint files into their corresponding subdirectories:

```
models/
├── mdeberta-v3/
│   ├── config.json
│   ├── pytorch_model.bin  (or model.safetensors)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── spm.model
├── xlm-roberta/
│   ├── config.json
│   ├── pytorch_model.bin  (or model.safetensors)
│   ├── tokenizer.json
│   └── sentencepiece.bpe.model
└── mbert/
    ├── config.json
    ├── pytorch_model.bin  (or model.safetensors)
    ├── tokenizer_config.json
    └── vocab.txt
```

## Why are model weights not included by default?
The experiment results archive (`transformer_comparison_results.zip`) includes:
- Evaluation metrics (`overall_metrics.csv`, `bilingual_comparison.csv`, `cross_lingual_comparison.csv`, `language_wise_metrics.csv`)
- Complete predictions and error logs (`predictions.csv`, `error_analysis.csv`, `error_summary.csv`)
- High-resolution evaluation charts (`.png`)

Full deep learning model weights (~1.5 GB per model) were not part of the evaluation output zip. In accordance with ML engineering best practices, the application **never fabricates predictions or model scores**. When weights are not detected, the live analyzer explicitly notifies the user and provides setup instructions.

## How to Export Fine-Tuned Weights from Hugging Face Trainer
If you trained the model using Hugging Face `Trainer`:
```python
trainer.save_model("models/mdeberta-v3")
tokenizer.save_pretrained("models/mdeberta-v3")
```
Once the files are saved, restart or refresh the Streamlit app. The live inference interface will immediately detect the model and allow real-time classification of new reviews.
