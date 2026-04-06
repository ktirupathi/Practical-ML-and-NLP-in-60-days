# Project 7: Financial News Risk Analyzer

End-to-end transformer-based NLP project for financial sentiment and risk analysis.
Fine-tunes FinBERT (or DistilBERT) on the Financial PhraseBank dataset to classify
financial news sentences as positive, negative, or neutral, and computes a
composite risk score.

## Architecture

```
Financial Text --> BERT Tokenizer --> Fine-tuned FinBERT --> Sentiment + Confidence
                                                        --> Risk Score Computation
```

## Transformer Fine-Tuning Approach

### Why FinBERT?
FinBERT (ProsusAI/finbert) is a BERT model pre-trained on financial communication
text (corporate reports, earnings calls, analyst reports). It understands financial
language nuances that general-purpose models miss -- e.g., "restructuring charges"
carries negative sentiment in finance but is neutral in general English.

### Fine-Tuning Strategy
1. **Feature extraction vs. fine-tuning**: We fine-tune the full model (not just the
   classification head) because the Financial PhraseBank is small (~2-5K samples).
   Fine-tuning all layers allows the model to adapt its representations.

2. **Learning rate schedule**: We use a low learning rate (2e-5) with linear warmup
   over 10% of training steps, then linear decay. This prevents catastrophic
   forgetting of pre-trained knowledge.

3. **Early stopping**: We monitor validation loss with patience=3 to prevent
   overfitting on the small dataset.

4. **Class weights**: Optional weighted loss to handle the imbalanced label
   distribution (neutral >> positive >> negative).

### Risk Score Computation
The risk score combines:
- **Sentiment polarity**: Negative sentiment increases risk
- **Confidence**: Low-confidence predictions suggest ambiguity (moderate risk)
- **Calibrated probabilities**: Temperature-scaled softmax for reliable confidence

```
risk_score = w_neg * P(negative) + w_neutral * P(neutral) * (1 - confidence) + base
```

## Project Structure

```
07-financial-news-risk-analyzer/
|-- app.py                          # FastAPI serving endpoint
|-- train.py                        # Training entry point
|-- predict.py                      # CLI prediction script
|-- requirements.txt
|-- dataset_link.md
|-- README.md
|-- logs/
|-- notebooks/
|-- src/
    |-- __init__.py
    |-- components/
    |   |-- __init__.py
    |   |-- data_ingestion.py       # Load from HuggingFace datasets
    |   |-- data_validation.py      # Validate data quality
    |   |-- data_transformation.py  # BERT tokenization, PyTorch datasets
    |   |-- model_trainer.py        # Fine-tune with HF Trainer API
    |   |-- model_evaluation.py     # Metrics, calibration, risk scores
    |-- pipeline/
    |   |-- __init__.py
    |   |-- training_pipeline.py
    |   |-- prediction_pipeline.py
    |-- utils/
    |   |-- __init__.py
    |   |-- common.py
    |-- config/
        |-- __init__.py
        |-- configuration.py
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

Options:
```bash
python train.py --model_name ProsusAI/finbert \
                --agreement_level sentences_allagree \
                --epochs 5 \
                --batch_size 16 \
                --learning_rate 2e-5
```

### Prediction
```bash
python predict.py --text "The company reported a 30% decline in quarterly revenue"
```

### API Server
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then query:
```bash
curl -X POST http://localhost:8000/analyze-risk \
  -H "Content-Type: application/json" \
  -d '{"text": "Operating profit surged 45% driven by strong demand"}'
```

Response:
```json
{
  "text": "Operating profit surged 45% driven by strong demand",
  "sentiment": "positive",
  "confidence": 0.96,
  "risk_score": 0.08,
  "probabilities": {
    "negative": 0.02,
    "neutral": 0.02,
    "positive": 0.96
  }
}
```

## Dataset
Financial PhraseBank: https://huggingface.co/datasets/financial_phrasebank

See `dataset_link.md` for full details on schema, agreement levels, and preprocessing.

## Model Performance (Expected)
| Metric    | FinBERT (allagree) | DistilBERT (allagree) |
|-----------|--------------------|-----------------------|
| Accuracy  | ~0.92              | ~0.87                 |
| F1 (macro)| ~0.90              | ~0.84                 |
| F1 (neg)  | ~0.85              | ~0.78                 |

## References
- Malo et al. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts.
- Araci (2019). FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models.
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers.
