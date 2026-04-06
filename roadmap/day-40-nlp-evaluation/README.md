# Day 40: NLP Evaluation Metrics

## Learning Objectives

- Understand task-specific NLP evaluation metrics
- Compute BLEU for machine translation and text generation
- Compute ROUGE for summarization evaluation
- Apply F1, precision, and recall for token-level NLP tasks
- Understand perplexity for language model evaluation

## Key Concepts

NLP evaluation is more nuanced than standard ML classification metrics because natural language has many valid ways to express the same meaning. Different NLP tasks require different evaluation approaches.

**BLEU (Bilingual Evaluation Understudy)** measures n-gram overlap between generated text and reference text. It computes precision for 1-grams through 4-grams with a brevity penalty for short outputs. BLEU is widely used in machine translation but has limitations — it cannot capture semantic similarity or paraphrasing. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** focuses on recall and is standard for summarization evaluation. ROUGE-1 measures unigram overlap, ROUGE-2 bigram overlap, and ROUGE-L longest common subsequence.

For token-level tasks like NER and POS tagging, we use **entity-level F1**: a predicted entity is correct only if both the span boundaries and entity type match exactly. For classification tasks (sentiment, intent), standard precision/recall/F1 apply. **Perplexity** measures how well a language model predicts a sequence — lower perplexity means the model assigns higher probability to the actual text.

## Hands-On Exercise

```python
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ROUGE for summarization
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
reference = "The cat sat on the mat and looked out the window."
hypothesis = "The cat was sitting on the mat looking outside."

scores = scorer.score(reference, hypothesis)
for metric, score in scores.items():
    print(f"{metric}: P={score.precision:.3f} R={score.recall:.3f} F1={score.fmeasure:.3f}")

# BLEU for translation
reference_tokens = [reference.lower().split()]
hypothesis_tokens = hypothesis.lower().split()
smoothing = SmoothingFunction().method1

bleu = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing)
print(f"\nBLEU score: {bleu:.3f}")

# Entity-level F1 for NER
def entity_f1(true_entities, pred_entities):
    true_set = set(true_entities)
    pred_set = set(pred_entities)
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"precision": precision, "recall": recall, "f1": f1}

true_ents = [("Apple", "ORG"), ("California", "LOC")]
pred_ents = [("Apple", "ORG"), ("California", "GPE")]
print(f"\nEntity F1: {entity_f1(true_ents, pred_ents)}")
```

## Resources

- [BLEU Paper](https://aclanthology.org/P02-1040/)
- [ROUGE Package](https://pypi.org/project/rouge-score/)
- [HuggingFace Evaluate Library](https://huggingface.co/docs/evaluate/)

## Next Day Preview

Tomorrow we dive into **Word Embeddings** — Word2Vec, GloVe, and FastText — the foundation of modern NLP representation learning.
