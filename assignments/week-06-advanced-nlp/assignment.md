# Week 6: Advanced NLP

## Topics Covered
Word Embeddings, Transformers, Attention Mechanisms, BERT, Text Summarization, Evaluation Metrics (BLEU, ROUGE)

---

## Part 1: Conceptual Questions (10 Questions)

### Q1. Word2Vec Architectures
**Difficulty:** Easy | **Estimated Time:** 5 minutes

Explain the difference between the Skip-gram and CBOW (Continuous Bag of Words) architectures in Word2Vec. Which architecture tends to perform better on rare words and why?

---

### Q2. Word Embedding Properties
**Difficulty:** Easy | **Estimated Time:** 5 minutes

What does it mean for word embeddings to capture "semantic relationships"? Illustrate with the classic analogy: `king - man + woman = queen`. What linear algebraic operation makes this possible?

---

### Q3. Limitations of Static Embeddings
**Difficulty:** Easy | **Estimated Time:** 5 minutes

Word2Vec and GloVe produce a single vector per word regardless of context. Explain why this is a limitation using the word "bank" as an example. How do contextual embeddings (ELMo, BERT) address this?

---

### Q4. The Attention Mechanism
**Difficulty:** Medium | **Estimated Time:** 10 minutes

Describe the scaled dot-product attention mechanism. Write the formula `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V` and explain: (a) What are Q, K, and V? (b) Why do we scale by `sqrt(d_k)`? (c) What happens without the scaling factor for large `d_k`?

---

### Q5. Multi-Head Attention
**Difficulty:** Medium | **Estimated Time:** 10 minutes

Why does the Transformer use multi-head attention instead of a single attention function? What does each "head" learn to attend to? If a model has 8 heads with `d_model=512`, what is the dimensionality of each head?

---

### Q6. Transformer Positional Encoding
**Difficulty:** Medium | **Estimated Time:** 10 minutes

The Transformer architecture has no built-in notion of token order. Explain how sinusoidal positional encodings solve this. Write the positional encoding formula and explain why both sine and cosine functions are used at different frequencies.

---

### Q7. BERT Pre-training Objectives
**Difficulty:** Medium | **Estimated Time:** 10 minutes

BERT is pre-trained with two objectives: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). Describe each objective in detail. Why was the MLM objective a departure from traditional left-to-right language models, and how does it enable bidirectional context?

---

### Q8. BERT vs. GPT Architecture
**Difficulty:** Hard | **Estimated Time:** 15 minutes

Compare BERT (encoder-only) and GPT (decoder-only) architectures. Why is BERT better suited for classification and extraction tasks while GPT excels at text generation? Discuss the role of masked self-attention in GPT versus full self-attention in BERT.

---

### Q9. BLEU and ROUGE Metrics
**Difficulty:** Hard | **Estimated Time:** 15 minutes

Explain BLEU and ROUGE evaluation metrics. How does BLEU measure precision using n-gram overlap? How does ROUGE measure recall? Why is BLEU preferred for machine translation while ROUGE is preferred for summarization? What is the brevity penalty in BLEU?

---

### Q10. Transfer Learning in NLP
**Difficulty:** Hard | **Estimated Time:** 15 minutes

Explain the "pre-train then fine-tune" paradigm in modern NLP. Why is pre-training on a large unlabeled corpus followed by fine-tuning on a small labeled dataset so effective? Compare this to training from scratch. Discuss catastrophic forgetting and strategies to mitigate it (learning rate scheduling, gradual unfreezing).

---

## Part 2: Coding Questions (10 Questions)

### CQ1. Train Word2Vec with Gensim
**Difficulty:** Easy | **Estimated Time:** 15 minutes

Using the `gensim` library, train a Word2Vec model on a list of tokenized sentences. Find the 5 most similar words to "machine" and demonstrate vector arithmetic (e.g., `king - man + woman`).

```python
from gensim.models import Word2Vec

sentences = [
    ["machine", "learning", "is", "a", "subset", "of", "artificial", "intelligence"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["natural", "language", "processing", "deals", "with", "text", "data"],
    # Add at least 20 more sentences for meaningful embeddings
]

# TODO: Train Word2Vec model with vector_size=100, window=5, min_count=1
# TODO: Find most_similar words to "machine"
# TODO: Perform analogy: "machine" - "learning" + "processing" = ?
```

---

### CQ2. Compute Scaled Dot-Product Attention
**Difficulty:** Medium | **Estimated Time:** 20 minutes

Implement scaled dot-product attention from scratch using NumPy.

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Args:
        Q: Query matrix (seq_len_q, d_k)
        K: Key matrix (seq_len_k, d_k)
        V: Value matrix (seq_len_k, d_v)
    Returns:
        output: Attention output (seq_len_q, d_v)
        weights: Attention weights (seq_len_q, seq_len_k)
    """
    # TODO: Implement the attention formula
    pass

# Test with random matrices
Q = np.random.randn(4, 8)
K = np.random.randn(6, 8)
V = np.random.randn(6, 16)
output, weights = scaled_dot_product_attention(Q, K, V)
# Verify: output.shape should be (4, 16), weights rows should sum to 1
```

---

### CQ3. Tokenization with HuggingFace
**Difficulty:** Easy | **Estimated Time:** 10 minutes

Use the HuggingFace `transformers` library to tokenize a sentence using the BERT tokenizer. Show the token IDs, attention mask, and decode back to text. Explain what `[CLS]` and `[SEP]` tokens are for.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "The Transformer architecture revolutionized natural language processing."

# TODO: Tokenize the text, show input_ids, attention_mask, and token_type_ids
# TODO: Decode the token IDs back to readable text
# TODO: Show the individual tokens using tokenizer.tokenize()
```

---

### CQ4. Sentence Embeddings with BERT
**Difficulty:** Medium | **Estimated Time:** 20 minutes

Extract sentence embeddings from BERT by mean-pooling the last hidden state. Compute cosine similarity between two sentences.

```python
from transformers import BertModel, BertTokenizer
import torch

# TODO: Load bert-base-uncased model and tokenizer
# TODO: Encode two sentences: "I love machine learning" and "AI is fascinating"
# TODO: Extract last_hidden_state, apply mean pooling (respecting attention mask)
# TODO: Compute cosine similarity between the two sentence embeddings
```

---

### CQ5. Fine-tune BERT for Text Classification
**Difficulty:** Hard | **Estimated Time:** 30 minutes

Write the training loop to fine-tune `bert-base-uncased` on a binary classification task using HuggingFace `Trainer` API.

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# TODO: Load the "imdb" dataset (use a small subset for quick training)
# TODO: Tokenize the dataset using BertTokenizer
# TODO: Define TrainingArguments (learning_rate=2e-5, num_train_epochs=2, batch_size=16)
# TODO: Create Trainer and call trainer.train()
# TODO: Evaluate on the test set and print accuracy
```

---

### CQ6. Compute BLEU Score
**Difficulty:** Medium | **Estimated Time:** 15 minutes

Compute the BLEU score between a reference and candidate translation using `nltk`.

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate = ["the", "cat", "is", "on", "the", "mat"]

# TODO: Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores
# TODO: Compute cumulative BLEU score with smoothing
# TODO: Explain what happens when candidate length is much shorter than reference
```

---

### CQ7. Compute ROUGE Score
**Difficulty:** Medium | **Estimated Time:** 15 minutes

Use the `rouge_score` library to compute ROUGE-1, ROUGE-2, and ROUGE-L scores between a reference and generated summary.

```python
from rouge_score import rouge_scorer

reference = "The quick brown fox jumps over the lazy dog near the river bank."
generated = "A fast brown fox leaps over a lazy dog by the river."

# TODO: Initialize scorer with ["rouge1", "rouge2", "rougeL"]
# TODO: Compute scores and print precision, recall, and F1 for each metric
# TODO: Explain the difference between ROUGE-1, ROUGE-2, and ROUGE-L
```

---

### CQ8. Word2Vec Visualization with t-SNE
**Difficulty:** Medium | **Estimated Time:** 20 minutes

Train a Word2Vec model and visualize the embeddings in 2D using t-SNE.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# TODO: Train or load a Word2Vec model
# TODO: Select 50 words and extract their vectors
# TODO: Apply t-SNE to reduce to 2 dimensions
# TODO: Plot the 2D embeddings with word labels
# TODO: Highlight clusters of semantically related words in different colors
```

---

### CQ9. Build an Extractive Summarizer
**Difficulty:** Hard | **Estimated Time:** 30 minutes

Build an extractive text summarizer using TF-IDF sentence scoring.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extractive_summarize(text, num_sentences=3):
    """
    Split text into sentences, score each by average TF-IDF weight,
    and return the top-scoring sentences in original order.
    """
    # TODO: Split text into sentences
    # TODO: Compute TF-IDF matrix for the sentences
    # TODO: Score each sentence by its mean TF-IDF value
    # TODO: Select top-k sentences and return them in original order
    pass
```

---

### CQ10. Multi-Head Attention in PyTorch
**Difficulty:** Hard | **Estimated Time:** 30 minutes

Implement multi-head attention using PyTorch (without using `nn.MultiheadAttention`).

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # TODO: Initialize linear projections for Q, K, V, and output
        pass

    def forward(self, query, key, value, mask=None):
        # TODO: Project inputs, split into heads, compute attention, concatenate
        pass

# Test: batch_size=2, seq_len=10, d_model=512, num_heads=8
mha = MultiHeadAttention(512, 8)
x = torch.randn(2, 10, 512)
output = mha(x, x, x)  # Self-attention
assert output.shape == (2, 10, 512)
```

---

## Part 3: Case Studies (2 Studies)

### Case Study 1: Document Search Engine
**Scenario:** You are building a document search engine for a legal firm with 100,000 legal documents. Users type natural language queries and expect relevant documents ranked by relevance.

**Tasks:**
1. Compare TF-IDF-based search versus embedding-based semantic search for this use case.
2. Design a pipeline that uses BERT embeddings for semantic search. How would you encode and index 100K documents efficiently?
3. What trade-offs exist between retrieval speed and accuracy? How would approximate nearest neighbor (ANN) search help?
4. How would you handle domain-specific legal terminology that may not be well-represented in a general BERT model?

---

### Case Study 2: Multi-Language Sentiment Analysis Pipeline
**Scenario:** An e-commerce company wants to analyze customer reviews in English, Spanish, and French. They have 50K labeled English reviews but only 2K labeled reviews in each of the other languages.

**Tasks:**
1. Propose a strategy leveraging multilingual models (e.g., mBERT, XLM-RoBERTa).
2. Explain zero-shot cross-lingual transfer: fine-tune on English, evaluate on Spanish/French.
3. How would you handle code-switching (reviews mixing two languages)?
4. Design the end-to-end pipeline including preprocessing, model selection, and deployment.

---

## Part 4: Practical Assignments (3 Assignments)

### Assignment 1: Train Word2Vec and Visualize with t-SNE
**Objective:** Train word embeddings from scratch and visualize semantic clusters.

**Dataset:** [text8 dataset](http://mattmahoney.net/dc/text8.zip) (100MB cleaned Wikipedia text)

**Requirements:**
1. Download and preprocess the text8 dataset into tokenized sentences.
2. Train a Word2Vec model with `vector_size=200, window=5, min_count=5, workers=4, epochs=10`.
3. Evaluate with analogy tasks: `king - man + woman = ?`, `paris - france + germany = ?`.
4. Select 100 words from 5 semantic categories (animals, countries, colors, professions, sports).
5. Apply t-SNE (perplexity=30) and create a scatter plot with color-coded categories.
6. Save the trained model and the visualization as a PNG.

**Hints:**
- Use `gensim.utils.simple_preprocess` for tokenization.
- Set `random_state=42` in t-SNE for reproducibility.
- Annotate interesting clusters and outliers in the plot.

**Deliverables:** Jupyter notebook with code, t-SNE visualization, and a 200-word analysis of the clusters.

---

### Assignment 2: Fine-tune BERT on IMDB Sentiment Classification
**Objective:** Fine-tune a pre-trained BERT model on the full IMDB dataset (50K reviews).

**Dataset:** [IMDB Dataset](https://huggingface.co/datasets/imdb) via HuggingFace Datasets (25K train, 25K test)

**Requirements:**
1. Load the IMDB dataset using `datasets.load_dataset("imdb")`.
2. Tokenize all reviews with `BertTokenizer` (max_length=256, truncation, padding).
3. Fine-tune `bert-base-uncased` with `BertForSequenceClassification`.
4. Training hyperparameters: learning_rate=2e-5, batch_size=16, epochs=3, weight_decay=0.01.
5. Implement early stopping based on validation loss.
6. Report accuracy, precision, recall, and F1 on the test set.
7. Plot training and validation loss curves.
8. Show 5 misclassified examples and analyze why the model failed.

**Hints:**
- Use a GPU runtime (Colab or similar) for reasonable training times.
- Create a validation split from training data (90/10 split).
- Use `Trainer` API with `compute_metrics` function for evaluation.
- Training should take approximately 30-45 minutes on a single GPU.

**Deliverables:** Jupyter notebook, training curves plot, classification report, error analysis.

---

### Assignment 3: Build an Extractive Text Summarizer
**Objective:** Build and evaluate an extractive summarization system using multiple scoring methods.

**Dataset:** [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) via HuggingFace Datasets

**Requirements:**
1. Load 1,000 articles from the CNN/DailyMail dataset (version 3.0.0).
2. Implement three sentence scoring methods:
   - TF-IDF average score per sentence
   - TextRank (graph-based, similar to PageRank for sentences)
   - Lead-3 baseline (select first 3 sentences)
3. For each method, select the top 3 sentences as the summary.
4. Evaluate all three methods using ROUGE-1, ROUGE-2, and ROUGE-L against reference summaries.
5. Compare results in a table and discuss which method works best and why.
6. Implement a simple sentence redundancy filter using cosine similarity (threshold=0.7).

**Hints:**
- Use `nltk.sent_tokenize` for sentence splitting.
- For TextRank, build a similarity matrix and use `networkx.pagerank`.
- The Lead-3 baseline is surprisingly strong for news articles.
- Use the `rouge_score` Python package for evaluation.

**Deliverables:** Jupyter notebook, comparison table, 300-word discussion of results.
