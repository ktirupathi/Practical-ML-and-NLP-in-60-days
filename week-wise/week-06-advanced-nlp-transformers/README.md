# Week 6: Advanced NLP and Transformers

*Days 36-42: From word embeddings to transformer architecture*

---

## 1. Text Summarization

### What is it
Text summarization generates a shorter version of a document while preserving key information. **Extractive** methods select important sentences from the original text. **Abstractive** methods generate new sentences that paraphrase the original content.

### Why it matters
Information overload is real: legal documents, research papers, news articles, customer feedback. Automated summarization saves hours of reading time and enables scalable content processing.

### Python code
```python
# Extractive summarization with TextRank (via sumy)
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

text = """Machine learning is a subset of artificial intelligence. It enables 
computers to learn from data without being explicitly programmed. Supervised 
learning uses labeled data. Unsupervised learning finds hidden patterns. 
Deep learning uses neural networks with many layers."""

parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = TextRankSummarizer()
summary = summarizer(parser.document, sentences_count=2)
for sentence in summary:
    print(sentence)
```

### Common mistakes
- Evaluating summaries only with ROUGE without human evaluation
- Not handling very long documents (need chunking before summarization)
- Confusing extractive (selects sentences) with abstractive (generates new text)

### Interview questions
- **Q: What is the difference between extractive and abstractive summarization?** A: Extractive selects existing sentences; abstractive generates new paraphrased text.
- **Q: How does TextRank work?** A: It builds a graph of sentence similarities, then uses PageRank to identify the most central sentences.
- **Q: What metrics evaluate summarization quality?** A: ROUGE-1 (unigram overlap), ROUGE-2 (bigram), ROUGE-L (longest common subsequence).

---

## 2. NLP Evaluation (BLEU, ROUGE)

### What is it
NLP evaluation metrics measure the quality of generated text by comparing it against reference texts. BLEU measures precision-based n-gram overlap (used in translation). ROUGE measures recall-based overlap (used in summarization).

### Math

**BLEU:** Geometric mean of n-gram precisions with brevity penalty.

BLEU = BP * exp(sum(w_n * log(p_n))) where p_n = matching n-grams / total n-grams in candidate

**ROUGE-N:** Recall of n-gram overlap.

ROUGE-N = (matching n-grams in reference) / (total n-grams in reference)

### Python code
```python
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = "The cat sat on the mat and looked out the window"
hypothesis = "The cat was sitting on the mat looking outside"

# ROUGE
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference, hypothesis)
for metric, score in scores.items():
    print(f"{metric}: P={score.precision:.3f} R={score.recall:.3f} F1={score.fmeasure:.3f}")

# BLEU
ref_tokens = [reference.lower().split()]
hyp_tokens = hypothesis.lower().split()
bleu = sentence_bleu(ref_tokens, hyp_tokens,
                      smoothing_function=SmoothingFunction().method1)
print(f"BLEU: {bleu:.3f}")
```

### Interview questions
- **Q: Why is BLEU precision-based and ROUGE recall-based?** A: BLEU penalizes extra words in translations; ROUGE penalizes missing information in summaries.
- **Q: What are limitations of BLEU?** A: It cannot capture semantic similarity, only lexical overlap. Paraphrases score poorly.

---

## 3. Word Embeddings

### What is it
Word embeddings map words to dense vectors where semantic similarity corresponds to vector proximity. Unlike one-hot encoding (sparse, no similarity), embeddings capture meaning: king - man + woman = queen.

### Why it matters
Embeddings are the foundation of modern NLP. They enable models to understand semantic relationships, generalize across similar words, and work with fixed-size inputs regardless of vocabulary size.

### Math: Word2Vec Skip-gram

Objective: maximize P(context | center word)

P(w_o | w_c) = exp(v_o . v_c) / sum(exp(v_w . v_c)) for all w in vocab

In practice, negative sampling approximates this: for each positive pair, sample k negative pairs.

### Python code
```python
from gensim.models import Word2Vec
import numpy as np

# Train Word2Vec
sentences = [
    ["king", "queen", "royal", "palace"],
    ["man", "woman", "child", "family"],
    ["cat", "dog", "pet", "animal"],
    ["python", "java", "code", "programming"],
]
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1, epochs=100)

# Find similar words
if "king" in model.wv:
    similar = model.wv.most_similar("king", topn=3)
    print(f"Similar to 'king': {similar}")

# Embedding arithmetic
# result = model.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
```

### Common mistakes
- Training Word2Vec on too small a corpus (need millions of tokens for good embeddings)
- Not lowercasing text before training (Hello and hello get different vectors)
- Using embeddings from a different domain without fine-tuning

### Interview questions
- **Q: What is the difference between CBOW and Skip-gram?** A: CBOW predicts center word from context; Skip-gram predicts context from center word. Skip-gram works better for rare words.
- **Q: How does negative sampling work?** A: Instead of computing softmax over entire vocabulary, sample k random "negative" words and update only their weights.
- **Q: What advantage does FastText have over Word2Vec?** A: FastText uses character n-grams, so it can generate embeddings for out-of-vocabulary words.

---

## 4. Transformer Architecture

### What is it
The Transformer is a neural architecture based entirely on attention mechanisms, replacing recurrence (RNNs) and convolution. Introduced in "Attention Is All You Need" (2017), it is the foundation of BERT, GPT, T5, and all modern language models.

### Why it matters
Transformers solved the key limitations of RNNs: they process all tokens in parallel (faster training), they capture long-range dependencies through attention (no vanishing gradient), and they scale to billions of parameters.

### Intuition
Imagine reading a sentence: "The animal didn't cross the street because it was too tired." What does "it" refer to? Your brain attends to "animal" — that is exactly what self-attention does. Each word computes how much it should "attend to" every other word.

### Math: Self-Attention

Given input X, compute Query (Q), Key (K), Value (V):

Q = X * W_Q, K = X * W_K, V = X * W_V

Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V

The division by sqrt(d_k) prevents the dot products from growing too large, which would push softmax into regions with tiny gradients.

**Multi-head attention** runs h separate attention heads in parallel, then concatenates:

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W_O

### Python code
```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)

# Usage
attn = SelfAttention(d_model=64, n_heads=4)
x = torch.randn(2, 10, 64)  # batch=2, seq_len=10, d_model=64
output = attn(x)
print(f"Output shape: {output.shape}")  # [2, 10, 64]
```

### Common mistakes
- Forgetting that attention is O(n^2) in sequence length — long documents need chunking or sparse attention
- Not understanding positional encoding: without it, transformers are permutation-invariant (order doesn't matter)
- Confusing encoder-only (BERT), decoder-only (GPT), and encoder-decoder (T5) architectures

### Interview questions
- **Q: Why divide by sqrt(d_k) in attention?** A: To prevent dot products from growing too large, which would cause softmax to produce near-zero gradients.
- **Q: What is the purpose of multi-head attention?** A: Different heads learn different types of relationships (syntactic, semantic, positional).
- **Q: How does positional encoding work in transformers?** A: Sinusoidal functions of different frequencies are added to input embeddings to inject position information.

---

## 5. BERT and Fine-tuning

### What is it
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language model that learns contextual word representations. It is pre-trained on masked language modeling (predict masked words) and next sentence prediction, then fine-tuned on downstream tasks like classification, NER, and QA.

### Why it matters
BERT revolutionized NLP by providing a single pre-trained model that could be fine-tuned for dozens of tasks with minimal task-specific architecture. It eliminated the need to train models from scratch for each NLP task.

### Python code
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import torch

# Load pre-trained BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize input
text = "This movie was absolutely fantastic!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
print(f"Input IDs shape: {inputs['input_ids'].shape}")

# Forward pass (before fine-tuning)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    print(f"Prediction: {prediction.item()}")

# Fine-tuning uses HuggingFace Trainer:
# training_args = TrainingArguments(output_dir="./results", num_train_epochs=3,
#     per_device_train_batch_size=16, learning_rate=2e-5, weight_decay=0.01)
# trainer = Trainer(model=model, args=training_args,
#     train_dataset=train_dataset, eval_dataset=eval_dataset)
# trainer.train()
```

### Common mistakes
- Using too high a learning rate for fine-tuning (2e-5 to 5e-5 is typical, not 1e-3)
- Not using a warmup schedule (sudden large updates destabilize pre-trained weights)
- Fine-tuning on too little data without freezing lower layers

### Interview questions
- **Q: What is masked language modeling?** A: Randomly mask 15% of tokens and train the model to predict them, forcing bidirectional context understanding.
- **Q: What is the [CLS] token used for?** A: It is a special token whose final hidden state serves as the aggregate representation for classification tasks.
- **Q: When would you use BERT vs GPT?** A: BERT (encoder) for understanding tasks (classification, NER); GPT (decoder) for generation tasks (text generation, summarization).

---

## Week 6 Assignment

See [assignments/week-06-advanced-nlp/](../../assignments/week-06-advanced-nlp/) for the full assignment.
