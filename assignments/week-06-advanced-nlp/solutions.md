# Week 6: Advanced NLP - Solutions

---

## Part 1: Conceptual Answers

### A1. Word2Vec Architectures
**Skip-gram** predicts context words given a target word. Given "learning", it predicts "machine", "is", etc. **CBOW** predicts the target word given surrounding context words. Skip-gram performs better on rare words because each occurrence of a rare word creates multiple training examples (one per context word). CBOW averages context vectors, which dilutes the signal for rare words. Skip-gram is slower to train but generally produces higher-quality embeddings for infrequent terms.

### A2. Word Embedding Properties
Word embeddings map words to dense vectors where geometric relationships encode semantic meaning. The analogy `king - man + woman = queen` works because the vector offset between "king" and "man" captures the concept of "royalty," and adding that offset to "woman" lands near "queen." This is a linear vector operation: `vec("king") - vec("man") + vec("woman") ≈ vec("queen")`. The embedding space organizes such that parallel relationships (gender, tense, plurality) correspond to consistent directional offsets.

### A3. Limitations of Static Embeddings
The word "bank" has multiple meanings: financial institution vs. river bank. Word2Vec assigns one vector to "bank" that is a weighted average of all contexts seen during training. This conflation hurts downstream tasks. Contextual embeddings like BERT produce different representations for "bank" in "I deposited money at the bank" versus "We sat on the river bank" because they process the full sentence through deep layers, making each token's representation context-dependent.

### A4. The Attention Mechanism
`Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`. **Q** (Query) represents what we are looking for. **K** (Key) represents what each position offers. **V** (Value) is the actual content to aggregate. The dot product `QK^T` computes alignment scores. We scale by `sqrt(d_k)` because for large `d_k`, dot products grow large in magnitude, pushing softmax into regions with extremely small gradients (near-zero for most positions, near-one for the max). Scaling keeps the variance manageable and gradients flowing.

### A5. Multi-Head Attention
Multi-head attention allows the model to attend to information from different representation subspaces at different positions. A single head can only focus on one type of relationship. With 8 heads and `d_model=512`, each head operates on `d_k = 512/8 = 64` dimensions. Different heads learn to attend to syntactic dependencies, semantic similarities, positional patterns, etc. The outputs are concatenated and linearly projected back to `d_model`.

### A6. Transformer Positional Encoding
Sinusoidal positional encoding: `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`, `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`. Each dimension uses a different frequency. Sine and cosine are used together so that relative positions can be represented as linear functions: `PE(pos+k)` can be expressed as a linear transformation of `PE(pos)`. Lower-frequency dimensions capture long-range position differences; higher-frequency dimensions capture fine-grained positions. This enables the model to generalize to sequence lengths not seen during training.

### A7. BERT Pre-training Objectives
**MLM:** Randomly masks 15% of tokens (80% replaced with [MASK], 10% random token, 10% unchanged) and trains the model to predict them. This enables bidirectional context -- unlike left-to-right LMs that can only see preceding tokens, BERT sees both left and right context simultaneously. **NSP:** Given sentence pairs, predicts whether sentence B actually follows sentence A. This helps with tasks requiring understanding of sentence relationships (QA, NLI). Later work (RoBERTa) showed NSP may not be necessary.

### A8. BERT vs. GPT Architecture
BERT uses the Transformer encoder with full (bidirectional) self-attention -- every token attends to every other token. This is ideal for understanding tasks (classification, NER, QA) because the representation is enriched by complete context. GPT uses the Transformer decoder with masked (causal) self-attention -- each token can only attend to previous tokens. This autoregressive property is essential for generation: predicting the next token sequentially. BERT cannot generate text naturally because it was not trained autoregressively. GPT underperforms on classification because it only has left context during pre-training.

### A9. BLEU and ROUGE Metrics
**BLEU** measures precision: what fraction of n-grams in the candidate appear in the reference. BLEU-4 uses n-grams up to 4. The brevity penalty `BP = exp(1 - r/c)` (when `c < r`) penalizes short candidates. **ROUGE-1** measures unigram recall (what fraction of reference unigrams appear in the candidate). **ROUGE-2** uses bigrams. **ROUGE-L** uses longest common subsequence. BLEU suits translation (precision matters -- the output should be fluent). ROUGE suits summarization (recall matters -- the summary should cover key content).

### A10. Transfer Learning in NLP
Pre-training learns general language representations (syntax, semantics, world knowledge) from vast unlabeled data. Fine-tuning adapts these representations to a specific task with limited labeled data. This is effective because: (1) Language structure is largely task-agnostic, (2) Pre-training captures knowledge that would require millions of labeled examples to learn from scratch. Catastrophic forgetting occurs when fine-tuning overwrites pre-trained knowledge. Mitigation strategies: use a small learning rate (2e-5), apply gradual unfreezing (fine-tune top layers first, then progressively unfreeze lower layers), use discriminative learning rates (lower LR for earlier layers), and employ early stopping.

---

## Part 2: Coding Solutions

### CQ1. Train Word2Vec with Gensim

```python
from gensim.models import Word2Vec

sentences = [
    ["machine", "learning", "is", "a", "subset", "of", "artificial", "intelligence"],
    ["deep", "learning", "uses", "neural", "networks", "for", "pattern", "recognition"],
    ["natural", "language", "processing", "deals", "with", "text", "data"],
    ["word", "embeddings", "capture", "semantic", "meaning", "of", "words"],
    ["neural", "networks", "learn", "representations", "from", "data"],
    ["supervised", "learning", "requires", "labeled", "training", "data"],
    ["unsupervised", "learning", "finds", "patterns", "without", "labels"],
    ["reinforcement", "learning", "uses", "rewards", "to", "train", "agents"],
    ["convolutional", "networks", "excel", "at", "image", "recognition"],
    ["recurrent", "networks", "process", "sequential", "data", "like", "text"],
    ["transformers", "use", "attention", "mechanisms", "for", "sequence", "modeling"],
    ["bert", "is", "a", "bidirectional", "transformer", "for", "language", "understanding"],
    ["gpt", "generates", "text", "using", "autoregressive", "language", "modeling"],
    ["transfer", "learning", "leverages", "pretrained", "models", "for", "new", "tasks"],
    ["text", "classification", "assigns", "labels", "to", "documents"],
    ["sentiment", "analysis", "detects", "positive", "or", "negative", "opinions"],
    ["named", "entity", "recognition", "extracts", "entities", "from", "text"],
    ["machine", "translation", "converts", "text", "between", "languages"],
    ["question", "answering", "systems", "find", "answers", "in", "text"],
    ["artificial", "intelligence", "aims", "to", "create", "intelligent", "machines"],
    ["data", "science", "combines", "statistics", "and", "machine", "learning"],
    ["feature", "engineering", "creates", "informative", "input", "representations"],
]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, epochs=100)

print("Most similar to 'machine':")
print(model.wv.most_similar("machine", topn=5))

# Vector arithmetic
result = model.wv.most_similar(positive=["machine", "processing"], negative=["learning"], topn=3)
print("\nmachine - learning + processing =")
print(result)

# Access raw vector
vector = model.wv["machine"]
print(f"\nVector shape: {vector.shape}")
```

---

### CQ2. Compute Scaled Dot-Product Attention

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    # Softmax along the last axis
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    output = weights @ V
    return output, weights

Q = np.random.randn(4, 8)
K = np.random.randn(6, 8)
V = np.random.randn(6, 16)
output, weights = scaled_dot_product_attention(Q, K, V)

print(f"Output shape: {output.shape}")       # (4, 16)
print(f"Weights shape: {weights.shape}")      # (4, 6)
print(f"Weights row sums: {weights.sum(axis=-1)}")  # All ~1.0
```

---

### CQ3. Tokenization with HuggingFace

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = "The Transformer architecture revolutionized natural language processing."

encoding = tokenizer(text, return_tensors="pt")
print("Input IDs:", encoding["input_ids"])
print("Attention Mask:", encoding["attention_mask"])
print("Token Type IDs:", encoding["token_type_ids"])

decoded = tokenizer.decode(encoding["input_ids"][0])
print("Decoded:", decoded)

tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)
# [CLS] marks the start and produces a sentence-level representation.
# [SEP] marks the boundary between segments (sentence pairs).
```

---

### CQ4. Sentence Embeddings with BERT

```python
from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden = outputs.last_hidden_state              # (1, seq_len, 768)
    mask = inputs["attention_mask"].unsqueeze(-1)    # (1, seq_len, 1)
    summed = (hidden * mask).sum(dim=1)
    counted = mask.sum(dim=1)
    embedding = summed / counted                     # Mean pooling
    return embedding.squeeze(0)

emb1 = get_sentence_embedding("I love machine learning")
emb2 = get_sentence_embedding("AI is fascinating")

cosine_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0))
print(f"Cosine similarity: {cosine_sim.item():.4f}")
```

---

### CQ5. Fine-tune BERT for Text Classification

```python
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

tokenized = dataset.map(tokenize_fn, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir="./bert-imdb",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate()
print(results)
```

---

### CQ6. Compute BLEU Score

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [["the", "cat", "sat", "on", "the", "mat"]]
candidate = ["the", "cat", "is", "on", "the", "mat"]
smoother = SmoothingFunction().method1

bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
bleu_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25),
                         smoothing_function=smoother)

print(f"BLEU-1: {bleu_1:.4f}")  # Unigram precision
print(f"BLEU-2: {bleu_2:.4f}")  # Up to bigram
print(f"BLEU-3: {bleu_3:.4f}")  # Up to trigram
print(f"BLEU-4: {bleu_4:.4f}")  # Up to 4-gram (with smoothing)
# Short candidates get penalized by the brevity penalty.
```

---

### CQ7. Compute ROUGE Score

```python
from rouge_score import rouge_scorer

reference = "The quick brown fox jumps over the lazy dog near the river bank."
generated = "A fast brown fox leaps over a lazy dog by the river."

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference, generated)

for metric, score in scores.items():
    print(f"{metric}: Precision={score.precision:.4f}, Recall={score.recall:.4f}, F1={score.fmeasure:.4f}")
# ROUGE-1: unigram overlap. ROUGE-2: bigram overlap. ROUGE-L: longest common subsequence.
```

---

### CQ8. Word2Vec Visualization with t-SNE

```python
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Assume `model` is a trained Word2Vec model
# Select words from semantic categories
categories = {
    "animals": ["cat", "dog", "horse", "fish", "bird", "lion", "tiger", "bear", "wolf", "deer"],
    "countries": ["france", "germany", "japan", "china", "india", "brazil", "canada", "italy", "spain", "russia"],
    "colors": ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "pink", "brown"],
    "professions": ["doctor", "teacher", "engineer", "lawyer", "scientist", "artist", "writer", "nurse", "pilot", "chef"],
    "sports": ["football", "basketball", "tennis", "baseball", "soccer", "cricket", "hockey", "golf", "boxing", "swimming"],
}

words, vectors, labels = [], [], []
colors_map = {"animals": "red", "countries": "blue", "colors": "green", "professions": "orange", "sports": "purple"}

for category, word_list in categories.items():
    for word in word_list:
        if word in model.wv:
            words.append(word)
            vectors.append(model.wv[word])
            labels.append(category)

vectors = np.array(vectors)
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
coords = tsne.fit_transform(vectors)

plt.figure(figsize=(14, 10))
for category in categories:
    mask = [l == category for l in labels]
    idx = np.where(mask)[0]
    plt.scatter(coords[idx, 0], coords[idx, 1], label=category, c=colors_map[category], alpha=0.7, s=60)
    for i in idx:
        plt.annotate(words[i], (coords[i, 0] + 0.5, coords[i, 1] + 0.5), fontsize=8)

plt.legend(fontsize=12)
plt.title("Word2Vec Embeddings Visualized with t-SNE")
plt.tight_layout()
plt.savefig("word2vec_tsne.png", dpi=150)
plt.show()
```

---

### CQ9. Build an Extractive Summarizer

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

def extractive_summarize(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Score each sentence by its mean TF-IDF value
    scores = np.array(tfidf_matrix.mean(axis=1)).flatten()

    # Get indices of top-scoring sentences
    top_indices = np.argsort(scores)[-num_sentences:]
    top_indices = sorted(top_indices)  # Preserve original order

    summary = " ".join([sentences[i] for i in top_indices])
    return summary

article = """
Artificial intelligence has transformed many industries. Machine learning models now power
recommendation systems, self-driving cars, and medical diagnostics. Deep learning, a subset
of machine learning, uses neural networks with many layers. These networks can learn complex
patterns from large datasets. Natural language processing is one of the most exciting areas
of AI. It enables machines to understand and generate human language. Recent advances like
BERT and GPT have pushed the boundaries of what NLP systems can achieve. Transfer learning
has made it possible to build powerful models with limited labeled data. The future of AI
promises even more remarkable capabilities.
"""
print(extractive_summarize(article, num_sentences=3))
```

---

### CQ10. Multi-Head Attention in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)

mha = MultiHeadAttention(512, 8)
x = torch.randn(2, 10, 512)
output = mha(x, x, x)
assert output.shape == (2, 10, 512)
print(f"Output shape: {output.shape}")
```
