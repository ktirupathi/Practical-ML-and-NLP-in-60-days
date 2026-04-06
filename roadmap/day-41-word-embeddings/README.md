# Day 41: Word Embeddings

## Learning Objectives

- Understand the intuition behind dense word representations and why they outperform one-hot encodings
- Compare and contrast Word2Vec (CBOW and Skip-gram), GloVe, and FastText architectures
- Load and use pretrained word embeddings for downstream NLP tasks
- Evaluate word embeddings using similarity, analogy, and clustering tasks
- Integrate pretrained embeddings into a PyTorch or Keras model

## Key Concepts

Word embeddings are dense, low-dimensional vector representations of words that capture
semantic and syntactic relationships. Unlike sparse one-hot vectors, embeddings place
semantically similar words close together in continuous vector space. The breakthrough
came with Word2Vec (Mikolov et al., 2013), which introduced two training objectives:
Continuous Bag of Words (CBOW), which predicts a target word from its surrounding
context, and Skip-gram, which predicts context words given a target word. Skip-gram
tends to work better on smaller datasets and for rare words, while CBOW is faster to
train.

GloVe (Global Vectors) takes a different approach by factorizing a global word
co-occurrence matrix, combining the benefits of count-based and prediction-based
methods. FastText, developed by Facebook Research, extends Word2Vec by representing
each word as a bag of character n-grams, which allows it to generate embeddings for
out-of-vocabulary words and handle morphologically rich languages more effectively.

In practice, most projects start with pretrained embeddings (trained on billions of
tokens) and either use them as fixed features or fine-tune them during training.
Popular pretrained options include Google News Word2Vec (300d), GloVe trained on
Common Crawl (300d), and FastText embeddings available for 157 languages. The choice
depends on your domain, vocabulary coverage, and whether you need subword handling.

## Practical Example

```python
import gensim.downloader as api
import numpy as np

# Load pretrained Word2Vec embeddings (Google News, 300d)
model = api.load("word2vec-google-news-300")

# Word similarity
similarity = model.similarity("king", "queen")
print(f"Similarity(king, queen): {similarity:.4f}")

# Classic analogy: king - man + woman = queen
result = model.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
print("king - man + woman =", result)

# Find words that don't belong
odd_one = model.doesnt_match(["breakfast", "lunch", "dinner", "python"])
print(f"Odd one out: {odd_one}")

# Using pretrained embeddings in PyTorch
import torch
import torch.nn as nn

vocab = ["king", "queen", "man", "woman", "child"]
embedding_dim = 300
embedding_matrix = np.zeros((len(vocab), embedding_dim))

for i, word in enumerate(vocab):
    if word in model:
        embedding_matrix[i] = model[word]

embedding_layer = nn.Embedding.from_pretrained(
    torch.FloatTensor(embedding_matrix), freeze=False
)
print(f"Embedding layer shape: {embedding_layer.weight.shape}")

# Compute cosine similarity in PyTorch
vec_king = embedding_layer(torch.tensor(0))
vec_queen = embedding_layer(torch.tensor(1))
cos_sim = torch.nn.functional.cosine_similarity(vec_king.unsqueeze(0), vec_queen.unsqueeze(0))
print(f"PyTorch cosine similarity (king, queen): {cos_sim.item():.4f}")
```

## Resources

- [Word2Vec paper: Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Gensim Word2Vec tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html)

## Next Day Preview

Day 42 dives into the Transformer architecture -- the foundation of modern NLP that replaced recurrence with self-attention.
