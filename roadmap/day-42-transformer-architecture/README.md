# Day 42: Transformer Architecture

## Learning Objectives

- Understand the self-attention mechanism and how it computes query, key, and value matrices
- Explain multi-head attention and why multiple attention heads capture different relationships
- Describe positional encoding and why Transformers need explicit position information
- Trace the full encoder-decoder architecture from the "Attention Is All You Need" paper
- Implement a simplified Transformer block from scratch in PyTorch

## Key Concepts

The Transformer architecture, introduced by Vaswani et al. (2017), revolutionized NLP
by replacing sequential processing (RNNs/LSTMs) with a purely attention-based mechanism.
The core innovation is self-attention: for each token in a sequence, the model computes
attention weights over all other tokens, allowing it to capture long-range dependencies
in constant computational depth. Self-attention works by projecting inputs into query (Q),
key (K), and value (V) matrices, computing scaled dot-product attention as
softmax(QK^T / sqrt(d_k)) * V. This enables every position to attend to every other
position in a single step.

Multi-head attention extends this by running multiple attention operations in parallel,
each with different learned projections. This allows the model to jointly attend to
information from different representation subspaces -- one head might capture syntactic
relations while another captures semantic ones. Since attention is permutation-invariant
(it has no notion of order), positional encodings are added to the input embeddings.
The original paper used sinusoidal functions of different frequencies, though learned
positional embeddings are also common.

The full architecture follows an encoder-decoder structure. The encoder consists of
stacked layers of multi-head self-attention and position-wise feed-forward networks,
each wrapped with residual connections and layer normalization. The decoder is similar
but adds masked self-attention (preventing positions from attending to future tokens)
and cross-attention over the encoder output. Modern variants like BERT use only the
encoder, while GPT uses only the decoder. Understanding this architecture is essential
for working with any modern language model.

## Practical Example

```python
import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project and reshape to (batch, heads, seq_len, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(context)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# Demo
block = TransformerBlock()
dummy_input = torch.randn(2, 10, 512)  # (batch=2, seq_len=10, d_model=512)
output = block(dummy_input)
print(f"Input shape:  {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print(f"Parameters:   {sum(p.numel() for p in block.parameters()):,}")
```

## Resources

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Harvard NLP: The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

## Next Day Preview

Day 43 covers BERT fine-tuning -- how to take a pretrained Transformer encoder and adapt it to specific classification tasks using HuggingFace.
