# Day 36: Text Summarization — Extractive and Abstractive

> **Phase 4 – NLP Foundations** | Week 6 | Estimated Time: 3-4 hours

## What You'll Learn
- Understand extractive summarization via TextRank algorithm
- Implement abstractive summarization using T5 and BART
- Evaluate summaries with ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- Build a document summarization pipeline

---

## 1. What Is Text Summarization?

Text summarization produces a shorter version of a document while preserving its key information and meaning.

Two main approaches:
- **Extractive summarization** – selects and concatenates the most important sentences from the source document verbatim.
- **Abstractive summarization** – generates new sentences that paraphrase and condense the source, like a human would write a summary.

A third hybrid approach combines both: extract key sentences, then paraphrase them.

---

## 2. Why Used?

| Use Case | Value |
|---|---|
| News aggregation | Summarize articles in 3 sentences |
| Legal document review | Condense 100-page contracts to key points |
| Scientific literature | Abstract generation from full papers |
| Customer support | Summarize long support threads |
| Meeting notes | Auto-generate meeting minutes |

The average knowledge worker processes 200+ documents per week. Automatic summarization reduces reading load by 80% while preserving 90%+ of key information.

---

## 3. Real-World Example

Google News uses extractive summarization to generate article previews. Reuters uses abstractive summarization (based on BART/T5) to produce first drafts of financial news articles from earnings reports. The Associated Press has used automated summarization since 2014 for quarterly earnings stories.

---

## 4. Intuition

**TextRank (extractive)**: Treat sentences as nodes in a graph. Add an edge between two sentences weighted by their similarity (cosine similarity of TF-IDF vectors). Apply PageRank to rank sentences by their "importance" (how many other important sentences they are similar to). Select the top-k sentences.

**T5/BART (abstractive)**: A sequence-to-sequence transformer trained on large paired (document, summary) datasets. It learns to generate summaries that are faithful to the source while being concise and fluent. T5 treats summarization as a text-to-text task with the prefix "summarize:".

---

## 5. Mathematical Intuition

```
=== TextRank (Extractive) ===

Similarity between sentences s_i and s_j:
  sim(s_i, s_j) = |{w : w ∈ s_i ∧ w ∈ s_j}| / (log|s_i| + log|s_j|)
  
  OR: cosine similarity of TF-IDF vectors

Graph G = (V, E) where:
  V = sentences
  E = edges weighted by similarity

PageRank update rule (iterative):
  WS(v_i) = (1 - d) + d * Σ_{j→i} [w_ji / Σ_{k, j→k} w_jk] * WS(v_j)
  d = damping factor (typically 0.85)
  
Convergence after ~30 iterations; top-k sentences by WS score form the summary.

=== ROUGE Evaluation ===
ROUGE-N (n-gram overlap):
  Recall    = (# n-gram matches) / (# n-grams in reference)
  Precision = (# n-gram matches) / (# n-grams in candidate)
  F1        = 2 * R * P / (R + P)

ROUGE-1: unigram overlap (content coverage)
ROUGE-2: bigram overlap (fluency/phrase-level)
ROUGE-L: Longest Common Subsequence (order-aware coverage)

Typical scores for good news summarization:
  ROUGE-1 F1 ≈ 0.40-0.45
  ROUGE-2 F1 ≈ 0.18-0.22
  ROUGE-L F1 ≈ 0.35-0.40
```

---

## 6. Worked Example

```
Source: "The Federal Reserve raised interest rates by 0.25 percentage points on Wednesday,
         the tenth increase since March 2022. Fed Chair Powell said inflation remains too
         high despite recent progress. Markets fell after the announcement."

Extractive summary (TextRank top-1 sentence):
  "Fed Chair Powell said inflation remains too high despite recent progress."
  → Selects most central sentence in similarity graph

Abstractive summary (T5):
  "The Federal Reserve hiked rates for the tenth time, with Chair Powell warning
   that inflation remains elevated."
  → New sentence, paraphrased, more concise
```

---

## 7. Python Implementation

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import re
import warnings
warnings.filterwarnings("ignore")

# ── TextRank Extractive Summarizer ─────────────────────────────────────────
def sentence_tokenize(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.split()) > 5]

def textrank_summarize(text: str, n_sentences: int = 3) -> str:
    sentences = sentence_tokenize(text)
    if len(sentences) <= n_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Build graph and apply PageRank
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph, alpha=0.85, max_iter=100)

    # Select top n sentences in original order
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_idx = sorted([idx for idx, _ in ranked[:n_sentences]])
    return " ".join(sentences[i] for i in top_idx)

# ── ROUGE Evaluation ───────────────────────────────────────────────────────
def compute_rouge_n(candidate: str, reference: str, n: int = 1) -> dict:
    def get_ngrams(text, n):
        tokens = text.lower().split()
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    cand_ngrams = get_ngrams(candidate, n)
    ref_ngrams  = get_ngrams(reference, n)

    cand_set = {}
    for ng in cand_ngrams:
        cand_set[ng] = cand_set.get(ng, 0) + 1

    ref_set = {}
    for ng in ref_ngrams:
        ref_set[ng] = ref_set.get(ng, 0) + 1

    matches = sum(min(cand_set.get(ng, 0), ref_set[ng]) for ng in ref_set)
    precision = matches / len(cand_ngrams) if cand_ngrams else 0
    recall    = matches / len(ref_ngrams)  if ref_ngrams  else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

# ── Abstractive with Transformers (T5) ─────────────────────────────────────
def abstractive_summarize(text: str, max_length: int = 130, min_length: int = 30) -> str:
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn",
                              device=-1)  # CPU
        result = summarizer(text, max_length=max_length, min_length=min_length,
                            do_sample=False)
        return result[0]["summary_text"]
    except ImportError:
        return "[transformers not installed]"
    except Exception as e:
        return f"[Error: {e}]"

# ── Demo ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    article = """
    The Federal Reserve raised its benchmark interest rate by a quarter of a percentage 
    point on Wednesday, pushing borrowing costs to their highest level in 22 years. 
    The decision, which was unanimous, brings the federal funds rate to a range of 
    5.25 to 5.5 percent. Fed Chair Jerome Powell said at a press conference that 
    inflation remains too high, despite recent signs of progress. He noted that the 
    labor market is still strong, with unemployment near historic lows at 3.6 percent. 
    Markets reacted negatively, with the S&P 500 falling 0.7 percent and the Nasdaq 
    declining 0.8 percent. Bond yields rose, with the 10-year Treasury yield climbing 
    to 3.97 percent. Economists are divided on whether this will be the final rate hike 
    of the current tightening cycle. Some analysts expect rates to remain elevated 
    through 2024, while others anticipate cuts beginning early next year.
    """

    print("=== Original Article ===")
    print(f"Word count: {len(article.split())}")
    print()

    print("=== TextRank Extractive Summary (3 sentences) ===")
    extractive = textrank_summarize(article, n_sentences=3)
    print(extractive)
    print(f"Word count: {len(extractive.split())}")

    reference = "The Federal Reserve raised rates to a 22-year high. Chair Powell warned inflation is still too high despite progress. Markets fell on the news."

    print("\n=== ROUGE Scores (Extractive vs Reference) ===")
    for n in [1, 2]:
        scores = compute_rouge_n(extractive, reference, n)
        print(f"ROUGE-{n}: P={scores['precision']:.3f}  R={scores['recall']:.3f}  F1={scores['f1']:.3f}")

    print("\n=== Abstractive Summary (BART) ===")
    abstractive = abstractive_summarize(article.strip())
    print(abstractive)
    if abstractive and not abstractive.startswith("["):
        abs_scores = compute_rouge_n(abstractive, reference, 1)
        print(f"\nROUGE-1: P={abs_scores['precision']:.3f}  R={abs_scores['recall']:.3f}  F1={abs_scores['f1']:.3f}")
```

---

## 8. Visualization

```
TextRank Graph (sentence nodes, similarity edges):

  s1 "Fed raised rates"
    ╲ 0.7
     s3 "Powell: inflation too high" ←──── Most central (highest PageRank)
    ╱ 0.5        ╲ 0.3
  s2 "Markets fell"   s4 "Labor market strong"

ROUGE-1 Recall calculation:
  Reference:  "Federal Reserve raised rates inflation"
  Candidate:  "Federal Reserve hiked rates"
              matched: {Federal, Reserve, rates} = 3 tokens
  Recall = 3/5 = 0.60

Summarization Performance (CNN/DailyMail benchmark):
  Method          ROUGE-1  ROUGE-2  ROUGE-L
  Lead-3 baseline   40.3    17.7     36.7
  TextRank          40.0    17.6     36.5
  BART (abstractive) 44.2   21.3     40.9
  GPT-4 (0-shot)    43.8    21.1     40.7
```

---

## 9. Common Mistakes

1. **Using ROUGE as the only evaluation metric** – ROUGE measures lexical overlap, not semantic quality. A paraphrase scores poorly even if it's a better summary.
2. **Applying abstractive summarization to very short texts** – Models like BART/T5 work best on documents >200 words.
3. **Ignoring max_length constraints** – Transformers have 512-1024 token limits. Chunk long documents and merge summaries.
4. **TextRank with duplicate sentences** – Pre-deduplicate sentences before building the similarity graph.
5. **Forgetting stemming/normalization for ROUGE** – ROUGE is case-sensitive by default; normalize before computing.
6. **Using GPU models in production without memory management** – BART is ~1.6GB; manage model loading carefully.

---

## 10. Interview Questions

| # | Question | Answer |
|---|---|---|
| 1 | What is the difference between extractive and abstractive summarization? | Extractive selects existing sentences verbatim; abstractive generates new sentences that paraphrase the source. |
| 2 | How does TextRank work? | Builds a sentence similarity graph, applies PageRank to score sentence centrality, selects top-k highest-scoring sentences. |
| 3 | What are ROUGE metrics? | Recall-Oriented Understudy for Gisting Evaluation. ROUGE-N measures n-gram overlap between candidate and reference summaries. |
| 4 | What is ROUGE-L? | Uses Longest Common Subsequence (LCS) instead of n-gram overlap; captures sentence-level structure without requiring contiguous matches. |
| 5 | Why is ROUGE insufficient for evaluating abstractive summaries? | Abstractive summaries use different words than the reference, even when semantically equivalent. ROUGE penalizes valid paraphrases. |
| 6 | What is BERTScore? | An evaluation metric using contextual BERT embeddings to measure semantic similarity between candidate and reference, handling paraphrases better than ROUGE. |
| 7 | What is T5 and how does it do summarization? | T5 (Text-to-Text Transfer Transformer) treats all NLP tasks as text-to-text. Summarization: input "summarize: {document}", output the summary. |
| 8 | How do you handle documents longer than BART's 1024 token limit? | Chunk the document into overlapping segments, summarize each, then summarize the concatenated summaries (hierarchical summarization). |
| 9 | What training data does BART use for summarization? | Pre-trained on text denoising, fine-tuned on CNN/DailyMail and XSum datasets of article-summary pairs. |
| 10 | When would you choose extractive over abstractive? | Extractive: when faithfulness is critical (legal, medical), when compute is limited, or when exact quotes are needed. |

---

## Exercises

1. Implement TextRank using cosine similarity of sentence embeddings (SBERT) instead of TF-IDF.
2. Compare ROUGE scores of TextRank vs BART vs Lead-3 baseline on CNN/DailyMail dataset.
3. Build a pipeline to summarize PDFs by extracting text, chunking, and summarizing.
4. Evaluate summaries with BERTScore alongside ROUGE; observe differences for paraphrases.
5. Fine-tune a T5-small model on a domain-specific summarization dataset using Hugging Face Trainer.

---

## Key Takeaways

- Extractive summarization (TextRank) selects key sentences using graph-based PageRank; no training needed.
- Abstractive summarization (BART/T5) generates new text; requires a fine-tuned seq2seq model.
- ROUGE measures n-gram/LCS overlap between candidate and reference summaries; use ROUGE-1, ROUGE-2, ROUGE-L.
- ROUGE undervalues semantically correct paraphrases; supplement with BERTScore or human evaluation.
- Long documents exceed transformer context limits; use hierarchical summarization with chunking.
- Lead-3 baseline (first 3 sentences) is surprisingly strong for news articles—always compare against it.
