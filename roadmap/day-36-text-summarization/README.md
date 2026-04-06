# Day 36: Text Summarization

## Learning Objectives

- Understand the difference between extractive and abstractive summarization
- Implement extractive summarization using TextRank and sentence scoring
- Use pretrained transformer models (BART, T5) for abstractive summarization
- Evaluate summary quality with ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- Build a summarization pipeline that handles long documents via chunking

## Key Concepts

### Extractive vs. Abstractive Summarization

**Extractive summarization** selects the most important sentences from the original text and
concatenates them into a summary. It never generates new words --- the output is a subset of
the input. TextRank, a graph-based algorithm inspired by PageRank, is the classic approach:
it builds a similarity graph over sentences and ranks them by centrality. Extractive methods
are fast, faithful to the source, and easy to implement, but they can produce choppy summaries
that lack coherence.

**Abstractive summarization** generates new text that paraphrases and condenses the source. It
requires language generation capabilities, which is why it was impractical before modern
transformers. Models like BART and T5, fine-tuned on summarization datasets (CNN/DailyMail,
XSum), can produce fluent, human-like summaries. The trade-off is higher computational cost
and the risk of hallucination --- generating facts not present in the source.

### Evaluation with ROUGE

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is the standard automatic metric
for summarization. **ROUGE-1** measures unigram overlap between the generated summary and a
reference summary, **ROUGE-2** measures bigram overlap, and **ROUGE-L** measures the longest
common subsequence. Each variant reports precision, recall, and F1. ROUGE correlates
reasonably with human judgments but is not perfect --- two summaries with identical ROUGE
scores can differ significantly in readability and factual accuracy.

## Practical Example

```python
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge_score import rouge_scorer

nltk.download("punkt_tab", quiet=True)

# --- Sample article ---
article = """
Artificial intelligence is transforming healthcare in remarkable ways. Researchers have
developed AI systems that can detect diseases from medical images with accuracy rivaling
human specialists. Machine learning algorithms analyze patient records to predict health
risks before symptoms appear. Natural language processing helps extract key information
from clinical notes, saving doctors hours of paperwork. However, challenges remain in
ensuring these systems are fair, transparent, and trustworthy. Bias in training data can
lead to disparities in care across demographic groups. Regulatory frameworks are still
catching up with the pace of innovation. Despite these hurdles, the potential of AI to
improve patient outcomes and reduce costs makes it one of the most promising areas of
modern medicine.
"""

# --- 1. Extractive summarization with TextRank ---
def textrank_summarize(text, num_sentences=3):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text.strip()

    # Build TF-IDF vectors for sentences
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Compute sentence similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)

    # Score sentences by sum of similarities (simplified TextRank)
    scores = sim_matrix.sum(axis=1)
    ranked_indices = scores.argsort()[::-1][:num_sentences]
    ranked_indices = sorted(ranked_indices)  # preserve original order

    summary = " ".join([sentences[i].strip() for i in ranked_indices])
    return summary

extractive_summary = textrank_summarize(article, num_sentences=3)
print("=== Extractive Summary ===")
print(extractive_summary)

# --- 2. Abstractive summarization with a pretrained model ---
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

abstractive_summary = summarizer(
    article,
    max_length=80,
    min_length=30,
    do_sample=False,
)[0]["summary_text"]

print("\n=== Abstractive Summary ===")
print(abstractive_summary)

# --- 3. Evaluate with ROUGE ---
reference = (
    "AI is transforming healthcare by detecting diseases and predicting health risks. "
    "Challenges include bias and regulation. AI has great potential to improve outcomes."
)

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

for name, summary in [("Extractive", extractive_summary), ("Abstractive", abstractive_summary)]:
    scores = scorer.score(reference, summary)
    print(f"\n{name} ROUGE scores:")
    for metric, values in scores.items():
        print(f"  {metric}: P={values.precision:.3f} R={values.recall:.3f} F1={values.fmeasure:.3f}")
```

## Resources

- [TextRank paper (Mihalcea and Tarau)](https://aclanthology.org/W04-3252/)
- [Hugging Face summarization guide](https://huggingface.co/docs/transformers/tasks/summarization)
- [rouge-score Python package](https://pypi.org/project/rouge-score/)

## Up Next

**Day 37 -- Project: Email Intent Detection:** Build a system that classifies emails by intent to power smart inbox routing.
