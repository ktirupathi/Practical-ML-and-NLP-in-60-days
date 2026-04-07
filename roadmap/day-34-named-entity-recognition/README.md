# Day 34: Named Entity Recognition (NER)

> **Phase 4 – NLP Foundations** | Week 5 | Estimated Time: 3-4 hours

## What You'll Learn
- Understand NER as a sequence labeling task with IOB tagging
- Use spaCy's pre-trained NER models for entity extraction
- Train a custom NER model with spaCy and EntityRuler
- Evaluate NER systems with entity-level F1

---

## 1. What Is Named Entity Recognition?

Named Entity Recognition (NER) is the task of locating and classifying named entities in text into predefined categories such as:
- **PERSON** – "Elon Musk", "Marie Curie"
- **ORG** – "Apple Inc.", "United Nations"
- **GPE** (Geo-Political Entity) – "France", "New York"
- **DATE** – "January 2024", "last Tuesday"
- **MONEY** – "$500 million", "€1.2B"
- **PRODUCT** – "iPhone 15", "Tesla Model S"

NER is a foundational component in information extraction pipelines, knowledge graph construction, and question answering systems.

---

## 2. Why Used?

| Use Case | NER Value |
|---|---|
| News analysis | Extract organizations, people, locations mentioned |
| Financial analysis | Extract companies, amounts, dates from filings |
| Healthcare | Extract drug names, diseases, procedures |
| Legal documents | Extract parties, dates, jurisdictions |
| Search engines | Improve query understanding |

Without NER, text is just a bag of tokens. With NER, you know *who* did *what* to *whom*, *where*, and *when*.

---

## 3. Real-World Example

Bloomberg Terminal processes millions of news articles daily. Its NER system identifies company names (ORG), stock tickers, financial amounts (MONEY), and dates (DATE) to automatically populate financial databases, trigger trading alerts, and connect news events to portfolio positions—all without human review.

---

## 4. Intuition

NER is a **sequence labeling** problem: given a sequence of tokens, predict a label for each token. The labels follow the IOB (Inside-Outside-Beginning) format:

```
"Apple  Inc  hired  Tim  Cook  in  2011"
  B-ORG  I-ORG   O    B-PER I-PER  O   B-DATE
```

Modern NER models (like spaCy's) use neural networks (CNN/transformer) to capture context—the word "Apple" is a company when followed by "Inc" or "announced", but a fruit when followed by "pie" or "tree".

---

## 5. Mathematical Intuition

```
=== IOB Tagging Scheme ===
B-TYPE : Beginning token of an entity of TYPE
I-TYPE : Inside (continuation) token of an entity of TYPE
O      : Outside — not part of any entity

Alternative: BIOES (B, I, O, E=End, S=Single)

=== Conditional Random Field (CRF) for NER ===
Score(y | x) = Σ_t [emission_score(y_t, x_t) + transition_score(y_{t-1}, y_t)]

Emission score: how likely is label y_t given token x_t
Transition score: how likely is y_t given previous label y_{t-1}
  (e.g., I-PER cannot follow B-ORG — captured by transition matrix)

=== Entity-Level F1 ===
True Positive:  entity span and type both match
False Positive: predicted entity doesn't match gold
False Negative: gold entity not predicted

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * P * R / (P + R)

Partial match scoring: B-only match, type mismatch — both modes used in evaluation
```

---

## 6. Worked Example

```python
import spacy
nlp = spacy.load("en_core_web_sm")

text = "Elon Musk founded SpaceX in 2002 in El Segundo, California."
doc = nlp(text)
for ent in doc.ents:
    print(f"{ent.text:<20} {ent.label_:<10} {spacy.explain(ent.label_)}")

# Output:
# Elon Musk            PERSON     People, including fictional
# SpaceX               ORG        Companies, agencies, institutions
# 2002                 DATE       Absolute or relative dates
# El Segundo           GPE        Countries, cities, states
# California           GPE        Countries, cities, states
```

---

## 7. Python Implementation

```python
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy import displacy

# ── 1. spaCy Pre-trained NER ───────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")

def extract_entities(texts: list[str]) -> list[list[tuple]]:
    """Extract named entities from a list of texts."""
    results = []
    for doc in nlp.pipe(texts, batch_size=32):
        ents = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
        results.append(ents)
    return results

sample_texts = [
    "Apple Inc. CEO Tim Cook announced the new iPhone 15 in San Francisco on September 12, 2023.",
    "The Federal Reserve raised interest rates by 25 basis points, affecting Wall Street traders.",
    "Amazon acquired Whole Foods for $13.7 billion in 2017.",
]

print("=== Pre-trained NER ===")
for text, ents in zip(sample_texts, extract_entities(sample_texts)):
    print(f"\nText: {text}")
    for entity, label, start, end in ents:
        print(f"  [{start:3d}:{end:3d}] {label:<12} '{entity}'")

# ── 2. EntityRuler for Custom Entities ────────────────────────────────────
def add_entity_ruler(nlp_model):
    """Add custom entity patterns using EntityRuler."""
    ruler = nlp_model.add_pipe("entity_ruler", before="ner")
    patterns = [
        {"label": "PRODUCT",  "pattern": "GPT-4"},
        {"label": "PRODUCT",  "pattern": "GPT-3"},
        {"label": "PRODUCT",  "pattern": [{"LOWER": "chatgpt"}]},
        {"label": "ORG",      "pattern": [{"LOWER": "openai"}]},
        {"label": "TECH",     "pattern": [{"LOWER": "large"}, {"LOWER": "language"}, {"LOWER": "model"}]},
        {"label": "TECH",     "pattern": "LLM"},
    ]
    ruler.add_patterns(patterns)
    return nlp_model

nlp_custom = spacy.load("en_core_web_sm")
nlp_custom = add_entity_ruler(nlp_custom)

custom_texts = [
    "OpenAI released GPT-4 in March 2023, surpassing ChatGPT in capabilities.",
    "The LLM field is dominated by large language model research from various labs.",
]
print("\n=== Custom EntityRuler ===")
for text in custom_texts:
    doc = nlp_custom(text)
    print(f"\nText: {text}")
    for ent in doc.ents:
        print(f"  {ent.label_:<12} '{ent.text}'")

# ── 3. NER Evaluation ──────────────────────────────────────────────────────
def evaluate_ner(nlp_model, examples: list[tuple]):
    """Compute entity-level precision, recall, F1."""
    tp = fp = fn = 0
    for text, gold_ents in examples:
        doc = nlp_model(text)
        pred_set = {(e.text, e.label_) for e in doc.ents}
        gold_set = set(gold_ents)
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

gold_data = [
    ("Apple Inc. hired Tim Cook.", [("Apple Inc.", "ORG"), ("Tim Cook", "PERSON")]),
    ("Amazon acquired Whole Foods in 2017.", [("Amazon", "ORG"), ("Whole Foods", "ORG"), ("2017", "DATE")]),
]
metrics = evaluate_ner(nlp, gold_data)
print(f"\n=== NER Evaluation ===")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall   : {metrics['recall']:.3f}")
print(f"F1       : {metrics['f1']:.3f}")

# ── 4. Visualize NER ───────────────────────────────────────────────────────
doc = nlp(sample_texts[0])
# displacy.serve(doc, style="ent")  # uncomment to view in browser
html = displacy.render(doc, style="ent", page=False)
with open("ner_visualization.html", "w") as f:
    f.write(html)
print("\nNER visualization saved to ner_visualization.html")
```

---

## 8. Visualization

```
IOB Tagging of a sentence:

Token:   "Elon"  "Musk"  "founded"  "SpaceX"  "in"  "2002"
Label:   B-PER   I-PER      O        B-ORG      O    B-DATE

Entity spans:
  [0:9]   B-PER + I-PER → PERSON: "Elon Musk"
  [18:24] B-ORG         → ORG:    "SpaceX"
  [28:32] B-DATE        → DATE:   "2002"

spaCy NER Pipeline:
  tok2vec (embeddings) → ner (transition-based parser with CRF-like scoring)

EntityRuler priority:
  Patterns matched BEFORE neural NER (when added "before='ner'")
  This ensures custom terms override pretrained model
```

---

## 9. Common Mistakes

1. **Using a small model for complex entities** – `en_core_web_sm` struggles with unusual entity types. Use `en_core_web_trf` for better accuracy.
2. **Not handling overlapping entities** – spaCy entities cannot overlap; plan your annotation scheme carefully.
3. **Ignoring entity boundaries in evaluation** – A partial span match ("New York" predicted as "New York City") counts as wrong in strict evaluation.
4. **Adding EntityRuler after NER** – Rules added `after='ner'` override neural predictions; adding `before='ner'` lets rules take priority.
5. **Training with too few examples** – NER requires at minimum 100-200 examples per entity type for reasonable performance.
6. **Forgetting to disable unused components** – When running only NER, disable parser/tagger with `nlp.select_pipes(enable=["ner"])` for speed.

---

## 10. Interview Questions

| # | Question | Answer |
|---|---|---|
| 1 | What is IOB tagging? | B=Beginning, I=Inside, O=Outside. Labels each token with its entity role. B-TYPE starts an entity, I-TYPE continues it, O is non-entity. |
| 2 | What types of entities does spaCy's en_core_web_sm recognize? | PERSON, ORG, GPE, LOC, DATE, TIME, MONEY, PERCENT, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, and more. |
| 3 | How do you add custom entity types with spaCy? | Use EntityRuler with pattern dictionaries, or train a custom NER component with annotated examples. |
| 4 | What is the difference between EntityRuler and custom NER training? | EntityRuler uses rule-based pattern matching (fast, deterministic). Custom NER training teaches the neural model to generalize from annotated examples. |
| 5 | How is NER evaluated? | Entity-level F1: a prediction is TP only if both the span boundary and entity type exactly match the gold annotation. |
| 6 | What is the CRF layer in NER? | Conditional Random Field models dependencies between consecutive token labels (e.g., I-PER cannot follow B-ORG). |
| 7 | How do you handle ambiguous entities like "Apple"? | Contextual embeddings (BERT-based models) resolve ambiguity using surrounding words. "Apple earnings" → ORG; "apple pie" → not an entity. |
| 8 | How do you speed up spaCy NER for large corpora? | Use `nlp.pipe(texts, batch_size=64)` for batch processing. Disable unused pipeline components. |
| 9 | What training format does spaCy v3 use? | spaCy v3 uses `.spacy` binary format (DocBin). Training is configured via `config.cfg` files. |
| 10 | When should you use a rule-based EntityRuler vs. a trained model? | Rules for known, fixed patterns (product codes, company names); trained model for generalization over unseen text with similar structure. |

---

## Exercises

1. Extract all organizations and people from 100 news articles and build a co-occurrence graph.
2. Add custom entities for a specific domain (medical drugs, legal terms) using EntityRuler.
3. Compare NER accuracy of `en_core_web_sm` vs `en_core_web_trf` on the CoNLL-2003 test set.
4. Build a pipeline that extracts (Subject, Verb, Object) triples using NER + dependency parsing.
5. Implement a simple entity linking system that maps extracted entities to Wikipedia URLs.

---

## Key Takeaways

- NER is a sequence labeling task using IOB tagging to mark entity boundaries and types.
- spaCy provides production-ready pre-trained NER models; use `nlp.pipe()` for batch efficiency.
- EntityRuler enables deterministic rule-based entity extraction alongside neural NER.
- Evaluate NER with entity-level F1 (both span and type must match).
- Contextual embeddings (transformers) greatly improve NER accuracy on ambiguous entities.
- For custom domains, annotate 100-200 examples per entity type and fine-tune or use rules.
