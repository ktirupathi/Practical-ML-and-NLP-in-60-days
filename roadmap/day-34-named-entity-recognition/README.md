# Day 34: Named Entity Recognition (NER)

## Learning Objectives

- Use spaCy's pretrained NER pipeline to extract entities from text
- Train a custom NER model to recognize domain-specific entity types
- Apply rule-based matching with spaCy's `EntityRuler` and `Matcher` for deterministic extraction
- Evaluate NER performance with precision, recall, and F1 at the entity level
- Combine rule-based and statistical approaches for robust entity extraction

## Key Concepts

### What Is Named Entity Recognition

Named Entity Recognition is the task of locating and classifying named entities in text into
predefined categories such as PERSON, ORG, GPE (geo-political entity), DATE, and MONEY.
spaCy ships with pretrained models (`en_core_web_sm`, `en_core_web_trf`) that handle these
standard categories out of the box. NER is a foundational NLP component that feeds into
higher-level tasks like relation extraction, knowledge graph construction, and information
retrieval.

### Custom Entity Training

When your domain uses entity types not covered by standard models --- for example, DRUG_NAME
in biomedical text or PRODUCT_CODE in e-commerce --- you can train a custom NER model. spaCy
3+ uses a config-driven training system where you define the pipeline components, training
data (in `.spacy` binary format), and hyperparameters in a `config.cfg` file. Training from a
pretrained model (transfer learning) requires far less labeled data than training from scratch.

### Rule-Based Matching

For entities that follow strict patterns (e.g., invoice numbers like "INV-2024-0001" or
phone numbers), statistical models are overkill. spaCy's `EntityRuler` lets you define
pattern-based rules that run alongside or instead of the statistical NER component. The
`Matcher` and `PhraseMatcher` provide even more flexible token-level pattern matching. A
common production strategy is to layer rules on top of the statistical model, letting rules
handle high-precision patterns and the model catch everything else.

## Practical Example

```python
import spacy
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
from spacy.tokens import DocBin
from spacy.training import Example

# --- 1. Pretrained NER ---
nlp = spacy.load("en_core_web_sm")
text = "Apple Inc. hired Tim Cook in 2011. The headquarters are in Cupertino, California."
doc = nlp(text)

print("Entities found:")
for ent in doc.ents:
    print(f"  {ent.text:25s} {ent.label_:10s} ({ent.start_char}-{ent.end_char})")

# --- 2. Rule-based entity matching ---
nlp_rules = English()
ruler = nlp_rules.add_pipe("entity_ruler")

patterns = [
    {"label": "PRODUCT", "pattern": "iPhone"},
    {"label": "PRODUCT", "pattern": [{"LOWER": "macbook"}, {"LOWER": "pro"}]},
    {"label": "TICKET_ID", "pattern": [{"TEXT": {"REGEX": r"TICK-\d{4,}"}}]},
]
ruler.add_patterns(patterns)

doc2 = nlp_rules("Please check TICK-4523 about the MacBook Pro and iPhone issues.")
for ent in doc2.ents:
    print(f"  {ent.text:25s} {ent.label_}")

# --- 3. Preparing training data for custom NER ---
TRAIN_DATA = [
    ("Aspirin 500mg was prescribed.", {"entities": [(0, 7, "DRUG")]}),
    ("Patient takes Metformin daily.", {"entities": [(14, 23, "DRUG")]}),
    ("Ibuprofen reduced the inflammation.", {"entities": [(0, 9, "DRUG")]}),
]

# Convert to spaCy format
nlp_blank = spacy.blank("en")
doc_bin = DocBin()
for text, annotations in TRAIN_DATA:
    doc = nlp_blank.make_doc(text)
    example = Example.from_dict(doc, annotations)
    doc_bin.add(example.reference)
doc_bin.to_disk("train.spacy")

# --- 4. Evaluate NER predictions ---
from sklearn.metrics import classification_report

def evaluate_ner(nlp_model, test_examples):
    """Simple token-level NER evaluation."""
    true_labels, pred_labels = [], []
    for text, annotations in test_examples:
        doc = nlp_model(text)
        # Build gold entity spans
        gold_spans = {(s, e): lab for s, e, lab in annotations["entities"]}
        pred_spans = {(ent.start_char, ent.end_char): ent.label_ for ent in doc.ents}

        all_spans = set(gold_spans.keys()) | set(pred_spans.keys())
        for span in all_spans:
            true_labels.append(gold_spans.get(span, "O"))
            pred_labels.append(pred_spans.get(span, "O"))

    print(classification_report(true_labels, pred_labels))

evaluate_ner(nlp, [
    ("Apple was founded by Steve Jobs.", {"entities": [(0, 5, "ORG"), (21, 31, "PERSON")]}),
])
```

## Resources

- [spaCy NER documentation](https://spacy.io/usage/linguistic-features#named-entities)
- [spaCy training guide](https://spacy.io/usage/training)
- [Rule-based matching in spaCy](https://spacy.io/usage/rule-based-matching)

## Up Next

**Day 35 -- Topic Modeling:** Discover hidden themes in document collections using LDA, NMF, and BERTopic.
