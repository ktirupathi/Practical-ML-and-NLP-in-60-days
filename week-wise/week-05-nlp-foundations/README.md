# Week 5: NLP Foundations

*Days 29-35: Text preprocessing, representation, classification, and information extraction*

---

## 1. Text Preprocessing

### What is it
Text preprocessing is the pipeline of transforming raw text into a clean, normalized form suitable for machine learning. It includes tokenization (splitting text into words or subwords), lowercasing, removing stopwords, stemming (crude suffix stripping), lemmatization (dictionary-based root form), and regex-based pattern extraction. The quality of any downstream NLP model depends directly on how well the text has been preprocessed.

### Why it matters
Raw text contains noise such as HTML tags, special characters, inconsistent casing, and inflected word forms. Models that ingest unprocessed text waste capacity learning these irrelevancies. Proper preprocessing can improve a simple bag-of-words classifier by 10-20% in accuracy and dramatically reduce vocabulary size, lowering memory usage and training time.

### Math / key concepts
- **Tokenization**: Splitting "I can't go" into ["I", "ca", "n't", "go"] (word-level) or ["I", "can", "'", "t", "go"] (character-aware). Subword tokenizers (BPE, WordPiece) balance vocabulary size against coverage.
- **Lemmatization vs stemming**: Stemming applies rules (e.g., "running" -> "run" via suffix removal). Lemmatization uses a morphological dictionary ("better" -> "good"). Lemmatization is more accurate but slower.
- **Regex fundamentals**: `\b\w+\b` matches words, `\d{3}-\d{4}` matches phone fragments, `[A-Z][a-z]+` matches capitalized words. Lookaheads `(?=...)` and lookbehinds `(?<=...)` enable zero-width assertions.

### Python code
```python
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

text = "The cats were running quickly across 3 rooftops! Email: test@mail.com"

tokens = word_tokenize(text.lower())
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

cleaned = [lemmatizer.lemmatize(t) for t in tokens
           if t.isalpha() and t not in stop_words]
print("Cleaned:", cleaned)

emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
print("Emails found:", emails)
```

### Common mistakes
1. **Removing stopwords blindly** -- In sentiment analysis, words like "not" and "no" are critical. Always consider the task before stripping stopwords.
2. **Stemming for information retrieval** -- Aggressive stemming can merge unrelated words ("university" and "universe" both stem to "univers"). Lemmatization or no reduction is often safer.
3. **Ignoring tokenization edge cases** -- Contractions ("don't"), hyphenated words ("state-of-the-art"), and URLs break naive whitespace splitting. Use a proper tokenizer library.

### Interview questions

**Q1: What is the difference between stemming and lemmatization? When would you prefer each?**
A: Stemming applies heuristic suffix rules and is fast but imprecise ("studies" -> "studi"). Lemmatization uses a dictionary to return valid root forms ("studies" -> "study"). Use stemming for speed in large-scale search indexing. Use lemmatization when downstream tasks need valid words, such as text generation or knowledge extraction.

**Q2: How does subword tokenization (BPE) handle out-of-vocabulary words?**
A: BPE starts with a character-level vocabulary and iteratively merges the most frequent adjacent pairs. Unknown words are decomposed into known subword units. For example, "unhappiness" might become ["un", "happiness"] or ["un", "happ", "iness"]. This eliminates the OOV problem entirely while keeping vocabulary size manageable.

**Q3: Why is text preprocessing task-dependent?**
A: Different tasks rely on different textual signals. Sentiment analysis needs negation words and punctuation (exclamation marks). Named entity recognition needs capitalization preserved. Machine translation needs the full raw text. A universal preprocessing pipeline does not exist; each decision (lowercasing, stopword removal, lemmatization) should be validated against task performance.

---

## 2. Text Representation (BoW and TF-IDF)

### What is it
Text representation converts variable-length text into fixed-size numerical vectors that machine learning models can process. Bag-of-Words (BoW) counts word occurrences in each document, producing a sparse matrix of shape (n_documents, vocabulary_size). TF-IDF extends BoW by weighting terms based on how informative they are: frequent within a document but rare across the corpus.

### Why it matters
ML models cannot consume raw strings. The choice of representation determines what information the model can access. BoW and TF-IDF remain competitive baselines for many classification tasks, are fully interpretable, and train in seconds. They also serve as the conceptual foundation for understanding dense embeddings used in modern NLP.

### Math / key concepts
- **Term Frequency**: TF(t, d) = count(t in d) / total_terms_in_d
- **Inverse Document Frequency**: IDF(t) = log(N / df(t)), where N is total documents and df(t) is the number of documents containing term t.
- **TF-IDF**: TF-IDF(t, d) = TF(t, d) * IDF(t). High TF-IDF means the word is frequent in this document but rare globally, making it a strong discriminator.
- **BoW sparsity**: With a 50K vocabulary, each document vector has ~50K dimensions but only ~100 nonzero entries. Scipy sparse matrices store these efficiently.
- **N-grams**: Extending BoW to bigrams ("machine learning") or trigrams captures local word order at the cost of exploding vocabulary size.

### Python code
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends",
]

bow = CountVectorizer()
X_bow = bow.fit_transform(corpus)
print("BoW shape:", X_bow.shape)
print("Vocab:", bow.get_feature_names_out()[:10])

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50)
X_tfidf = tfidf.fit_transform(corpus)
print("TF-IDF shape:", X_tfidf.shape)
print("Top features:", tfidf.get_feature_names_out()[:10])
```

### Common mistakes
1. **Fitting TF-IDF on test data** -- The vectorizer must be fit only on training data. Calling fit_transform on the full dataset leaks information about test document frequencies.
2. **Ignoring max_features** -- Without limiting vocabulary, TF-IDF on large corpora creates matrices with hundreds of thousands of columns, causing memory issues and overfitting.
3. **Using raw counts for long documents** -- Longer documents naturally have higher word counts. TF normalization or using TF-IDF instead of raw BoW prevents length bias.

### Interview questions

**Q1: Explain the TF-IDF formula and give an intuition for each component.**
A: TF-IDF(t,d) = TF(t,d) * log(N/df(t)). TF captures local importance: how often term t appears in document d. IDF captures global rarity: log(N/df) is high for rare terms (appearing in few documents) and low for common terms like "the" that appear everywhere. The product rewards terms that are both locally frequent and globally distinctive.

**Q2: When would TF-IDF fail as a representation?**
A: TF-IDF fails when meaning depends on word order ("dog bites man" vs "man bites dog"), synonymy (different words with same meaning get different dimensions), and polysemy (same word with different meanings maps to one dimension). It also struggles with very short texts (tweets) where term frequencies are unreliable. Dense embeddings address these limitations.

**Q3: How do n-grams improve BoW, and what is the trade-off?**
A: Unigram BoW loses word order entirely. Adding bigrams ("not good") and trigrams captures local context and phrases, improving classification of negation and multi-word expressions. The trade-off is combinatorial vocabulary explosion: a 50K unigram vocabulary can become millions of bigrams, requiring aggressive feature selection (max_features, min_df) to remain tractable.

---

## 3. Text Classification (Naive Bayes and SVM)

### What is it
Text classification assigns predefined labels to documents. Multinomial Naive Bayes applies Bayes theorem with a naive conditional independence assumption: it estimates P(class|document) by computing P(word|class) for each word independently. Support Vector Machines (SVM) find a maximum-margin hyperplane in the high-dimensional TF-IDF feature space. Despite its simplicity, Naive Bayes is a surprisingly strong baseline; linear SVM often achieves state-of-the-art among non-neural methods.

### Why it matters
Text classification powers spam detection, sentiment analysis, intent recognition, content moderation, and document routing. In production, these simple models offer sub-millisecond inference, full interpretability, and no GPU requirements. They remain the recommended starting point before reaching for transformers.

### Math / key concepts
- **Bayes theorem for classification**: P(c|d) = P(d|c) * P(c) / P(d). Since P(d) is constant across classes, we compare P(d|c) * P(c) for each class c.
- **Naive assumption**: P(d|c) = product of P(w_i|c) for each word w_i in document d. Each word contributes independently.
- **Laplace smoothing**: P(w|c) = (count(w,c) + alpha) / (total_words_in_c + alpha * V) to avoid zero probabilities for unseen words.
- **SVM for text**: Linear kernel works well because TF-IDF features are already high-dimensional and approximately linearly separable. The regularization parameter C controls the bias-variance trade-off.

### Python code
```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

train = fetch_20newsgroups(subset="train", categories=["sci.space", "rec.sport.baseball"])
test = fetch_20newsgroups(subset="test", categories=["sci.space", "rec.sport.baseball"])

nb_pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=10000)), ("clf", MultinomialNB())])
nb_pipe.fit(train.data, train.target)
print("Naive Bayes:\n", classification_report(test.target, nb_pipe.predict(test.data)))

svm_pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=10000)), ("clf", LinearSVC())])
svm_pipe.fit(train.data, train.target)
print("Linear SVM:\n", classification_report(test.target, svm_pipe.predict(test.data)))
```

### Common mistakes
1. **Forgetting Laplace smoothing** -- Without smoothing (alpha=0), a single unseen word zeroes out the entire class probability. Always use alpha >= 1 (the sklearn default).
2. **Using Multinomial NB with negative features** -- TF-IDF values are non-negative and work fine, but if you standardize features (zero-mean), Multinomial NB breaks. Use GaussianNB or ComplementNB for normalized features.
3. **Not tuning SVM regularization** -- The default C=1 is often not optimal. Use cross-validated grid search over C in [0.01, 0.1, 1, 10] to find the best trade-off between margin width and training error.

### Interview questions

**Q1: Why does Naive Bayes work well for text despite the independence assumption being wrong?**
A: The independence assumption is violated (words are correlated), but Naive Bayes still produces good ranking of classes because classification only needs the correct argmax, not calibrated probabilities. The bias from the naive assumption is offset by low variance from having very few parameters to estimate, especially beneficial with small training sets.

**Q2: Why is a linear kernel preferred over RBF for text SVM?**
A: Text data in TF-IDF space is already very high-dimensional (often 10K-100K features) and sparse. In such spaces, data is often linearly separable or nearly so. RBF kernel would be computationally expensive (O(n^2) kernel matrix) without significant accuracy gain. Linear SVM also scales better and produces interpretable feature weights.

**Q3: How would you handle a text classification problem with 1000 classes?**
A: Use a one-vs-rest strategy (LinearSVC handles this natively). Consider hierarchical classification if classes have a taxonomy. Use TF-IDF with aggressive feature selection (max_features, chi2). Evaluate with macro and micro F1 to understand per-class vs overall performance. For very large label spaces, consider embedding-based approaches where labels and documents share a vector space.

---

## 4. Named Entity Recognition (NER)

### What is it
Named Entity Recognition identifies and classifies named entities in text into predefined categories such as person names, organizations, locations, dates, and monetary values. Modern NER systems use neural sequence labeling: each token receives a BIO tag (B-PER for beginning of a person name, I-PER for inside, O for outside any entity). SpaCy provides production-ready NER models that achieve near-human accuracy on standard benchmarks.

### Why it matters
NER is foundational for information extraction, knowledge graph construction, question answering, and document understanding. In production systems, NER extracts structured data from unstructured text: pulling company names from news articles, identifying drug names in clinical notes, or tagging locations in customer reviews for geographic analysis.

### Math / key concepts
- **BIO tagging scheme**: "Barack Obama visited Paris" -> [B-PER, I-PER, O, B-LOC]. B marks the beginning of an entity, I marks continuation, O marks non-entities.
- **Sequence labeling**: Unlike classification which assigns one label per document, NER assigns one label per token. CRF (Conditional Random Field) layers on top of neural encoders enforce valid tag transitions (I-PER cannot follow B-LOC).
- **Entity types**: Common categories include PER (person), ORG (organization), LOC (location), DATE, MONEY, GPE (geopolitical entity). Domain-specific NER adds types like DRUG, GENE, or LEGAL_CLAUSE.

### Python code
```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California on April 1, 1976."
doc = nlp(text)

for ent in doc.ents:
    print(f"{ent.text:25s} {ent.label_:10s} ({ent.start_char}-{ent.end_char})")

# Custom entity extraction with EntityRuler
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [{"label": "PRODUCT", "pattern": "MacBook Pro"},
            {"label": "PRODUCT", "pattern": [{"LOWER": "iphone"}, {"IS_DIGIT": True}]}]
ruler.add_patterns(patterns)

doc2 = nlp("The iPhone 15 and MacBook Pro were announced today.")
for ent in doc2.ents:
    print(f"{ent.text:25s} {ent.label_}")
```

### Common mistakes
1. **Evaluating NER with token-level accuracy** -- Since most tokens are O (outside), token accuracy is misleadingly high. Use entity-level precision, recall, and F1 with exact span matching instead.
2. **Ignoring entity boundary errors** -- A model that predicts "Barack" as PER but misses "Obama" is not half-correct; it extracted the wrong entity. Strict evaluation counts this as both a false positive and a false negative.
3. **Not fine-tuning for domain text** -- SpaCy's pretrained models are trained on news and web text. Medical, legal, or financial text has different entity types and writing styles. Fine-tuning on even 200-500 domain-annotated sentences significantly improves performance.

### Interview questions

**Q1: Explain the BIO tagging scheme. Why not just tag each word as PERSON or NOT?**
A: BIO distinguishes entity boundaries. In "New York and Los Angeles", a flat tag would produce [LOC, LOC, O, LOC, LOC], making it ambiguous whether "New York" is one entity or two. BIO produces [B-LOC, I-LOC, O, B-LOC, I-LOC], clearly delimiting two separate location entities. This is critical when consecutive entities of the same type appear.

**Q2: How would you build a custom NER system for a new domain?**
A: Start with a pretrained model (spaCy or BERT) and fine-tune on domain-annotated data. Annotate 500-2000 sentences using a tool like Prodigy or Label Studio with domain experts. Use active learning to prioritize ambiguous examples. Augment with gazetteers (known entity lists) via rule-based matching. Evaluate with entity-level F1 per type and iterate on low-performing categories.

**Q3: What is the role of a CRF layer in neural NER?**
A: A CRF layer models dependencies between adjacent tags. Without it, each token is labeled independently, allowing invalid sequences like I-PER following B-LOC. The CRF learns transition probabilities (B-PER -> I-PER is likely; I-PER -> I-LOC is unlikely) and uses the Viterbi algorithm at inference to find the globally optimal tag sequence, improving boundary detection.

---

## 5. Topic Modeling (LDA and BERTopic)

### What is it
Topic modeling discovers latent thematic structure in a collection of documents without labeled data. Latent Dirichlet Allocation (LDA) is a probabilistic generative model that represents each document as a mixture of topics and each topic as a distribution over words. BERTopic is a modern alternative that clusters document embeddings (from sentence-transformers) and extracts topic representations using c-TF-IDF, combining the power of pretrained language models with interpretable topic descriptions.

### Why it matters
Topic modeling enables exploratory analysis of large text corpora: understanding what customers discuss in reviews, tracking research trends across thousands of papers, organizing news articles, and discovering themes in social media. It provides unsupervised structure that would take human analysts weeks to identify manually.

### Math / key concepts
- **LDA generative process**: For each document, sample a topic distribution theta ~ Dir(alpha). For each word position, sample a topic z ~ Multinomial(theta), then sample a word w ~ Multinomial(beta_z). The plate notation shows documents contain words, each drawn from a latent topic.
- **Dirichlet prior**: Alpha controls topic sparsity. Low alpha (< 1) encourages documents to focus on few topics. High alpha encourages uniform topic mixtures.
- **c-TF-IDF (BERTopic)**: After clustering embeddings, c-TF-IDF computes importance of word t in cluster c as: tf(t,c) * log(1 + A/tf(t)). This extracts the most representative words for each cluster-topic.
- **Coherence score**: Measures topic quality by checking if top words co-occur in the corpus. Higher coherence (C_v metric) indicates more interpretable topics.

### Python code
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "machine learning algorithms improve predictions",
    "neural networks learn feature representations",
    "stock market trading strategies and risk",
    "portfolio optimization and asset allocation",
    "deep learning models for image recognition",
    "financial derivatives pricing and hedging",
]

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(docs)

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

feature_names = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-5:]]
    print(f"Topic {idx}: {', '.join(top_words)}")
```

### Common mistakes
1. **Choosing n_topics arbitrarily** -- Use coherence scores (C_v), perplexity on held-out data, or manual inspection across a range (5, 10, 15, 20, 30) to select the optimal number. There is no universal right answer.
2. **Feeding LDA raw text without preprocessing** -- LDA operates on word counts and is extremely sensitive to stopwords, rare words, and punctuation. Always preprocess (remove stopwords, lemmatize, filter by document frequency).
3. **Treating LDA topics as ground truth** -- LDA topics are statistical artifacts, not necessarily meaningful categories. Always validate topics with domain experts and compare multiple random seeds, as LDA is non-deterministic.

### Interview questions

**Q1: How does LDA differ from BERTopic in approach and when would you use each?**
A: LDA is a generative probabilistic model that operates on word counts (BoW). It assumes a Dirichlet prior over topic distributions and uses variational inference or Gibbs sampling. BERTopic embeds documents with sentence-transformers, clusters them with HDBSCAN, and extracts topics via c-TF-IDF. Use LDA for smaller corpora where interpretability and a probabilistic framework matter. Use BERTopic when you want semantic understanding, can leverage GPU embeddings, and have larger corpora where LDA's bag-of-words assumption is too limiting.

**Q2: What is the Dirichlet distribution and why is it used in LDA?**
A: The Dirichlet distribution is a distribution over probability distributions (a distribution over simplices). In LDA, it serves as a prior for topic mixtures (per-document) and word distributions (per-topic). A low alpha Dirichlet prior concentrates probability on sparse mixtures (documents about few topics), which matches real-world intuition. It is conjugate to the multinomial, making posterior inference tractable.

**Q3: How do you evaluate topic model quality when there are no ground-truth labels?**
A: Use intrinsic metrics: coherence scores (C_v measures pairwise word co-occurrence of top topic words; higher is better), perplexity on held-out documents (lower is better, but does not always correlate with human judgment). Use extrinsic evaluation: human judges rate topic interpretability (word intrusion test, topic intrusion test). Also check downstream utility: do the discovered topics improve a supervised classifier when used as features?
