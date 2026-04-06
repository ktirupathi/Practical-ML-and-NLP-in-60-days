# Week 5: NLP Foundations

## Topics Covered
Text Preprocessing, TF-IDF, Text Classification, Named Entity Recognition (NER), Topic Modeling

---

## Part 1: Conceptual Questions (10 Questions)

### Q1. Text Preprocessing Pipeline
**Difficulty:** Easy | **Estimated Time:** 5 minutes

Describe the standard text preprocessing pipeline for NLP tasks. What is the purpose of each step (tokenization, lowercasing, stop-word removal, stemming, lemmatization)? When might you choose to skip certain steps?

---

### Q2. Bag-of-Words vs. TF-IDF
**Difficulty:** Easy | **Estimated Time:** 5 minutes

Compare Bag-of-Words (BoW) and TF-IDF representations. Why does TF-IDF often outperform raw BoW for text classification? Write the TF-IDF formula and explain each component.

---

### Q3. Stemming vs. Lemmatization
**Difficulty:** Easy | **Estimated Time:** 5 minutes

What is the difference between stemming and lemmatization? Give an example where stemming produces an incorrect root form but lemmatization does not. When would you prefer one over the other?

---

### Q4. Named Entity Recognition Architecture
**Difficulty:** Medium | **Estimated Time:** 10 minutes

Explain the BIO (Beginning, Inside, Outside) tagging scheme used in NER. Why is this scheme preferred over simple entity/non-entity binary tagging? Provide an example sentence with BIO tags for at least two entity types.

---

### Q5. Topic Modeling with LDA
**Difficulty:** Medium | **Estimated Time:** 10 minutes

Explain the generative process assumed by Latent Dirichlet Allocation (LDA). What are the two Dirichlet distributions involved, and what do their hyperparameters (alpha and beta) control? How does changing alpha affect the topic distribution per document?

---

### Q6. Evaluation Metrics for Text Classification
**Difficulty:** Medium | **Estimated Time:** 10 minutes

For a multi-class text classification task with imbalanced classes, explain the difference between macro-averaged, micro-averaged, and weighted F1 scores. Which would you report if one class has 10x more samples than the others, and why?

---

### Q7. TF-IDF Limitations
**Difficulty:** Medium | **Estimated Time:** 8 minutes

List three fundamental limitations of TF-IDF as a text representation. For each limitation, briefly describe a more modern approach that addresses it.

---

### Q8. N-grams in Text Classification
**Difficulty:** Medium | **Estimated Time:** 8 minutes

What are n-grams and why are they useful in text classification? What are the trade-offs of using higher-order n-grams (e.g., trigrams vs. unigrams)? How does the vocabulary size grow with n-gram order?

---

### Q9. Topic Coherence
**Difficulty:** Hard | **Estimated Time:** 12 minutes

What is topic coherence and how does it differ from perplexity as a metric for evaluating topic models? Describe the C_v coherence measure. Why might a model with lower perplexity have less interpretable topics?

---

### Q10. CRF for Sequence Labeling
**Difficulty:** Hard | **Estimated Time:** 15 minutes

Explain how Conditional Random Fields (CRFs) improve upon independent classifiers for NER. What is the key advantage of modeling label dependencies? How does the Viterbi algorithm fit into CRF decoding at inference time?

---

## Part 2: Coding Questions (10 Questions)

### CQ1. Custom Tokenizer
**Difficulty:** Easy | **Estimated Time:** 10 minutes

Write a Python function that takes a raw text string and returns a list of cleaned tokens. The function should: (a) lowercase the text, (b) remove punctuation, (c) tokenize on whitespace, (d) remove stop words, (e) apply lemmatization using NLTK.

**Hint:** Use `nltk.corpus.stopwords`, `nltk.stem.WordNetLemmatizer`, and `string.punctuation`.

---

### CQ2. TF-IDF from Scratch
**Difficulty:** Medium | **Estimated Time:** 20 minutes

Implement TF-IDF computation from scratch (without sklearn). Given a list of documents (each a string), compute the TF-IDF matrix. Validate your result against `sklearn.feature_extraction.text.TfidfVectorizer`.

**Hint:** TF(t,d) = count(t in d) / len(d). IDF(t) = log(N / df(t)) where df(t) is the number of documents containing term t.

---

### CQ3. Text Classification Pipeline
**Difficulty:** Medium | **Estimated Time:** 15 minutes

Using sklearn, build a text classification pipeline that chains `TfidfVectorizer` with `LogisticRegression`. Train it on a subset of the 20 Newsgroups dataset (pick 4 categories) and print the classification report.

**Hint:** Use `sklearn.pipeline.Pipeline` and `sklearn.datasets.fetch_20newsgroups`.

---

### CQ4. Regex-based NER
**Difficulty:** Easy | **Estimated Time:** 10 minutes

Write a function that uses regular expressions to extract email addresses, phone numbers (US format), and URLs from a given text. Return a dictionary with keys `emails`, `phones`, and `urls`.

**Hint:** Use `re.findall()` with appropriate patterns.

---

### CQ5. SpaCy NER
**Difficulty:** Easy | **Estimated Time:** 10 minutes

Using spaCy, write a function that takes a text string and returns all named entities grouped by their entity type. For example: `{"PERSON": ["John Smith"], "ORG": ["Google"], ...}`.

**Hint:** Use `spacy.load("en_core_web_sm")` and iterate over `doc.ents`.

---

### CQ6. LDA with Gensim
**Difficulty:** Medium | **Estimated Time:** 20 minutes

Using Gensim, fit an LDA model on a list of documents. Write code that: (a) preprocesses the text, (b) creates a dictionary and corpus, (c) trains LDA with 5 topics, (d) prints the top 10 words per topic.

**Hint:** Use `gensim.corpora.Dictionary`, `gensim.models.LdaMulticore`.

---

### CQ7. Document Similarity with TF-IDF
**Difficulty:** Medium | **Estimated Time:** 15 minutes

Given a corpus of documents, write a function that takes a query string and returns the top-k most similar documents using TF-IDF vectors and cosine similarity.

**Hint:** Use `sklearn.metrics.pairwise.cosine_similarity` and transform the query using the same fitted vectorizer.

---

### CQ8. Text Augmentation
**Difficulty:** Medium | **Estimated Time:** 15 minutes

Implement three text augmentation techniques from scratch: (a) synonym replacement using WordNet, (b) random word deletion, (c) random word swap. Each function should take a sentence and return an augmented version.

**Hint:** Use `nltk.corpus.wordnet` for synonyms and `random` for sampling.

---

### CQ9. Confusion Matrix Visualization
**Difficulty:** Easy | **Estimated Time:** 10 minutes

Train a Naive Bayes classifier on the 20 Newsgroups dataset (pick 5 categories) and plot a confusion matrix heatmap using matplotlib/seaborn. Which categories are most often confused?

**Hint:** Use `sklearn.metrics.confusion_matrix` and `seaborn.heatmap`.

---

### CQ10. BERTopic Exploration
**Difficulty:** Hard | **Estimated Time:** 25 minutes

Use the BERTopic library to fit a topic model on a corpus of at least 500 documents. Visualize the topics using BERTopic's built-in visualization methods. Compare the topics qualitatively with those from an LDA model on the same data.

**Hint:** `from bertopic import BERTopic; model = BERTopic(); topics, probs = model.fit_transform(docs)`.

---

## Part 3: Mini Case Studies (2 Studies)

### Case Study 1: Spam Detection System

A telecom company wants to build a spam detection system for SMS messages. They have a labeled dataset of 5,574 messages (4,827 ham, 747 spam). The system must achieve at least 95% precision on the spam class to avoid blocking legitimate messages.

**Questions:**
1. The dataset is highly imbalanced (87% ham, 13% spam). What preprocessing and modeling strategies would you use to handle this imbalance?
2. Why is precision on the spam class more important than recall in this scenario? Under what business circumstances might recall become more important?
3. Design a complete pipeline from raw SMS text to prediction. Specify each step and your choice of algorithm at each stage.
4. The company wants to deploy this as a real-time filter. What latency constraints might you face, and how would they influence your model choice?

---

### Case Study 2: Customer Feedback Topic Analysis

An e-commerce platform receives 50,000 customer reviews per month. They want to automatically identify the main topics being discussed (product quality, shipping, customer service, pricing, etc.) and track how these topics trend over time.

**Questions:**
1. Compare LDA and BERTopic for this use case. Which would you recommend and why?
2. How would you determine the optimal number of topics? Describe at least two approaches.
3. The reviews contain mixed languages (primarily English with some Spanish and French). How would you handle this?
4. Design a monthly reporting pipeline that tracks topic trends and flags emerging issues. What metrics would you track?

---

## Part 4: Practical Assignments (3 Assignments)

### Assignment 1: Text Classification on 20 Newsgroups Dataset
**Estimated Time:** 2-3 hours

**Objective:** Build and compare multiple text classification models on the 20 Newsgroups dataset (18,846 documents across 20 categories).

**Tasks:**
1. Load the full 20 Newsgroups dataset using sklearn. Perform exploratory data analysis: class distribution, average document length, sample documents.
2. Implement a preprocessing pipeline: lowercase, remove headers/footers/quotes, remove punctuation, tokenize, remove stop words, lemmatize.
3. Train and evaluate the following models using TF-IDF features:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM (LinearSVC)
   - Random Forest
4. For each model, report accuracy, macro F1, and per-class F1. Use 5-fold cross-validation.
5. Perform hyperparameter tuning on the best model using GridSearchCV. Tune at least: TF-IDF max_features, ngram_range, and one model-specific parameter.
6. Analyze errors: plot a confusion matrix for the best model. Identify the most commonly confused category pairs and examine misclassified examples.
7. Write a summary comparing all models with a table of results.

**Deliverable:** A Jupyter notebook with code, results, and analysis.

---

### Assignment 2: Named Entity Recognition on CoNLL-2003 Dataset
**Estimated Time:** 2-3 hours

**Objective:** Build an NER system using the CoNLL-2003 dataset from HuggingFace, comparing a CRF-based approach with a transformer-based approach.

**Tasks:**
1. Load the CoNLL-2003 dataset from HuggingFace (`datasets.load_dataset("conll2003")`). Explore the dataset: entity type distribution, sentence lengths, sample annotations.
2. Implement a feature-based CRF model using `sklearn-crfsuite`:
   - Extract features: word shape, prefixes/suffixes, POS tags, capitalization, neighboring words.
   - Train the CRF model and evaluate with entity-level F1 using `seqeval`.
3. Fine-tune a pre-trained transformer model for NER:
   - Use `bert-base-cased` with HuggingFace Transformers.
   - Train for 3 epochs with appropriate learning rate and batch size.
   - Evaluate with entity-level precision, recall, and F1.
4. Compare both approaches: accuracy by entity type (PER, LOC, ORG, MISC), training time, inference speed.
5. Perform error analysis on the transformer model: what types of entities are hardest to recognize? Provide specific examples.

**Deliverable:** A Jupyter notebook with code, results, and comparative analysis.

---

### Assignment 3: Topic Modeling on BBC News Dataset
**Estimated Time:** 2-3 hours

**Objective:** Perform topic modeling on the BBC News dataset (2,225 articles across 5 categories: business, entertainment, politics, sport, tech) using both LDA and BERTopic.

**Tasks:**
1. Download and load the BBC News dataset. Perform EDA: category distribution, word clouds per category, document length statistics.
2. Preprocess the text for LDA: tokenize, remove stop words, lemmatize, build a Gensim dictionary and bag-of-words corpus.
3. Train LDA models with varying numbers of topics (3, 5, 7, 10, 15). For each, compute coherence score (C_v). Plot coherence vs. number of topics and select the optimal k.
4. For the best LDA model, visualize using pyLDAvis. Print top 15 words per topic and manually label each topic.
5. Train a BERTopic model on the same data. Compare the discovered topics with LDA topics.
6. Evaluate both models: (a) compare discovered topics to the known 5 categories, (b) compute topic diversity (percentage of unique words in the top 25 words across all topics).
7. Build a simple document-topic assignment function: given a new article, assign it to the most likely topic from each model.

**Deliverable:** A Jupyter notebook with code, visualizations, and comparative analysis.
