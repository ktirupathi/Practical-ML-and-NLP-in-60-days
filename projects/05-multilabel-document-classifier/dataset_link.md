# EUR-Lex (EURLEX57K) Dataset

## Source
- **Name:** EURLEX57K -- Large-Scale Multi-Label Text Classification Dataset
- **Link:** http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/
- **Paper:** "Large-Scale Multi-Label Text Classification on EU Legislation" (Chalkidis et al., ACL 2019)
- **Format:** JSON files (one per document) grouped into train/dev/test directories
- **Size:** 57,000 EU law documents with EUROVOC concept labels

## Description
EURLEX57K contains 57,000 legislative documents from the EUR-Lex portal of European Union law.
Each document is annotated with multiple EUROVOC concept labels (descriptors) from a controlled
vocabulary of approximately 4,271 unique labels. This makes it an ideal benchmark for multi-label
text classification, where each document can belong to many categories simultaneously.

Documents include the title, header (recitals), and main body of EU legislation. The average
document has approximately 5 EUROVOC labels assigned to it, though some may have over 15.

## Schema
Each JSON document contains:

| Field       | Type         | Description                                           |
|-------------|--------------|-------------------------------------------------------|
| celex_id    | string       | Unique identifier for the EU legal document           |
| title       | string       | Title of the legislative document                     |
| header      | string       | Recitals / introductory text of the document          |
| recitals    | string       | Recital paragraphs (may overlap with header)          |
| main_body   | string       | Full body text of the legislation                     |
| attachments | string       | Any attached annexes or schedules                     |
| concepts    | list[string] | EUROVOC concept labels assigned to the document       |

## Multi-Label Structure
Unlike single-label classification where each sample has exactly one label, multi-label
classification allows each document to have **zero or more** labels simultaneously:

- **Average labels per document:** ~5
- **Max labels per document:** ~20+
- **Total unique labels:** ~4,271 (full EUROVOC vocabulary)
- **Label distribution:** Highly imbalanced -- some labels appear in thousands of documents,
  others in fewer than 10

### Label Frequency Tiers
The EURLEX57K dataset organizes labels into frequency tiers:
- **Frequent labels:** Appear in more than 50 documents (~746 labels)
- **Few-shot labels:** Appear in 10--50 documents (~1,169 labels)
- **Zero-shot labels:** Appear in fewer than 10 training documents (~2,356 labels)

## Preprocessing Steps
1. **Document parsing** -- Load JSON files from train/dev/test directories
2. **Text concatenation** -- Combine title + header + recitals into a single text field
   (main_body is often too long; title + header provides a good signal-to-length ratio)
3. **Text cleaning** -- Lowercase, remove special characters, collapse whitespace
4. **Stopword removal** -- Remove common English stopwords using NLTK
5. **Lemmatization** -- Reduce words to base form using WordNet lemmatizer
6. **Label filtering** -- Keep only frequent labels (appearing in >= 50 documents) to make
   the problem tractable with classical ML. This reduces the label space to ~746 labels
7. **MultiLabelBinarizer** -- Convert list of label strings to binary indicator matrix
8. **TF-IDF Vectorization** -- Convert cleaned text to sparse numerical features
   (max 50,000 features, unigrams + bigrams)
9. **Train/Dev/Test Split** -- Use the official splits provided in the dataset
