# Resume Dataset

## Source
- **Name:** Resume Dataset
- **Link:** https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
- **Format:** CSV
- **Size:** 2,484 resumes

## Description
A collection of 2,484 resumes scraped and categorized into 25 professional categories.
Each resume contains raw text (often with HTML artifacts) and an associated job category label.
This dataset is suitable for multi-class text classification tasks in NLP.

## Schema
| Column   | Type   | Description                              |
|----------|--------|------------------------------------------|
| Category | string | Job category label (e.g., Data Science)  |
| Resume   | string | Raw resume text (may contain HTML tags)  |

## Categories (25)
1. Advocate
2. Arts
3. Automation Testing
4. Blockchain
5. Business Analyst
6. Civil Engineer
7. Data Science
8. Database
9. DevOps Engineer
10. DotNet Developer
11. ETL Developer
12. Electrical Engineering
13. HR
14. Hadoop
15. Health and Fitness
16. Java Developer
17. Mechanical Engineer
18. Network Security Engineer
19. Operations Manager
20. PMO
21. Python Developer
22. SAP Developer
23. Sales
24. Testing
25. Web Designing

## Preprocessing Steps
1. **HTML tag removal** — Strip all HTML/XML tags from resume text
2. **URL removal** — Remove hyperlinks
3. **Special character removal** — Remove non-alphanumeric characters (keep spaces)
4. **Lowercasing** — Convert all text to lowercase
5. **Tokenization** — Split text into individual tokens
6. **Stopword removal** — Remove common English stopwords
7. **Lemmatization** — Reduce words to their base/dictionary form
8. **TF-IDF Vectorization** — Convert cleaned text to numerical feature vectors (max 5000 features)
9. **Label Encoding** — Encode category labels as integers
