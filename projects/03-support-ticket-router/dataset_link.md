# Bitext Customer Support LLM Chatbot Training Dataset

## Source
- **Name:** Bitext Customer Support LLM Chatbot Training Dataset
- **Link:** https://huggingface.co/datasets/Bitext/Bitext-customer-support-llm-chatbot-training-dataset
- **Format:** Parquet (HuggingFace Datasets)
- **Size:** 26,872 rows

## Description
A high-quality dataset designed for training customer support chatbots and LLM fine-tuning.
Each row contains a customer support query (instruction), the corresponding intent label,
a broader category grouping, and a reference response. The dataset covers 27 distinct intents
across 11 service categories, making it ideal for multi-class text classification of support tickets.

## Schema
| Column      | Type   | Description                                          |
|-------------|--------|------------------------------------------------------|
| flags       | int    | Internal flags (not used for classification)         |
| instruction | string | Customer support query / ticket text                 |
| category    | string | High-level service category (e.g., ORDER, ACCOUNT)  |
| intent      | string | Fine-grained intent label (e.g., cancel_order)       |
| response    | string | Reference response for the query                     |

## Intent Categories (27)
1. cancel_order
2. change_order
3. change_shipping_address
4. check_cancellation_fee
5. check_invoices
6. check_payment_methods
7. check_refund_policy
8. complaint
9. contact_customer_service
10. contact_human_agent
11. create_account
12. delete_account
13. delivery_options
14. delivery_period
15. edit_account
16. get_invoice
17. get_refund
18. newsletter_subscription
19. payment_issue
20. place_order
21. recover_password
22. registration_problems
23. review
24. set_up_shipping_address
25. switch_account
26. track_order
27. track_refund

## High-Level Categories (11)
1. ORDER
2. ACCOUNT
3. SHIPPING
4. INVOICE
5. PAYMENT
6. FEEDBACK
7. CONTACT
8. CANCELLATION_FEE
9. REFUND
10. DELIVERY
11. SUBSCRIPTION

## Preprocessing Steps
1. **Text cleaning** -- Lowercase all instruction text
2. **Punctuation normalization** -- Remove excess punctuation and special characters
3. **Whitespace normalization** -- Collapse multiple spaces into single spaces
4. **Stopword removal** -- Remove common English stopwords using NLTK
5. **Lemmatization** -- Reduce words to base form using WordNet lemmatizer
6. **TF-IDF Vectorization** -- Convert cleaned text to numerical features (max 10,000 features, bigrams)
7. **Label Encoding** -- Encode intent labels as integer classes
8. **Train/Validation/Test Split** -- 70/15/15 stratified split on intent labels
