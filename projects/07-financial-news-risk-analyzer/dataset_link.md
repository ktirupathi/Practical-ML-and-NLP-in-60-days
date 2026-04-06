# Financial PhraseBank Dataset

## Source
- **HuggingFace**: https://huggingface.co/datasets/financial_phrasebank
- **Original Paper**: Malo et al. (2014) - "Good debt or bad debt: Detecting semantic orientations in economic texts"

## Overview
4,846 English financial news sentences manually annotated by 5-8 domain experts
(finance and business professionals) for sentiment polarity.

## Schema
| Field      | Type   | Description                                      |
|------------|--------|--------------------------------------------------|
| sentence   | string | Financial news sentence                          |
| label      | int    | Sentiment label: 0 = negative, 1 = neutral, 2 = positive |

## Annotator Agreement Levels
The dataset provides multiple subsets based on inter-annotator agreement:

| Subset                    | Description                              | Approx Size |
|---------------------------|------------------------------------------|-------------|
| `sentences_allagree`      | 100% annotator agreement                 | ~2,264      |
| `sentences_75agree`       | >= 75% annotator agreement               | ~3,453      |
| `sentences_66agree`       | >= 66% annotator agreement               | ~4,217      |
| `sentences_50agree`       | >= 50% annotator agreement (all data)    | ~4,846      |

Higher agreement levels yield cleaner labels but fewer samples.

## Label Distribution (sentences_allagree)
- **Neutral**: ~59%
- **Positive**: ~28%
- **Negative**: ~13%

The dataset is imbalanced, with neutral sentiment dominating.

## Preprocessing Notes
1. **Text cleaning**: Minimal cleaning needed; sentences are already well-formed
2. **Tokenization**: Use BERT/FinBERT tokenizer (WordPiece) with max_length=128
3. **Label encoding**: Labels are already integer-encoded (0, 1, 2)
4. **Train/Val/Test split**: Dataset ships as a single split; we create 70/15/15 splits
5. **Agreement level selection**: Default to `sentences_allagree` for highest quality;
   use `sentences_75agree` for more data with slight noise

## Supplementary Data
Reuters financial news articles can be used for:
- Additional unlabeled data for domain adaptation
- Testing generalization of trained models on unseen financial text
