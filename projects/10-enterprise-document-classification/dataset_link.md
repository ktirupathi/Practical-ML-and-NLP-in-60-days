# Dataset: RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing)

## Source

- **HuggingFace**: <https://huggingface.co/datasets/rvl_cdip>
- **Original Paper**: Harley, A.W., Ufkes, A., Derpanis, K.G., "Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval," ICDAR 2015.

## Overview

| Property | Value |
|---|---|
| Total Images | 400,000 |
| Training Set | 320,000 |
| Validation Set | 40,000 |
| Test Set | 40,000 |
| Image Format | Grayscale, variable resolution |
| Number of Classes | 16 |
| Balanced | Yes (25,000 images per class) |

## 16 Document Classes

| Label ID | Class Name |
|---|---|
| 0 | letter |
| 1 | form |
| 2 | email |
| 3 | handwritten |
| 4 | advertisement |
| 5 | scientific_report |
| 6 | scientific_publication |
| 7 | specification |
| 8 | file_folder |
| 9 | news_article |
| 10 | budget |
| 11 | invoice |
| 12 | presentation |
| 13 | questionnaire |
| 14 | resume |
| 15 | memo |

## Schema

```python
{
    "image": Image,       # PIL Image object (grayscale document scan)
    "label": ClassLabel,  # Integer 0-15 mapping to one of 16 classes
}
```

## Preprocessing Steps

1. **Load** via HuggingFace `datasets` library: `load_dataset("rvl_cdip")`
2. **Validate** images for corruption and readability
3. **Resize** to 224x224 (model input resolution)
4. **Convert** grayscale to RGB (3-channel) for compatibility with pretrained vision models
5. **Normalize** pixel values using ImageNet statistics or model-specific processor
6. **Transform** to PyTorch tensors via `AutoImageProcessor` from the chosen model checkpoint

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("rvl_cdip")
# dataset["train"], dataset["validation"], dataset["test"]
```
