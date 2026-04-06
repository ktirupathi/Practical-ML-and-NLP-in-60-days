# Day 43: BERT Fine-Tuning

## Learning Objectives

- Understand BERT's architecture: masked language modeling, next sentence prediction, and bidirectional context
- Use HuggingFace Transformers to load pretrained BERT models and tokenizers
- Fine-tune BERT for text classification with proper learning rate scheduling
- Handle tokenization details including special tokens, padding, truncation, and attention masks
- Evaluate fine-tuned models and understand when BERT is the right choice

## Key Concepts

BERT (Bidirectional Encoder Representations from Transformers) was a landmark model
that demonstrated the power of pretraining a deep bidirectional Transformer on large
unlabeled text corpora. Unlike GPT which reads text left-to-right, BERT uses masked
language modeling (randomly masking 15% of tokens and predicting them) to learn
representations that incorporate both left and right context simultaneously. This
bidirectional understanding makes BERT particularly effective for tasks that require
full sentence comprehension, such as classification, named entity recognition, and
question answering.

Fine-tuning BERT involves adding a task-specific head (typically a linear layer) on top
of the pretrained model and training the entire network end-to-end on labeled data for
your specific task. The HuggingFace Transformers library makes this straightforward,
providing pretrained checkpoints, tokenizers that handle the necessary preprocessing
(adding [CLS] and [SEP] tokens, creating attention masks), and a Trainer API that
manages the training loop. A key detail is using a small learning rate (2e-5 to 5e-5)
with warm-up to avoid catastrophic forgetting of the pretrained weights.

In practice, BERT-base (110M parameters) works well for most classification tasks
with modest compute. For resource-constrained settings, DistilBERT provides 97% of
BERT's performance at 60% of the size. The [CLS] token's final hidden state serves
as the aggregate sequence representation used for classification, though pooling
strategies over all token outputs can sometimes improve results.

## Practical Example

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load dataset and tokenizer
dataset = load_dataset("imdb", split={"train": "train[:2000]", "test": "test[:500]"})
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)

tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load pretrained BERT with a classification head
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./bert-imdb",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)
trainer.train()

# Evaluate
results = trainer.evaluate()
print(f"Eval accuracy: {results.get('eval_loss', 'N/A')}")

# Inference on new text
inputs = tokenizer("This movie was absolutely fantastic!", return_tensors="pt", truncation=True)
with torch.no_grad():
    logits = model(**inputs).logits
prediction = torch.argmax(logits, dim=-1).item()
print(f"Prediction: {'positive' if prediction == 1 else 'negative'}")
```

## Resources

- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2019)](https://arxiv.org/abs/1810.04805)
- [HuggingFace Transformers documentation](https://huggingface.co/docs/transformers/)
- [Fine-tuning a pretrained model (HuggingFace tutorial)](https://huggingface.co/docs/transformers/training)

## Next Day Preview

Day 44 explores sentence embeddings with Sentence-BERT, enabling efficient semantic similarity comparisons between entire sentences.
