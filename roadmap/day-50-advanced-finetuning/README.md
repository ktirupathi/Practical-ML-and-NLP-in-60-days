# Day 50: Advanced Fine-Tuning Techniques

## Learning Objectives

- Understand why full fine-tuning of large models is often impractical and how parameter-efficient methods solve this
- Explain LoRA (Low-Rank Adaptation) and how it injects trainable low-rank matrices into frozen model layers
- Use QLoRA to fine-tune quantized models that fit in consumer GPU memory
- Apply the PEFT library to fine-tune a language model with adapters in under 1% of total parameters
- Compare adapter-based, prompt-tuning, and prefix-tuning approaches for parameter-efficient fine-tuning

## Key Concepts

Full fine-tuning of modern language models updates all parameters, which for a 7B model
means storing optimizer states for 7 billion weights -- requiring 100+ GB of GPU memory.
Parameter-efficient fine-tuning (PEFT) methods freeze the pretrained weights and only
train a small number of additional parameters, dramatically reducing memory and compute
requirements while achieving comparable performance. LoRA (Low-Rank Adaptation) is
the most popular approach: it decomposes weight updates into two small matrices
(rank r, typically 8-64), so instead of updating a 4096x4096 matrix, you train two
matrices of size 4096xr and rx4096. This reduces trainable parameters by 100-1000x.

QLoRA pushes efficiency further by quantizing the base model to 4-bit precision (using
NF4 quantization) and applying LoRA adapters on top. This allows fine-tuning a 65B
parameter model on a single 48GB GPU -- a task that would otherwise require a cluster.
The key insight is that 4-bit quantized models retain most of their capability, and the
LoRA adapters learn the task-specific adjustments in full precision. Double quantization
further reduces the memory footprint of quantization constants.

The HuggingFace PEFT library provides a unified interface for multiple parameter-efficient
methods: LoRA, prefix tuning (prepending trainable tokens to the input), prompt tuning
(learning soft prompts), and adapters (inserting small bottleneck layers). In practice,
LoRA applied to the attention projection matrices (q_proj and v_proj) delivers the best
quality-efficiency trade-off for most tasks. Trained adapters are tiny (often 10-50MB)
and can be hot-swapped at inference time, enabling one base model to serve many tasks.

## Practical Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import torch

# Load base model (using a small model for demo purposes)
model_name = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float32, device_map="auto"
)

# Check baseline parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Base model parameters: {total_params:,}")

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                     # Rank of the low-rank matrices
    lora_alpha=32,            # Scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
    bias="none",
)

# Apply LoRA to the model
peft_model = get_peft_model(model, lora_config)

# Compare parameter counts
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in peft_model.parameters())
print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")
print(f"Total parameters:     {total:,}")

# Print the LoRA architecture
peft_model.print_trainable_parameters()

# Prepare a small dataset
dataset = load_dataset("imdb", split="train[:500]")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
tokenized.set_format("torch")

# Training arguments (lightweight for demo)
training_args = TrainingArguments(
    output_dir="./lora-opt-350m",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    logging_steps=25,
    save_strategy="no",
)

# Save just the adapter (tiny file)
peft_model.save_pretrained("./lora-adapter")
print("\nAdapter saved! Check the file size -- it will be very small.")

# Later, load the adapter on top of any copy of the base model:
# from peft import PeftModel
# base = AutoModelForCausalLM.from_pretrained(model_name)
# model_with_adapter = PeftModel.from_pretrained(base, "./lora-adapter")
```

## Resources

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [HuggingFace PEFT documentation](https://huggingface.co/docs/peft/)

## Next Day Preview

Day 51 shifts to ML system design -- the principles and patterns for building scalable, production-grade machine learning systems.
