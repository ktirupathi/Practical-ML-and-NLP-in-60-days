# Day 55: Performance Optimization

## Learning Objectives

- Apply model quantization (INT8, FP16, INT4) to reduce model size and inference latency
- Understand pruning techniques that remove redundant weights while preserving accuracy
- Implement knowledge distillation to train a small student model from a large teacher model
- Use ONNX Runtime to accelerate inference across different hardware backends
- Design caching strategies that eliminate redundant model calls in production

## Key Concepts

Deploying ML models in production often requires meeting strict latency, throughput,
and cost targets that raw model inference cannot satisfy. A BERT-base model takes
roughly 10ms per inference on a GPU but 100ms+ on a CPU -- too slow for real-time
applications at scale. Performance optimization techniques address this gap without
retraining from scratch. Quantization reduces the numerical precision of model weights
and activations (e.g., from FP32 to INT8), cutting model size by 2-4x and speeding up
inference by 2-3x on CPUs that have optimized integer arithmetic paths. Dynamic
quantization is the simplest form, requiring no calibration data.

Pruning removes weights (or entire neurons, attention heads, or layers) that contribute
least to the model's output. Structured pruning removes whole units (e.g., attention
heads), making the pruned model directly faster on standard hardware. Unstructured
pruning zeroes out individual weights, achieving higher compression ratios but requiring
sparse matrix support to realize speedups. Knowledge distillation trains a smaller
"student" model to mimic the outputs of a larger "teacher" model, often matching 95%+
of the teacher's accuracy at a fraction of the size. DistilBERT is a well-known example:
it retains 97% of BERT's accuracy with 40% fewer parameters.

ONNX (Open Neural Network Exchange) provides a hardware-agnostic model format, and ONNX
Runtime optimizes inference with graph optimizations, operator fusion, and hardware-
specific accelerations. Converting a PyTorch or TensorFlow model to ONNX and running it
through ONNX Runtime typically yields 2-3x speedup with no accuracy loss. At the
application level, caching predictions for repeated or similar inputs (using Redis or
in-memory LRU caches) can eliminate model calls entirely for common queries.

## Practical Example

```python
"""
Performance optimization techniques: quantization, ONNX, and caching.
"""
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import time
import functools

# Load model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

sample_text = "This is a test sentence for performance benchmarking."
inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)

# Baseline inference time
def benchmark(model_fn, inputs, n_runs=50):
    # Warmup
    for _ in range(5):
        model_fn(**inputs)
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model_fn(**inputs)
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    return elapsed

baseline_ms = benchmark(model, inputs)
print(f"Baseline (FP32):     {baseline_ms:.1f} ms/inference")

# 1. Dynamic quantization (INT8)
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
quant_ms = benchmark(quantized_model, inputs)
print(f"Quantized (INT8):    {quant_ms:.1f} ms/inference ({baseline_ms/quant_ms:.1f}x speedup)")

# Compare model sizes
import os, tempfile

def model_size_mb(m):
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(m.state_dict(), f.name)
        return os.path.getsize(f.name) / 1e6

print(f"\nFP32 model size: {model_size_mb(model):.0f} MB")
print(f"INT8 model size: {model_size_mb(quantized_model):.0f} MB")

# 2. Export to ONNX
onnx_path = "/tmp/bert_classifier.onnx"
dummy_input = {
    "input_ids": inputs["input_ids"],
    "attention_mask": inputs["attention_mask"],
}

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
    opset_version=14,
)
print(f"\nONNX model exported to {onnx_path}")
print(f"ONNX model size: {os.path.getsize(onnx_path) / 1e6:.0f} MB")

# 3. Simple prediction cache
class CachedPredictor:
    def __init__(self, model, tokenizer, cache_size=1024):
        self.model = model
        self.tokenizer = tokenizer
        self.cache_hits = 0
        self.total_calls = 0
        self._predict = functools.lru_cache(maxsize=cache_size)(self._predict_impl)

    def _predict_impl(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.argmax(logits, dim=-1).item()

    def predict(self, text):
        self.total_calls += 1
        return self._predict(text)

cached = CachedPredictor(quantized_model, tokenizer)
texts = ["Great movie!", "Terrible film.", "Great movie!", "Great movie!", "Terrible film."]
for t in texts:
    cached.predict(t)
info = cached._predict.cache_info()
print(f"\nCache stats: {info.hits} hits, {info.misses} misses, {info.hits/(info.hits+info.misses):.0%} hit rate")
```

## Resources

- [PyTorch quantization documentation](https://pytorch.org/docs/stable/quantization.html)
- [ONNX Runtime: accelerate ML inference](https://onnxruntime.ai/)
- [DistilBERT: a distilled version of BERT (Sanh et al.)](https://arxiv.org/abs/1910.01108)

## Next Day Preview

Day 56 covers security for ML APIs -- authentication, input validation, rate limiting, and protecting against adversarial inputs.
