# Stackformer

A lightweight, modular Transformer library in PyTorch.

## What is included
- Reusable Transformer modules (attention, normalization, feed-forward, positional encodings)
- Reference language model implementations (GPT-style, LLaMA-style, Gemma-style)
- Vision Transformer (`ViT`)
- Training and text-generation utilities

## Installation

```bash
pip install stackformer
```

For training schedulers from Hugging Face Transformers:

```bash
pip install "stackformer[train]"
```

For local development/tests:

```bash
pip install "stackformer[dev]"
```

## Quick example

```python
import torch
from stackformer.models.OpenAI import GPT_2

model = GPT_2(
    vocab_size=128,
    num_layers=2,
    embed_dim=32,
    num_heads=4,
    seq_len=16,
    dropout=0.0,
    hidden_dim=64,
)

x = torch.randint(0, 128, (2, 16))
logits = model(x)
print(logits.shape)  # torch.Size([2, 16, 128])
```

## API stability
- Public exports are listed in `stackformer.__all__`.
- Backwards-compatibility policy: see [BACKWARDS_COMPATIBILITY.md](BACKWARDS_COMPATIBILITY.md).
- Release notes: see [CHANGELOG.md](CHANGELOG.md).

## License
MIT
