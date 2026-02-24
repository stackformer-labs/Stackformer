<p align="center">
  <img src="assets/logo.png" alt="Stackformer logo" width="560" />
</p>

# Stackformer

> **Stackformer** is a modular PyTorch library for building, training, and extending Transformer architectures for language and vision tasks.

Stackformer is currently in an **early development stage**. The goal is to provide a clean, reusable, and developer-friendly foundation for industrial prototyping and research experimentation.

---

## Why Stackformer?

If you work on Transformers often, you need three things: reusable modules, understandable code, and flexibility to experiment quickly.

Stackformer focuses on exactly that:

- **Modular design**: core building blocks are separated and easy to reuse.
- **Research-friendly**: quickly try model variants and custom ideas.
- **Engineering-ready**: typed package, practical trainer, and clear project structure.
- **Multi-domain**: includes both NLP-oriented and vision-oriented transformer models.

---

## Repository structure (for developers)

```text
Stackformer/
├── stackformer/
│   ├── __init__.py
│   ├── generate.py
│   ├── trainer.py
│   ├── modules/
│   │   ├── Attention.py
│   │   ├── Feed_forward.py
│   │   ├── Normalization.py
│   │   └── position_embedding.py
│   ├── models/
│   │   ├── OpenAI.py
│   │   ├── Meta.py
│   │   ├── Google.py
│   │   └── Transformer.py
│   └── vision/
│       ├── vit.py
│       └── segformer.py
├── tests/
├── assets/
├── pyproject.toml
└── README.md
```
---

## Features

### 1) Core Transformer modules
- Attention variants (`Multi_Head_Attention`, `Multi_query_Attention`, `Group_query_Attention`, RoPE variants, KV cache helpers)
- Feed-forward blocks (`FF_ReLU`, `FF_GELU`, `FF_SwiGLU`, and more)
- Normalization layers (`LayerNormalization`, `RMSNormalization`)
- Positional embeddings (absolute, sinusoidal, RoPE)

### 2) Model implementations
- **OpenAI-style**: GPT family (`GPT_1`, `GPT_2`)
- **Meta-style**: LLaMA family (`llama_1`, `llama_2`)
- **Google-style**: Gemma family (`gemma_1_2b`, `gemma_1_7b`)
- **General transformer**: baseline `transformer`

### 3) Vision
- Vision Transformer (`ViT`)
- SegFormer (`SegFormerB0`)

### 4) Utilities
- Text generation helper (`text_generate`)
- Training utility (`Trainer`) with optimizer/scheduler options, evaluation, checkpoints, and resume support

---

## Installation guide

### A) Standard install
```bash
pip install stackformer
```

### B) With training extras
```bash
pip install "stackformer[train]"
```

### C) For development and tests
```bash
pip install "stackformer[dev]"
```

### D) Recommended virtual environment setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -U pip
pip install stackformer
```

### E) Conda setup
```bash
conda create -n stackformer python=3.10 -y
conda activate stackformer
pip install stackformer
```

---

## Getting started

### 1) Quick model example (GPT-2 style)
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
print(logits.shape)  # [2, 16, 128]
```

### 2) Using modules directly (attention + feed-forward)
```python
import torch
from stackformer.modules.Attention import Multi_Head_Attention
from stackformer.modules.Feed_forward import FF_GELU

x = torch.randn(2, 16, 64)
attn = Multi_Head_Attention(embed_dim=64, num_heads=4, dropout=0.1)
ff = FF_GELU(embed_dim=64, hidden_dim=128, dropout=0.1)

h = attn(x)
out = ff(h)
print(out.shape)
```

### 3) Trainer usage (minimal workflow)
```python
from torch.utils.data import TensorDataset
import torch
from stackformer.models.OpenAI import GPT_2
from stackformer.trainer import Trainer

# Dummy tokenized dataset
inputs = torch.randint(0, 128, (256, 16))
targets = torch.randint(0, 128, (256, 16))
train_ds = TensorDataset(inputs, targets)
eval_ds = TensorDataset(inputs[:64], targets[:64])

model = GPT_2(
    vocab_size=128,
    num_layers=2,
    embed_dim=32,
    num_heads=4,
    seq_len=16,
    dropout=0.0,
    hidden_dim=64,
)

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    train_batch_size=8,
    eval_batch_size=8,
    vocab_size=128,
    output_dir="./outputs",
    num_epoch=1,
    lr=3e-4,
)

trainer.train()
```

### 4) Creating a custom model with Stackformer blocks
```python
import torch
import torch.nn as nn
from stackformer.modules.Attention import Multi_Head_Attention
from stackformer.modules.Feed_forward import FF_SwiGLU
from stackformer.modules.Normalization import RMSNormalization

class TinyCustomTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, hidden_dim=128):
        super().__init__()
        self.norm1 = RMSNormalization(embed_dim)
        self.attn = Multi_Head_Attention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)
        self.norm2 = RMSNormalization(embed_dim)
        self.ff = FF_SwiGLU(embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=0.1)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

model = TinyCustomTransformer()
y = model(torch.randn(2, 16, 64))
print(y.shape)
```

---

## Communication & community

- **Issues (bugs/feature requests):** https://github.com/Gurumurthy30/Stackformer/issues
- **Discussions (Q&A, ideas):** https://github.com/Gurumurthy30/Stackformer/discussions
- **Releases:** https://github.com/Gurumurthy30/Stackformer/releases
- **Maintainer GitHub:** https://github.com/Gurumurthy30

If you are using Stackformer in industry or research, open a discussion and share feedback—use-cases help shape the roadmap.

---

## Roadmap

Planned improvements include:

- Optimize existing model implementations for better speed and stability.
- Add more transformer-based architectures.
- Improve training optimization workflows.
- Add monitoring features for training visibility and experiment tracking.

---

## License

MIT License. See [LICENSE](LICENSE).
