<p align="center">
  <img src="assets/logo.png" alt="StackFormer logo" width="560" />
</p>

<p align="center">
  <a href="https://pypi.org/project/stackformer/"><img src="https://img.shields.io/pypi/v/stackformer.svg" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/stackformer/"><img src="https://img.shields.io/pypi/pyversions/stackformer.svg" alt="Python versions" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" /></a>
  <a href="https://github.com/Gurumurthy30/Stackformer/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/Gurumurthy30/Stackformer/core-tests.yml?branch=main&label=CI" alt="CI status" />
  </a>
</p>

# StackFormer

StackFormer is a modular PyTorch framework for building, training, and experimenting with Transformer architectures.

## Overview

StackFormer is designed for fast experimentation with reusable Transformer building blocks and model implementations. It supports both language and vision workflows in a single modular codebase. The framework is built for research, prototyping, and iterative model development with practical training infrastructure.

## Key Features

- Modular transformer components
- GPT / LLaMA / Gemma-style model implementations
- Vision models (ViT, SegFormer)
- Trainer infrastructure with AMP mixed precision and DDP support
- Logging and metrics utilities
- Checkpointing and resume training
- CI-tested training infrastructure

## Project Structure

```text
Stackformer/
├── assets/
├── docs/
│   ├── user_docs/
│   └── developer_docs/
├── examples/
├── stackformer/
│   ├── __init__.py
│   ├── generate.py
│   ├── metrics.py
│   ├── py.typed
│   ├── amp/
│   │   └── scaler.py
│   ├── distributed/
│   │   └── ddp.py
│   ├── engine/
│   │   ├── checkpoint.py
│   │   ├── engine.py
│   │   ├── state.py
│   │   └── trainer.py
│   ├── logging/
│   │   ├── csv_logger.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── tensorboard_logger.py
│   │   ├── wandb_logger.py
│   │   └── wb_logger.py
│   ├── models/
│   │   ├── OpenAI.py
│   │   ├── Meta.py
│   │   ├── Google.py
│   │   └── Transformer.py
│   ├── modules/
│   │   ├── Attention.py
│   │   ├── Feed_forward.py
│   │   ├── Masking.py
│   │   ├── Normalization.py
│   │   ├── position_embedding.py
│   ├── optim/
│   │   ├── factories.py
│   │   └── loss_fn.py
│   ├── training/
│   │   └── loops.py
│   ├── utils/
│   │   ├── device.py
│   │   └── utils.py
│   └── vision/
│       ├── vit.py
│       └── segformer.py
├── tests/
│   ├── integration/
│   ├── models/
│   ├── modules/
│   ├── trainer/
│   ├── utils/
│   ├── conftest.py
│   ├── test_distributed.py
│   └── test_vision.py
├── LICENSE
├── pyproject.toml
└── README.md
```

## Installation

Python >= 3.10

### Install from PyPI

```bash
pip install stackformer
```

### Install from source

```bash
git clone https://github.com/Gurumurthy30/Stackformer.git
cd Stackformer
pip install -e .
```

## Quick Start

```python
from stackformer.engine import Trainer
import torch.nn as nn

model = nn.Linear(10, 1)

trainer = Trainer(model=model)
trainer.fit(dataset)
```

## Examples

More runnable examples are available in:

```text
examples/
```

```text
examples/simple_train.py
examples/simple_trainer_v2.py
examples/train_ddp.py
```

## Documentation

- User documentation: [docs/user_docs/installation.md](docs/user_docs/installation.md)
- Developer documentation: [docs/developer_docs/architecture.md](docs/developer_docs/architecture.md)

## Community

- Issues: https://github.com/Gurumurthy30/Stackformer/issues
- Discussions: https://github.com/Gurumurthy30/Stackformer/discussions
- Releases: https://github.com/Gurumurthy30/Stackformer/releases
