# Architecture (Developer Guide)

This document provides a high-level architecture overview for contributors.

## Design goals

- Keep transformer components modular and reusable
- Support both language and vision model implementations
- Provide practical training utilities for research and prototyping
- Keep training infrastructure testable and CI-friendly

## Package layout

```text
stackformer/
├── modules/      # core transformer blocks (attention, FFN, norms, masking, embeddings)
├── models/       # language model implementations (OpenAI/Meta/Google-style + baseline)
├── vision/       # vision architectures (ViT, SegFormer)
├── engine/       # training engine internals (state, checkpoint, trainer)
├── training/     # loop implementations
├── distributed/  # DDP utilities
├── amp/          # mixed precision scaler/helpers
├── logging/      # logging backends and metrics logging utilities
├── optim/        # optimizer/loss factories
└── utils/        # common helper functions
```

## Training stack

At a high level, model training composes:

1. Model definition (`stackformer.models`, `stackformer.vision`, or custom modules)
2. Engine/trainer orchestration (`stackformer.engine` and `stackformer.trainer`)
3. Optimization and scaling (`stackformer.optim`, `stackformer.amp`)
4. Distributed execution (`stackformer.distributed`)
5. Logging, metrics, and checkpoints (`stackformer.logging`, `stackformer.engine.checkpoint`)

## Contributor notes

- Use tests in `tests/` to validate behavior before submitting changes.
- Keep new components modular and package-scoped by feature area.
- Prefer adding user-facing guidance to `docs/user_docs/` and contributor guidance to `docs/developer_docs/`.
