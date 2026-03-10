# Installation (User Guide)

This guide covers the recommended ways to install StackFormer.

## Requirements

- Python 3.10 or newer
- pip (latest recommended)

## Install from PyPI (recommended)

```bash
pip install stackformer
```

## Install from source (editable mode)

```bash
git clone https://github.com/Gurumurthy30/Stackformer.git
cd Stackformer
pip install -e .
```

## Verify installation

```bash
python -c "import stackformer; print(stackformer.__version__ if hasattr(stackformer, '__version__') else 'stackformer imported')"
```

## Optional: virtual environment setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install stackformer
```
