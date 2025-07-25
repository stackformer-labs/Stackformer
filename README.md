## 🧱 Stackformer

**Stackformer** is a modular transformer-building framework written entirely in PyTorch. It is designed primarily for experimentation, providing various transformer blocks such as attention mechanisms, normalization layers, feed-forward networks, and a simple model architecture. The project is a work-in-progress with plans for further enhancements and expansions.

---

## 📖 About Me

My name is **Gurumurthy**, and I am a final-year Bachelor of Engineering student from India. I created this library as my own size project to showcase my skills and knowledge in deep learning and transformer architectures.

I am also interested and free to work with others on different projects for knowledge sharing and building connections.

---

## 🌟 Features

- Multiple attention mechanisms including multi-head, group query, linear, local, and KV cache variants  
- Token embedding via `tiktoken`  
- Absolute and sinusoidal positional embeddings  
- Normalization layers like LayerNorm and RMSNorm  
- Several feed-forward network variants with activations such as ReLU, GELU, SiLU, LeakyReLU, and Sigmoid  
- A simple GPT-style transformer model implementation  

---

## 📁 Project Structure

stackformer/ \
|-- modules/ \
|   |-- tokenizer.py            # Token embedding using tiktoken \
|   |-- position_embedding.py   # Absolute and sinusoidal embeddings \
|   |-- Attention.py            # Attention mechanisms \
|   |-- Normalization.py        # LayerNorm and RMSNorm \
|   |-- Feed_forward.py         # Feed-forward layers with various activations \
|-- models/ \
|   -- GPT_2.py               # GPT-style transformer stack model \
-- trainer.py                 # Training loop and utilities \

---

## 💻 Installation

✅ Method 1: Install from PyPI:
```bash
pip install Stackformer
import stackformer
```

🔧 Method 2: Clone the repository:
```bash
git clone https://github.com/Gurumurthy30/Stackformer
cd Stackformer
pip install -e .
```

---

## 🚀 Future Plans
Currently, I am working on improving and optimizing the existing components while fixing known bugs and issues. After stabilizing the current modules, I plan to add more advanced blocks like Mixture of Experts (MoE), mask handling, and other essential transformer components. Eventually, I will expand the library by developing more comprehensive model architectures.