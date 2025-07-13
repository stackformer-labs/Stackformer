google colab: `https://colab.research.google.com/drive/1L7feQvzgxWigEEA8EE6j9ADMz8sJetQv#scrollTo=nlZcK6X7RnUb`

1) first complete work on attention mechanism:
    - flash attention tooo deficult
    - Mixture of expose (MoE)
    - big bird (Spars attention) -> global + random + window masks 


4) position embedding:
    - Abosulet embedding (arange()) | 3 ✅
    - relative position embedding | 1
    - sinusoidal position embedding | 4 ✅
    - rotery position embedding | 2 (bone class ah mathi add pana num)

5) tokenization and embedding

6) popular architecture:
easy:
    1, GPT-2 decoder-only (basic MHA, layer norm, simple stack)
    2, BERD encoder-only
    3, RoBERta encoder-only
    4, Transformers encoder-decoder
    5, ViT encoder-decoder
intermediate:
    1, GPT-Neo decoder-only
    2, OPT decoder-only
    3, T5 encoder-decoder
    4, Gemma decoder-only
    5, LLama 1 decoder-only
Advanced:
    1, Llama 2/3 decoder-only
    2, Mistral decoder-only
    3, Mixtral decoder-only
    4, Flamingo Multimodel
    5, Perceiver IO MultiModel
Special Arch:
    1, RetNet Transformers-like
    2, RWKV RNN-style

7) multi-model arch added