we only get emb_dim, n_head, max_token_id and dropout as input to our model

1) first complete work on attention mechanism:
    1) self attention ✅
    2) multi head attention ✅
    3) multi head cross attention ✅
    4) multi query attention ✅
    5) grouped query attention ✅
    6) multi head latent attention ✅ 
    7) linear attention ✅
    8) KV catch with MLA ✅
    9) KV catch with GQA ✅


todo:
    - flash attention tooo deficult (may be with another library)
    - big bird (Spars attention) -> global + random + window masks 

2) Normalization and skip connection:
- layer normalization ✅
- RMS normalization ✅

3) Feed forward layer:
    - standard feed forward
    - GLU feed forward

Activation function:
    - sigmoid ✅
    - relu ✅
    - gelu ✅
    - leaky relu ✅
    - slu/swish ✅

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