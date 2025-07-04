import torch
import torch.nn as nn
import tiktoken

# letter Embedding
# def letter_Embedding(data):
#     vocab = list(set(data))
#     vocab.sort()
#     vocab = {letter:i for i, letter in enumerate(vocab)}
#     return vocab

# word level Embedding
# def word_Embedding(data):
#     vocab = list(set(data.split()))
#     vocab.sort()
#     vocab = {word:i for i, word in enumerate(vocab)}
#     return vocab

# Bite pair (BPE) Embedding using tokienizer
def Embedding_using_tiktoken(data,embedding_dim,model: str):
    # 'gpt2', 'r50k_base', 'p50k_base', 'p50k_edit', 'cl100k_base', 'o200k_base'
    tokenizer = tiktoken.get_encoding(model)
    max_token_id = tokenizer.n_vocab
    print("Max Token ID: ",max_token_id,"\n")
    embedding_layer = nn.Embedding(num_embeddings = max_token_id, embedding_dim = embedding_dim)
    tensors = torch.tensor(tokenizer.encode(data))
    embedded = embedding_layer(tensors)
    return embedded