import torch
import torch.nn as nn
import tiktoken

# Bite pair (BPE) Embedding using tokienizer
class Embedding_using_tiktoken:
    def __init__(self,data,embedding_dim,model: str):
        self.tokenizer = tiktoken.get_encoding(model)
        
        
    def encoding(self,data,embedding_dim):
        max_token_id = self.tokenizer.n_vocab
        embedding_layer = nn.Embedding(num_embeddings = max_token_id, embedding_dim = embedding_dim)
        tensors = torch.tensor(self.tokenizer.encode(data))
        embedded = embedding_layer(tensors)
        return embedded
    
    def decoding(self,data):
        return self.tokenizer.decode(data)
    
    def vocab_size(self):
        return self.tokenizer.n_vocab
    
    def model_list(self):
        return tiktoken.list_encoding_names()
