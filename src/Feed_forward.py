import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self,emb_dim,hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,emb_dim)
        self.activation_layer = nn.ReLU()                   # nn.GELU() / nn.LeakyReLU()
    def forward(self,x):
        x = self.linear1(x)
        x = self.activation_layer(x)
        return self.linear2(x)
    
class FF_LeakyReLU(nn.Module):
    def __init__(self,emb_dim,hidden_dim,negative_slope=0.1):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,emb_dim)
        self.activation_layer = nn.LeakyReLU(negative_slope=negative_slope)       
    def forward(self,x):
        x = self.linear1(x)
        x = self.activation_layer(x)
        return self.linear2(x)
    
class FF_GELU(nn.Module):
    def __init__(self,emb_dim,hidden_dim,negative_slope=0.1):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,emb_dim)
        self.activation_layer = nn.GELU()  
    def forward(self,x):
        x = self.linear1(x)
        x = self.activation_layer(x)
        return self.linear2(x)
    