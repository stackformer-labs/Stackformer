import torch
import torch.nn as nn
import torch.nn.functional as F

class FF_ReLU(nn.Module):
    def __init__(self,emb_dim,hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,emb_dim)
    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        return self.linear2(x)
    
class FF_LeakyReLU(nn.Module):
    def __init__(self,emb_dim,hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,emb_dim)      
    def forward(self,x,negative_slope=0.1):
        x = self.linear1(x)
        x = F.leaky_relu(x,negative_slope)
        return self.linear2(x)
    
class FF_GELU(nn.Module):
    def __init__(self,emb_dim,hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,emb_dim)
    def forward(self,x):
        x = self.linear1(x)
        x = F.gelu(x)
        return self.linear2(x)
    
class FF_Sigmoid(nn.Module):
    def __init__(self,emb_dim,hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,emb_dim)
    def forward(self,x):
        x = self.linear1(x)
        x = F.sigmoid(x)
        return self.linear2(x)
    
class FF_SiLU(nn.Module):
    def __init__(self,emb_dim,hidden_dim,negative_slope=0.1):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,emb_dim)
    def forward(self,x):
        x = self.linear1(x)
        x = F.silu(x)
        return self.linear2(x)
    
