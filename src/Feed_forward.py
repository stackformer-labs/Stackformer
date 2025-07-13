import torch
import torch.nn as nn
import torch.nn.functional as F

class FF_ReLU(nn.Module):
    def __init__(self,emb_dim,hidden_dim,device='cpu',dtype=torch.float32):
        super().__init__()
        self.relu=nn.Sequential(
            nn.Linear(emb_dim,hidden_dim,device=device,dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim,emb_dim,device=device,dtype=dtype),
        )
    def forward(self,x):
        return self.relu(x)
    
class FF_LeakyReLU(nn.Module):
    def __init__(self,emb_dim,hidden_dim,negative_slope=0.1,device='cpu',dtype=torch.float32):
        super().__init__()
        self.l_relu=nn.Sequential(
            nn.Linear(emb_dim,hidden_dim,device=device,dtype=dtype),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim,emb_dim,device=device,dtype=dtype),
        )
    def forward(self,x):
        return self.l_relu(x)
    
class FF_GELU(nn.Module):
    def __init__(self,emb_dim,hidden_dim,device='cpu',dtype=torch.float32):
        super().__init__()
        self.gelu=nn.Sequential(
            nn.Linear(emb_dim,hidden_dim,device=device,dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_dim,emb_dim,device=device,dtype=dtype),
        )
    def forward(self,x):
        return self.gelu(x)
    
class FF_Sigmoid(nn.Module):
    def __init__(self,emb_dim,hidden_dim,device='cpu',dtype=torch.float32):
        super().__init__()
        self.sigmoid=nn.Sequential(
            nn.Linear(emb_dim,hidden_dim,device=device,dtype=dtype),
            nn.Sigmoid(),
            nn.Linear(hidden_dim,emb_dim,device=device,dtype=dtype),
        )
    def forward(self,x):
        return self.sigmoid(x)
    
class FF_SiLU(nn.Module):
    def __init__(self,emb_dim,hidden_dim,device='cpu',dtype=torch.float32):
        super().__init__()
        self.silu=nn.Sequential(
            nn.Linear(emb_dim,hidden_dim,device=device,dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim,emb_dim,device=device,dtype=dtype),
        )
    def forward(self,x):
        return self.silu(x)
    
