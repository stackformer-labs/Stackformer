import torch
import torch.nn as nn
import torch.nn.functional as F

class FF_ReLU(nn.Module):
    def __init__(self,embed_dim,hidden_dim,dropout = 0.0, device='cpu',dtype=torch.float32):
        super().__init__()
        self.relu=nn.Sequential(
            nn.Linear(embed_dim,hidden_dim,device=device,dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,embed_dim,device=device,dtype=dtype),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        x = self.relu(x)
        return x.to(device=x.device,dtype=x.dtype)
    
class FF_LeakyReLU(nn.Module):
    def __init__(self,embed_dim,hidden_dim, dropout = 0.0, negative_slope=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        self.l_relu=nn.Sequential(
            nn.Linear(embed_dim,hidden_dim,device=device,dtype=dtype),
            nn.LeakyReLU(negative_slope),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,embed_dim,device=device,dtype=dtype),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        x = self.l_relu(x)
        return x.to(device=x.device,dtype=x.dtype)
    
class FF_GELU(nn.Module):
    def __init__(self,embed_dim,hidden_dim, dropout = 0.0, device='cpu', dtype=torch.float32):
        super().__init__()
        self.gelu=nn.Sequential(
            nn.Linear(embed_dim,hidden_dim,device=device,dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,embed_dim,device=device,dtype=dtype),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        x = self.gelu(x)
        return x.to(device=x.device,dtype=x.dtype)
    
class FF_Sigmoid(nn.Module):
    def __init__(self,embed_dim,hidden_dim, dropout = 0.0, device='cpu', dtype=torch.float32):
        super().__init__()
        self.sigmoid=nn.Sequential(
            nn.Linear(embed_dim,hidden_dim,device=device,dtype=dtype),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,embed_dim,device=device,dtype=dtype),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        x = self.sigmoid(x)
        return x.to(device=x.device,dtype=x.dtype)
    
class FF_SiLU(nn.Module):
    def __init__(self,embed_dim,hidden_dim, dropout = 0.0, device='cpu',dtype=torch.float32):
        super().__init__()
        self.silu=nn.Sequential(
            nn.Linear(embed_dim,hidden_dim,device=device,dtype=dtype),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,embed_dim,device=device,dtype=dtype),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        x = self.silu(x)
        return x.to(device=x.device,dtype=x.dtype)

class FF_SwiGLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, device='cpu', dtype=torch.float32):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim * 2, device=device, dtype=dtype)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x_proj = self.linear1(x)            
        x1, x2 = x_proj.chunk(2, dim=-1)  
        x = x2 * F.silu(x1)                 
        x = self.dropout1(x)
        x = self.linear2(x)                 
        x = self.dropout2(x)
        return x.to(device=x.device,dtype=x.dtype)

class FF_GeGLU(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, device='cpu', dtype=torch.float32):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim * 2, device=device, dtype=dtype)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x_proj = self.linear1(x)            
        x1, x2 = x_proj.chunk(2, dim=-1)  
        x = x2 * F.gelu(x1)                 
        x = self.dropout1(x)
        x = self.linear2(x)                 
        x = self.dropout2(x)
        return x.to(device=x.device,dtype=x.dtype)