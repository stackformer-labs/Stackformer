import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x 

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(emb_dim, device=device, dtype=dtype))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.weight + self.bias


# RMS = sqrt(Xn ** 2)
# Norm = Xn / RMS
class RMSNormilization(nn.Module):
    
    def __init__(self,dim,eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self,x):
        rms = x.pow(2).mean(-1,keepdim=True).sqrt()
        norm = self.weight * x / (rms + self.eps)
        return norm