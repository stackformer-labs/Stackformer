import torch
import torch.nn as nn

# Norm = (Xn - Mean) / sqrt(var) + Bias
# output = γ x Norn + β
class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        output = self.weight * norm_x + self.bias
        return output.to(device=x.device,dtype=x.dtype)

# RMS = sqrt(Xn ** 2)
# Norm = Xn / RMS
class RMSNormilization(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        norm = self.weight * x / (rms + self.eps)
        return norm.to(device=x.device, dtype=x.dtype)