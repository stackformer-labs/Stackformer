"""
Feed Forward Neural Network Modules for StackFormer Library

This module contains various feed-forward network implementations with different 
activation functions commonly used in transformer architectures and deep learning models.
modules such as:
    - FF with ReLU
    - FF with Leaky ReLU
    - FF with SiLU
    - FF with GELU
    - FF with Sigmoid
    - FF with SwiGLU
    - FF with GeGLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FF_ReLU(nn.Module):
    """
    Feed Forward Network with ReLU activation function.
    
    Implements a two-layer feed-forward network:
    Input -> Linear -> ReLU -> Dropout -> Linear -> Dropout -> Output
    
    Args:
        embed_dim (int): Input and output embedding dimension
        hidden_dim (int): Hidden layer dimension (typically 4x embed_dim)
        dropout (float, optional): Dropout probability. Default: 0.0
        device (str, optional): Device to place tensors ('cpu' or 'cuda'). Default: 'cpu'
        dtype (torch.dtype, optional): Data type for parameters. Default: torch.float32
    
    Example:
        >>> ff = FF_ReLU(embed_dim=512, hidden_dim=2048, dropout=0.1)
        >>> x = torch.randn(32, 100, 512)  # (batch, seq_len, embed_dim)
        >>> output = ff(x)  # Shape: (32, 100, 512)
    """
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, device='cpu', dtype=torch.float32):
        super().__init__()
        
        # Two-layer MLP with ReLU activation
        self.relu = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),  # Expand to hidden_dim
            nn.ReLU(),                                                     # ReLU activation
            nn.Dropout(dropout),                                          # Regularization
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype), # Project back to embed_dim
            nn.Dropout(dropout),                                          # Final regularization
        )
    
    def forward(self, x):
        """
        Forward pass through the feed-forward network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., embed_dim)
        
        Returns:
            torch.Tensor: Output tensor of same shape as input
        """
        x = self.relu(x)
        return x.to(device=x.device, dtype=x.dtype)  # Ensure correct device/dtype


class FF_LeakyReLU(nn.Module):
    """
    Feed Forward Network with Leaky ReLU activation function.
    
    Leaky ReLU allows small negative values to pass through, helping prevent
    the "dying ReLU" problem where neurons can become permanently inactive.
    
    Args:
        embed_dim (int): Input and output embedding dimension
        hidden_dim (int): Hidden layer dimension
        dropout (float, optional): Dropout probability. Default: 0.0
        negative_slope (float, optional): Slope for negative values. Default: 0.1
        device (str, optional): Device to place tensors. Default: 'cpu'
        dtype (torch.dtype, optional): Data type for parameters. Default: torch.float32
    
    Example:
        >>> ff = FF_LeakyReLU(embed_dim=256, hidden_dim=1024, negative_slope=0.01)
        >>> x = torch.randn(16, 50, 256)
        >>> output = ff(x)
    """
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, negative_slope=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        
        self.l_relu = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.LeakyReLU(negative_slope),  # Allows small negative values through
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """Forward pass with Leaky ReLU activation."""
        x = self.l_relu(x)
        return x.to(device=x.device, dtype=x.dtype)


class FF_GELU(nn.Module):
    """
    Feed Forward Network with GELU (Gaussian Error Linear Unit) activation.
    
    GELU is a smooth activation function that works particularly well with
    transformer models and NLP tasks. It's used in BERT, GPT, and other
    state-of-the-art models.
    
    Formula: GELU(x) = x * Φ(x), where Φ(x) is the CDF of standard normal distribution
    
    Args:
        embed_dim (int): Input and output embedding dimension
        hidden_dim (int): Hidden layer dimension
        dropout (float, optional): Dropout probability. Default: 0.0
        device (str, optional): Device to place tensors. Default: 'cpu'
        dtype (torch.dtype, optional): Data type for parameters. Default: torch.float32
    
    Example:
        >>> ff = FF_GELU(embed_dim=768, hidden_dim=3072)  # BERT-base dimensions
        >>> x = torch.randn(8, 512, 768)
        >>> output = ff(x)
    """
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, device='cpu', dtype=torch.float32):
        super().__init__()
        
        self.gelu = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.GELU(),  # Smooth, probabilistic activation function
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """Forward pass with GELU activation."""
        x = self.gelu(x)
        return x.to(device=x.device, dtype=x.dtype)


class FF_Sigmoid(nn.Module):
    """
    Feed Forward Network with Sigmoid activation function.
    
    Sigmoid squashes inputs to the range (0, 1) and provides smooth gradients.
    Useful for binary classification and gating mechanisms.
    
    Note: Can suffer from vanishing gradients in deep networks.
    
    Args:
        embed_dim (int): Input and output embedding dimension
        hidden_dim (int): Hidden layer dimension
        dropout (float, optional): Dropout probability. Default: 0.0
        device (str, optional): Device to place tensors. Default: 'cpu'
        dtype (torch.dtype, optional): Data type for parameters. Default: torch.float32
    
    Example:
        >>> ff = FF_Sigmoid(embed_dim=128, hidden_dim=512)
        >>> x = torch.randn(64, 20, 128)
        >>> output = ff(x)  # All values in (0, 1)
    """
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, device='cpu', dtype=torch.float32):
        super().__init__()
        
        self.sigmoid = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.Sigmoid(),  # Squashes to (0, 1) range
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """Forward pass with Sigmoid activation."""
        x = self.sigmoid(x)
        return x.to(device=x.device, dtype=x.dtype)


class FF_SiLU(nn.Module):
    """
    Feed Forward Network with SiLU (Sigmoid Linear Unit) activation.
    
    SiLU, also known as Swish, is defined as: SiLU(x) = x * sigmoid(x)
    It's smooth, non-monotonic, and often outperforms ReLU in deep networks.
    
    Args:
        embed_dim (int): Input and output embedding dimension
        hidden_dim (int): Hidden layer dimension
        dropout (float, optional): Dropout probability. Default: 0.0
        device (str, optional): Device to place tensors. Default: 'cpu'
        dtype (torch.dtype, optional): Data type for parameters. Default: torch.float32
    
    Example:
        >>> ff = FF_SiLU(embed_dim=384, hidden_dim=1536)
        >>> x = torch.randn(32, 196, 384)  # Vision transformer patches
        >>> output = ff(x)
    """
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, device='cpu', dtype=torch.float32):
        super().__init__()
        
        self.silu = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, device=device, dtype=dtype),
            nn.SiLU(),  # Smooth, self-gated activation: x * sigmoid(x)
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        """Forward pass with SiLU activation."""
        x = self.silu(x)
        return x.to(device=x.device, dtype=x.dtype)


class FF_SwiGLU(nn.Module):
    """
    Feed Forward Network with SwiGLU (Swish Gated Linear Unit) activation.
    
    SwiGLU uses a gating mechanism with SiLU activation:
    - Projects input to 2*hidden_dim
    - Splits into two parts: gate and value
    - Applies SiLU to gate part
    - Element-wise multiplication: value * SiLU(gate)
    
    This architecture is used in modern LLMs like PaLM and LLaMA for better performance.
    
    Args:
        embed_dim (int): Input and output embedding dimension
        hidden_dim (int): Hidden layer dimension (note: first layer outputs 2*hidden_dim)
        dropout (float, optional): Dropout probability. Default: 0.0
        device (str, optional): Device to place tensors. Default: 'cpu'
        dtype (torch.dtype, optional): Data type for parameters. Default: torch.float32
    
    Example:
        >>> ff = FF_SwiGLU(embed_dim=1024, hidden_dim=2048)
        >>> x = torch.randn(16, 128, 1024)
        >>> output = ff(x)
    """
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, device='cpu', dtype=torch.float32):
        super().__init__()
        
        # First linear layer projects to 2*hidden_dim for gating
        self.linear1 = nn.Linear(embed_dim, hidden_dim * 2, device=device, dtype=dtype)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second linear layer projects back to embed_dim
        self.linear2 = nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass with SwiGLU gating mechanism.
        
        Process:
        1. Project to 2*hidden_dim
        2. Split into gate and value parts
        3. Apply gating: value * SiLU(gate)
        4. Project back to embed_dim
        """
        # Project to 2*hidden_dim
        x_proj = self.linear1(x)
        
        # Split into two equal parts along last dimension
        x1, x2 = x_proj.chunk(2, dim=-1)  # Each has shape (..., hidden_dim)
        
        # Gating mechanism: x2 (value) * SiLU(x1) (gate)
        x = x2 * F.silu(x1)
        x = self.dropout1(x)
        
        # Project back to original dimension
        x = self.linear2(x)
        x = self.dropout2(x)
        
        return x.to(device=x.device, dtype=x.dtype)


class FF_GeGLU(nn.Module):
    """
    Feed Forward Network with GeGLU (GELU Gated Linear Unit) activation.
    
    Similar to SwiGLU but uses GELU instead of SiLU for the gating mechanism.
    GeGLU combines the smoothness of GELU with gating for enhanced expressiveness.
    
    Architecture:
    - Projects input to 2*hidden_dim
    - Splits into gate and value components  
    - Applies GELU to gate part
    - Element-wise multiplication: value * GELU(gate)
    
    Args:
        embed_dim (int): Input and output embedding dimension
        hidden_dim (int): Hidden layer dimension
        dropout (float, optional): Dropout probability. Default: 0.0
        device (str, optional): Device to place tensors. Default: 'cpu'
        dtype (torch.dtype, optional): Data type for parameters. Default: torch.float32
    
    Example:
        >>> ff = FF_GeGLU(embed_dim=512, hidden_dim=1024)
        >>> x = torch.randn(8, 256, 512)
        >>> output = ff(x)
    """
    
    def __init__(self, embed_dim, hidden_dim, dropout=0.0, device='cpu', dtype=torch.float32):
        super().__init__()
        
        # First linear layer for gating (outputs 2*hidden_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim * 2, device=device, dtype=dtype)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second linear layer to project back
        self.linear2 = nn.Linear(hidden_dim, embed_dim, device=device, dtype=dtype)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass with GeGLU gating mechanism.
        
        Process:
        1. Project to 2*hidden_dim
        2. Split into gate and value parts
        3. Apply gating: value * GELU(gate)
        4. Project back to embed_dim
        """
        # Project to 2*hidden_dim for gating
        x_proj = self.linear1(x)
        
        # Split into gate and value components
        x1, x2 = x_proj.chunk(2, dim=-1)  # Each: (..., hidden_dim)
        
        # GeGLU gating: value * GELU(gate)
        x = x2 * F.gelu(x1)
        x = self.dropout1(x)
        
        # Project back to original embedding dimension
        x = self.linear2(x)
        x = self.dropout2(x)
        
        return x.to(device=x.device, dtype=x.dtype)