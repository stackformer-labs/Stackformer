"""Feed-forward blocks used in Stackformer transformer layers.

This module implements several MLP variants used after attention:
- standard 2-layer FFN with different activations (ReLU, LeakyReLU, GELU, Sigmoid, SiLU)
- gated FFN variants (SwiGLU, GeGLU)

Canonical transformer FFN equation:

    FFN(x) = W2 * act(W1 * x + b1) + b2

Gated variants use two parallel projections and elementwise gating:

    z = Wg * x, v = Wv * x
    y = W2 * (v ⊙ gate(z))

Notation:
- ``B``: batch size
- ``T``: sequence length
- ``C``: embedding dimension (input/output)
- ``H``: hidden/intermediate dimension

Design notes:
- All blocks preserve the last dimension from ``C`` back to ``C``.
- Dropout is applied in hidden and/or output stage based on class design.
- Inputs may be any rank as long as the last dimension equals ``embed_dim``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FF_ReLU(nn.Module):
    """Two-layer transformer FFN with ReLU activation.

    Computation:
        y = Dropout(W2 * Dropout(ReLU(W1 * x)))

    Constructor args:
        embed_dim (int, required): Input/output feature size ``C``.
        hidden_dim (int, required): Intermediate size ``H`` (often 2x-8x ``C``).
        dropout (float, optional, default=0.0): Dropout probability in hidden and
            output projections.
        device (str or torch.device, optional, default='cpu'): Parameter device.
        dtype (torch.dtype, optional, default=torch.float32): Parameter dtype.

    Forward args:
        x (torch.Tensor): Shape ``(..., C)``.

    Returns:
        torch.Tensor: Shape ``(..., C)``.

    Example:
        >>> ff = FF_ReLU(embed_dim=512, hidden_dim=2048, dropout=0.1)
        >>> x = torch.randn(4, 32, 512)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([4, 32, 512])
    
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
    """Two-layer FFN with LeakyReLU nonlinearity.

    Use this when you want non-zero negative slope to reduce dead-neuron behavior.

    Constructor args:
        embed_dim (int, required): Input/output size ``C``.
        hidden_dim (int, required): Intermediate size ``H``.
        dropout (float, optional, default=0.0): Dropout probability.
        negative_slope (float, optional, default=0.1): LeakyReLU slope for
            negative inputs. Typical values: 1e-2 to 1e-1.
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): Shape ``(..., C)``.

    Returns:
        torch.Tensor: Shape ``(..., C)``.

    Example:
        >>> ff = FF_LeakyReLU(embed_dim=256, hidden_dim=1024, negative_slope=0.01)
        >>> x = torch.randn(2, 16, 256)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([2, 16, 256])
    
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
    """Two-layer FFN with GELU activation (common in BERT/GPT-style models).

    GELU form (conceptual): ``GELU(x) = x * Phi(x)``.

    Constructor args:
        embed_dim (int, required): Input/output size.
        hidden_dim (int, required): Intermediate size.
        dropout (float, optional, default=0.0).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): Shape ``(..., embed_dim)``.

    Returns:
        torch.Tensor: Shape ``(..., embed_dim)``.

    Example:
        >>> ff = FF_GELU(embed_dim=768, hidden_dim=3072)
        >>> x = torch.randn(2, 128, 768)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([2, 128, 768])
    
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
    """Two-layer FFN with Sigmoid activation.

    Suitable for experiments requiring bounded hidden activations in (0,1), but
    usually less preferred than GELU/SiLU in deep transformer stacks.

    Constructor args:
        embed_dim (int, required).
        hidden_dim (int, required).
        dropout (float, optional, default=0.0).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): Shape ``(..., embed_dim)``.

    Returns:
        torch.Tensor: Shape ``(..., embed_dim)``.

    Example:
        >>> ff = FF_Sigmoid(embed_dim=128, hidden_dim=512)
        >>> x = torch.randn(3, 20, 128)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([3, 20, 128])
    
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
    """Two-layer FFN with SiLU/Swish activation.

    SiLU formula: ``SiLU(x) = x * sigmoid(x)``.

    Constructor args:
        embed_dim (int, required).
        hidden_dim (int, required).
        dropout (float, optional, default=0.0).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): Shape ``(..., embed_dim)``.

    Returns:
        torch.Tensor: Shape ``(..., embed_dim)``.

    Example:
        >>> ff = FF_SiLU(embed_dim=512, hidden_dim=2048, dropout=0.1)
        >>> x = torch.randn(1, 64, 512)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([1, 64, 512])
    
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
    """Gated FFN with SwiGLU activation.

    Computation:
        [g, v] = split(W1*x)  # each in R^H
        h = v ⊙ SiLU(g)
        y = W2*h

    Constructor args:
        embed_dim (int, required): Input/output size ``C``.
        hidden_dim (int, required): Gated branch size ``H``.
        dropout (float, optional, default=0.0): Applied after gating and output.
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): Shape ``(..., C)``.

    Returns:
        torch.Tensor: Shape ``(..., C)``.

    Notes:
        ``linear1`` projects to ``2 * hidden_dim`` and is split into gate/value.

    Example:
        >>> ff = FF_SwiGLU(embed_dim=1024, hidden_dim=2048, dropout=0.1)
        >>> x = torch.randn(2, 32, 1024)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([2, 32, 1024])
    
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
    """Gated FFN with GeGLU activation (GELU-based gate).

    Computation:
        [g, v] = split(W1*x)
        h = v ⊙ GELU(g)
        y = W2*h

    Constructor args:
        embed_dim (int, required).
        hidden_dim (int, required).
        dropout (float, optional, default=0.0).
        device (optional, default='cpu').
        dtype (optional, default=torch.float32).

    Forward args:
        x (torch.Tensor): Shape ``(..., embed_dim)``.

    Returns:
        torch.Tensor: Shape ``(..., embed_dim)``.

    Example:
        >>> ff = FF_GeGLU(embed_dim=512, hidden_dim=1536)
        >>> x = torch.randn(2, 40, 512)
        >>> y = ff(x)
        >>> y.shape
        torch.Size([2, 40, 512])
    
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
