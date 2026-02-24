"""Original Transformer (encoder-decoder) model implementation.

Contains a sequence-to-sequence Transformer aligned with the 2017 baseline
architecture and enhanced with simple, practical documentation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F 

from stackformer.modules.Attention import Multi_Head_Attention, Cross_MultiHead_Attention
from stackformer.modules.position_embedding import SinusoidalPositionalEmbedding
from stackformer.modules.Feed_forward import FF_ReLU
from stackformer.modules.Normalization import LayerNormalization

class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_Head_Attention(embed_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm1 = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)
        self.ff_relu = FF_ReLU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm2 = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)
        
    def forward(self, x):
        residual = x
        x = self.attention(x, mask = False)
        x = self.norm1(x)
        x = x + residual 
        
        residual = x
        x = self.ff_relu(x)
        x = self.norm2(x)
        x = x + residual  
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim, eps=1e-5, device='cpu', dtype=torch.float32):
        super().__init__()
        self.attention = Multi_Head_Attention(embed_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm1 = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)
        self.cross_attention = Cross_MultiHead_Attention(embed_dim, num_heads, dropout, device=device, dtype=dtype)
        self.norm2 = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)
        self.ff_relu = FF_ReLU(embed_dim, hidden_dim, dropout, device=device, dtype=dtype)
        self.norm3 = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)
        
    def forward(self, x, enc_output):
        residual = x
        x = self.attention(x)
        x = self.norm1(x)
        x = x + residual 
        
        residual = x
        x = self.cross_attention(x, context = enc_output)
        x = self.norm2(x)
        x = x + residual
        
        residual = x
        x = self.ff_relu(x)
        x = self.norm3(x)
        x = x + residual  
        
        return x
        
class transformer(nn.Module):
    """Vaswani et al. style encoder-decoder Transformer.

    Simple explanation:
        This model has two parts: an encoder that reads the source sequence and
        a decoder that generates a target sequence conditioned on encoder output.
        It is useful for translation, summarization, and general seq2seq tasks.

    Architecture details:
        - Backbone: Encoder-Decoder Transformer.
        - Encoder attention: Multi-Head self-attention (non-causal).
        - Decoder attention: causal self-attention + cross-attention to encoder.
        - Position encoding: Sinusoidal positional embedding.
        - Feed-forward: ReLU MLP blocks.
        - Normalization: LayerNorm.

    Research context:
        - Canonical source: “Attention Is All You Need” (2017).
        - Importance: replaced recurrent/convolutional sequence models in many
          NLP pipelines and enabled scaling.
        - Paper/report: TODO (add link manually).

    Example:
        >>> import torch
        >>> from stackformer.models.Transformer import transformer
        >>> model = transformer(
        ...     vocab_size=32000,
        ...     embed_dim=512,
        ...     num_heads=8,
        ...     dropout=0.1,
        ...     hidden_dim=2048,
        ...     encoder_layers=6,
        ...     decoder_layers=6,
        ...     seq_len=128,
        ... )
        >>> src = torch.randint(0, 32000, (2, 40))
        >>> tgt = torch.randint(0, 32000, (2, 30))
        >>> logits = model(src, tgt)
    """
    def __init__(self, vocab_size, embed_dim, num_heads, dropout, hidden_dim, 
                encoder_layers, decoder_layers, seq_len, eps=1e-5, device='cpu', dtype=torch.float32,
                ):
        super().__init__()
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        
        self.token_emb = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        self.pos = SinusoidalPositionalEmbedding(seq_len=seq_len, embed_dim=embed_dim)
        
        self.encoder_stack = nn.ModuleList([
            Encoder(embed_dim, num_heads, dropout, hidden_dim, eps=eps, device=device, dtype=dtype)
            for _ in range(encoder_layers)
        ])
        
        self.decoder_stack = nn.ModuleList([
            Decoder(embed_dim, num_heads, dropout, hidden_dim, eps=eps, device=device, dtype=dtype)
            for _ in range(decoder_layers)
        ])
        
        # --- final norm ---
        self.final_norm = LayerNormalization(embed_dim, eps=eps, device=device, dtype=dtype)
        
        # --- output projection ---
        self.out_proj = nn.Linear(embed_dim, vocab_size, device=device, dtype=dtype)
        
    def encoder(self, x):
        x = self.token_emb(x) + self.pos(x)
        for block in self.encoder_stack:
            x = block(x)
        return x
    
    def decoder(self, x, enc_output):
        x = self.token_emb(x) + self.pos(x)
        for block in self.decoder_stack:
            x = block(x, enc_output)
        return x
    
    def forward(self, source, target):
        enc_output = self.encoder(source)
        out = self.decoder(target, enc_output)
        out = self.final_norm(out)
        out = self.out_proj(out)
        return out