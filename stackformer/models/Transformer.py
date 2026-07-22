"""Original Transformer (encoder-decoder) model implementation.

Rebuilt on top of the Stackformer block builders
(BlockConfig + TransformerEncoder + TransformerDecoder).

Contains a sequence-to-sequence Transformer aligned with the 2017 "Attention
Is All You Need" baseline:
    - Sinusoidal positional embeddings
    - ReLU feed-forward blocks
    - Post-LayerNorm
    - Bidirectional encoder, causal + cross-attention decoder

Paper: https://arxiv.org/abs/1706.03762
"""

import torch
import torch.nn as nn

from stackformer.modules.layer import BlockConfig, TransformerEncoder, TransformerDecoder


class transformer(nn.Module):
    """Vaswani et al. style encoder-decoder Transformer.

    Simple explanation:
        This model has two parts: an encoder that reads the source sequence
        in full (bidirectional) and a decoder that generates a target sequence
        token-by-token, attending to both its own past outputs (causal
        self-attention) and the full encoder output (cross-attention).

    Architecture details:
        - Encoder attention:  Multi-Head self-attention, no causal mask.
        - Decoder attention:  Causal MHA self-attention + cross-attention.
        - Position encoding:  Sinusoidal embeddings (fixed, no learned params).
        - Feed-forward:       ReLU MLP.
        - Normalization:      Post-LayerNorm  (pre_norm=False, original paper).

    Research context:
        - Canonical source: "Attention Is All You Need", Vaswani et al. 2017.
        - Importance: replaced RNN/CNN sequence models and enabled the scaling
          era of large language models.
        - Paper: https://arxiv.org/abs/1706.03762

    Example::

        import torch
        from stackformer.models.Transformer import transformer

        model = transformer(
            vocab_size=32000,
            embed_dim=512,
            num_heads=8,
            dropout=0.1,
            hidden_dim=2048,
            encoder_layers=6,
            decoder_layers=6,
            seq_len=128,
        )
        src    = torch.randint(0, 32000, (2, 40))
        tgt    = torch.randint(0, 32000, (2, 30))
        logits = model(src, tgt)                    # (2, 30, 32000)
    """

    def __init__(
        self,
        vocab_size:     int,
        embed_dim:      int,
        num_heads:      int,
        dropout:        float       = 0.1,
        hidden_dim:     int         = 0,     # 0 → 4 × embed_dim
        encoder_layers: int         = 6,
        decoder_layers: int         = 6,
        seq_len:        int         = 512,
        eps:            float       = 1e-5,
        device:         str         = "cpu",
        dtype:          torch.dtype = torch.float32,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        cfg = BlockConfig(
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            hidden_dim = hidden_dim,
            attention  = "mha",
            ffn        = "relu",
            norm       = "layernorm",
            pre_norm   = False,   # ← Post-LN: original 2017 paper
            dropout    = dropout,
            device     = device,
            dtype      = dtype,
        )

        self.token_emb = nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)

        # Encoder: bidirectional (mask=False) + sinusoidal PE
        self.encoder_stack = TransformerEncoder(
            cfg,
            num_layers    = encoder_layers,
            pos_embedding = "sinusoidal",
            max_seq_len   = seq_len,
        )

        # Decoder: causal self-attn + cross-attn + sinusoidal PE
        self.decoder_stack = TransformerDecoder(
            cfg,
            num_layers    = decoder_layers,
            pos_embedding = "sinusoidal",
            max_seq_len   = seq_len,
        )

        self.out_proj = nn.Linear(embed_dim, vocab_size, device=device, dtype=dtype)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        """Encode source tokens → memory  (B, S, C)."""
        return self.encoder_stack(self.token_emb(src), mask=False)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Decode target tokens given encoder memory  (B, T, C)."""
        return self.decoder_stack(self.token_emb(tgt), memory)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        memory = self.encode(source)              # (B, S, C)
        out    = self.decode(target, memory)      # (B, T, C)
        return self.out_proj(out)                 # (B, T, vocab_size)