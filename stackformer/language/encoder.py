# BERT/RoBERTa encoder-only language modules

import torch
import torch.nn as nn

from stackformer.modules.layer import BlockConfig, TransformerEncoder
from stackformer.modules.position_embedding import AbsolutePositionEmbedding


#  BERT
class BERT(nn.Module):
    """BERT style encoder-only model.

    Simple explanation:
        Bidirectional encoder that reads the entire input sequence at once.
        Pre-trained with Masked Language Modeling (MLM) and Next Sentence
        Prediction (NSP), then fine-tuned on downstream tasks.

    Architecture details:
        - Attention:         Multi-Head Attention (MHA), no causal mask.
        - Masking:           Padding mask only (bidirectional).
        - Position encoding: Learned absolute positional embeddings.
        - Segment encoding:  Learned token-type (sentence A/B) embeddings.
        - Feed-forward:      GELU MLP.
        - Normalization:     Post-LayerNorm (pre_norm=False).
        - QKV bias:          True.
        - Weight tying:      LM-head weights shared with token embeddings.

    Research context:
        - Family: BERT encoder-only models.
        - Key idea: bidirectional context from MLM gives richer representations
          than unidirectional LMs for classification and span tasks.
        - Paper: https://arxiv.org/abs/1810.04805
    """

    def __init__(
        self,
        vocab_size:  int,
        num_layers:  int,
        embed_dim:   int,
        num_heads:   int,
        seq_len:     int,
        dropout:     float       = 0.1,
        hidden_dim:  int         = 0,         # 0 → 4 × embed_dim inside BlockConfig
        num_segments: int        = 2,         # sentence A / sentence B
        qkv_bias:    bool        = True,
        eps:         float       = 1e-5,
        device:      str         = "cpu",
        dtype:       torch.dtype = torch.float32,
    ):
        super().__init__()
        self.seq_len = seq_len

        cfg = BlockConfig(
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            hidden_dim = hidden_dim,
            attention  = "mha",
            ffn        = "gelu",
            norm       = "layernorm",
            mask_type  = ["no"],      # no causal mask — bidirectional
            pre_norm   = False,       # post-LayerNorm, as in original BERT
            dropout    = dropout,
            qkv_bias   = qkv_bias,
            device     = device,
            dtype      = dtype,
        )

        # ── Embeddings ──────────────────────────────────────────────────────
        self.token_embedding    = nn.Embedding(vocab_size,   embed_dim, device=device, dtype=dtype)
        self.position_embedding = AbsolutePositionEmbedding(seq_len, embed_dim, device=device, dtype=dtype)
        self.segment_embedding  = nn.Embedding(num_segments, embed_dim, device=device, dtype=dtype)
        self.embed_norm         = nn.LayerNorm(embed_dim, eps=eps, device=device, dtype=dtype)
        self.embed_dropout      = nn.Dropout(dropout)

        # ── Transformer backbone ─────────────────────────────────────────────
        # pos_embedding=None because we handle positional + segment in forward
        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)

        # ── MLM head ────────────────────────────────────────────────────────
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, device=device, dtype=dtype)
        # Weight tying: share token-embedding matrix with the MLM projection
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids:      torch.Tensor,                    # (B, T)  token ids
        token_type_ids: torch.Tensor | None = None,      # (B, T)  0 = sent A, 1 = sent B
    ) -> torch.Tensor:                                   # (B, T, vocab_size)

        B, T = input_ids.shape

        # Default to sentence A (all zeros) when token_type_ids not provided
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Sum of three embeddings (standard BERT input representation)
        x  = self.token_embedding(input_ids)            # (B, T, D)
        x  = x + self.position_embedding(input_ids)     # + learned absolute position
        x  = x + self.segment_embedding(token_type_ids) # + sentence A/B segment
        x  = self.embed_norm(x)
        x  = self.embed_dropout(x)

        # Bidirectional self-attention — NO causal mask
        x = self.backbone(x, mask=False)

        return self.lm_head(x)                          # (B, T, vocab_size)


#  RoBERTa
class RoBERTa(nn.Module):
    """RoBERTa style encoder-only model.

    Simple explanation:
        A robustly optimised BERT. Removes NSP, uses dynamic masking,
        trains longer on more data with larger batches, and swaps
        segment embeddings for a single-sentence embedding only.

    Architecture details vs BERT:
        - Segment embeddings:  REMOVED (no NSP task, no sentence-pair embedding).
        - Masking strategy:    Dynamic masking applied per-batch (done in the
                               data pipeline, not inside the model).
        - Tokenizer:           Byte-Pair Encoding (BPE) instead of WordPiece —
                               handled outside the model.
        - Position encoding:   Learned absolute positional embeddings (same as
                               BERT), but offset by padding_idx so that
                               positions start at 1 (Fairseq convention).
        - Feed-forward:        GELU MLP.
        - Normalization:       Post-LayerNorm (pre_norm=False).
        - QKV bias:            True.
        - Weight tying:        LM-head weights shared with token embeddings.

    Key differences from BERT (summary):
        ┌─────────────────────┬──────────────────┬──────────────────┐
        │ Feature             │ BERT             │ RoBERTa          │
        ├─────────────────────┼──────────────────┼──────────────────┤
        │ Pre-training tasks  │ MLM + NSP        │ MLM only         │
        │ Segment embeddings  │ ✓ (2 segments)   │ ✗                │
        │ Masking             │ Static           │ Dynamic          │
        │ Tokenizer           │ WordPiece        │ BPE              │
        │ Training data       │ 16 GB            │ 160 GB           │
        │ Batch size          │ 256              │ 8192             │
        └─────────────────────┴──────────────────┴──────────────────┘

    Research context:
        - Family: BERT encoder-only models.
        - Key idea: BERT was significantly under-trained; more data, dynamic
          masking, and removing NSP yield consistent gains across GLUE/SQuAD.
        - Paper: https://arxiv.org/abs/1907.11692
    """

    def __init__(
        self,
        vocab_size:   int,
        num_layers:   int,
        embed_dim:    int,
        num_heads:    int,
        seq_len:      int,
        dropout:      float       = 0.1,
        hidden_dim:   int         = 0,        # 0 → 4 × embed_dim
        padding_idx:  int         = 1,        # <pad> token id (Fairseq convention)
        qkv_bias:     bool        = True,
        eps:          float       = 1e-5,
        device:       str         = "cpu",
        dtype:        torch.dtype = torch.float32,
    ):
        super().__init__()
        self.seq_len     = seq_len
        self.padding_idx = padding_idx

        cfg = BlockConfig(
            embed_dim  = embed_dim,
            num_heads  = num_heads,
            hidden_dim = hidden_dim,
            attention  = "mha",
            ffn        = "gelu",
            norm       = "layernorm",
            mask_type  = ["no"],      # no causal mask — bidirectional
            pre_norm   = False,       # post-LayerNorm
            dropout    = dropout,
            qkv_bias   = qkv_bias,
            device     = device,
            dtype      = dtype,
        )

        # ── Embeddings ──────────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim,
            padding_idx = padding_idx,        # pad token gets zero embedding
            device=device, dtype=dtype,
        )
        # RoBERTa: positions offset by padding_idx (Fairseq convention)
        # max position = seq_len + padding_idx
        self.position_embedding = AbsolutePositionEmbedding(
            seq_len + padding_idx, embed_dim, device=device, dtype=dtype
        )
        # NOTE: No segment_embedding — RoBERTa removes NSP and sentence-pair embeddings

        self.embed_norm    = nn.LayerNorm(embed_dim, eps=eps, device=device, dtype=dtype)
        self.embed_dropout = nn.Dropout(dropout)

        # ── Transformer backbone ─────────────────────────────────────────────
        self.backbone = TransformerEncoder(cfg, num_layers=num_layers, pos_embedding=None)

        # ── MLM head ────────────────────────────────────────────────────────
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False, device=device, dtype=dtype)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    def _create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Fairseq-style position ids: non-padding tokens get ids starting
        from padding_idx + 1; padding tokens get padding_idx.

        Example (padding_idx=1):
            input_ids  = [5, 12, 1, 1]  (1 = <pad>)
            position   = [2,  3, 1, 1]
        """
        mask = input_ids.ne(self.padding_idx).int()      # 1 for real tokens
        return (mask.cumsum(dim=-1) + self.padding_idx) * mask + self.padding_idx * (1 - mask)

    def forward(
        self,
        input_ids: torch.Tensor,    # (B, T) token ids
    ) -> torch.Tensor:              # (B, T, vocab_size)

        position_ids = self._create_position_ids(input_ids)  # (B, T)

        x  = self.token_embedding(input_ids)                 # (B, T, D)
        x  = x + self.position_embedding(position_ids)       # + learned absolute position
        # No segment embedding in RoBERTa
        x  = self.embed_norm(x)
        x  = self.embed_dropout(x)

        # Bidirectional self-attention — NO causal mask
        x = self.backbone(x, mask=False)

        return self.lm_head(x)                               # (B, T, vocab_size)