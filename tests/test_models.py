import torch

from stackformer import GPT_1, GPT_2, gemma_1_2b, llama_1
from stackformer.models.Meta import llama_2
from stackformer.models.Transformer import transformer


def test_import_stackformer():
    import stackformer
    assert hasattr(stackformer, "GPT_2")


def test_small_model_forward_and_generation():
    torch.manual_seed(0)

    vocab_size = 32
    seq_len = 6
    batch_size = 2
    embed_dim = 8       # must allow even head_dim
    num_heads = 2       # 8 / 2 = 4 (even) ✅
    hidden_dim = 16

    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    gpt1 = GPT_1(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=embed_dim,
        num_heads=num_heads,
        seq_len=seq_len,
        dropout=0.0,
        hidden_dim=hidden_dim,
    )

    gpt2 = GPT_2(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=embed_dim,
        num_heads=num_heads,
        seq_len=seq_len,
        dropout=0.0,
        hidden_dim=hidden_dim,
    )

    l1 = llama_1(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=embed_dim,
        num_heads=num_heads,
        seq_len=seq_len,
        dropout=0.0,
        hidden_dim=hidden_dim,
    )

    g12 = gemma_1_2b(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=embed_dim,
        num_heads=num_heads,
        seq_len=seq_len,
        dropout=0.0,
        hidden_dim=hidden_dim,
    )

    for model in (gpt1, gpt2, l1, g12):
        y = model(x)
        assert y.shape == (batch_size, seq_len, vocab_size)

        # generation test (single sample)
        out = model.generate(x[0], max_context_len=seq_len, max_new_tokens=2)
        assert out.shape[-1] == seq_len + 2


def test_llama2_and_transformer_forward():
    torch.manual_seed(0)

    vocab_size = 32
    seq_len = 6
    batch_size = 2
    embed_dim = 8
    query_heads = 2
    kv_heads = 1        # 8 / 2 = 4 head_dim (even) ✅
    hidden_dim = 16

    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    l2 = llama_2(
        vocab_size=vocab_size,
        num_layers=1,
        embed_dim=embed_dim,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        kv_seq_len=seq_len,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        dropout=0.0,
    )

    y = l2(x)
    assert y.shape == (batch_size, seq_len, vocab_size)

    seq2seq = transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=query_heads,
        dropout=0.0,
        hidden_dim=hidden_dim,
        encoder_layers=1,
        decoder_layers=1,
        seq_len=seq_len,
    )

    y2 = seq2seq(x, x)
    assert y2.shape == (batch_size, seq_len, vocab_size)
