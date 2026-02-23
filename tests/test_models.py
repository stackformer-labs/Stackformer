import torch

from stackformer import GPT_1, GPT_2, gemma_1_2b, llama_1
from stackformer.models.Meta import llama_2
from stackformer.models.Transformer import transformer


def test_import_stackformer() -> None:
    import stackformer

    assert hasattr(stackformer, "GPT_2")


def test_small_model_forward_and_generation() -> None:
    vocab_size = 32
    seq_len = 8
    x = torch.randint(0, vocab_size, (2, seq_len))

    gpt1 = GPT_1(vocab_size=vocab_size, num_layers=1, embed_dim=4, num_heads=4, seq_len=seq_len, dropout=0.0, hidden_dim=8)
    gpt2 = GPT_2(vocab_size=vocab_size, num_layers=1, embed_dim=4, num_heads=4, seq_len=seq_len, dropout=0.0, hidden_dim=8)
    l1 = llama_1(vocab_size=vocab_size, num_layers=1, embed_dim=4, num_heads=4, seq_len=seq_len, dropout=0.0, hidden_dim=8)
    g12 = gemma_1_2b(vocab_size=vocab_size, num_layers=1, embed_dim=4, num_heads=4, seq_len=seq_len, dropout=0.0, hidden_dim=8)

    for model in (gpt1, gpt2, l1, g12):
        y = model(x)
        assert y.shape == (2, seq_len, vocab_size)

        out = model.generate(x[0], max_context_len=seq_len, max_new_tokens=2)
        assert out.dim() == 2
        assert out.size(-1) == seq_len + 2


def test_llama2_and_transformer_forward() -> None:
    vocab_size = 32
    seq_len = 8
    x = torch.randint(0, vocab_size, (2, seq_len))

    l2 = llama_2(
        num_layers=1,
        embed_dim=4,
        num_query_heads=4,
        num_kv_heads=2,
        batch_size=2,
        kv_seq_len=seq_len,
        vocab_size=vocab_size,
        hidden_dim=8,
        dropout=0.0,
    )
    y = l2(x)
    assert y.shape == (2, seq_len, vocab_size)

    seq2seq = transformer(
        vocab_size=vocab_size,
        embed_dim=4,
        num_heads=4,
        dropout=0.0,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        seq_len=seq_len,
    )
    y2 = seq2seq(x, x)
    assert y2.shape == (2, seq_len, vocab_size)
