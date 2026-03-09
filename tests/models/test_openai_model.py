import torch

from stackformer import GPT_1, GPT_2


def test_gpt_models_forward_and_generate():
    torch.manual_seed(0)
    vocab_size, seq_len, batch_size = 32, 6, 2

    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    for model_cls in (GPT_1, GPT_2):
        model = model_cls(
            vocab_size=vocab_size,
            num_layers=1,
            embed_dim=8,
            num_heads=2,
            seq_len=seq_len,
            dropout=0.0,
            hidden_dim=16,
        )
        y = model(x)
        assert y.shape == (batch_size, seq_len, vocab_size)

        out = model.generate(x[0], max_context_len=seq_len, max_new_tokens=2)
        assert out.shape[-1] == seq_len + 2
