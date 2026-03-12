import pytest
import torch

from stackformer import GPT_1, GPT_2, gemma_1_2b, llama_1


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2, gemma_1_2b, llama_1])
def test_generation_shape_and_stability_single_and_batch(model_cls):
    torch.manual_seed(0)
    x = torch.randint(0, 32, (2, 6))
    model = model_cls(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)

    single = model.generate(x[0], max_context_len=6, max_new_tokens=3)
    batch = model.generate(x, max_context_len=6, max_new_tokens=2)

    assert single.shape == (1, 9)
    assert batch.shape == (2, 8)
    assert torch.isfinite(single.float()).all() and torch.isfinite(batch.float()).all()


@pytest.mark.parametrize("model_cls", [GPT_1, GPT_2])
def test_generation_is_deterministic_with_seed_reset(model_cls):
    x = torch.randint(0, 32, (1, 6))
    model = model_cls(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)

    torch.manual_seed(0)
    out1 = model.generate(x, max_context_len=6, max_new_tokens=3)
    torch.manual_seed(0)
    out2 = model.generate(x, max_context_len=6, max_new_tokens=3)

    assert torch.equal(out1, out2)
