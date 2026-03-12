import pytest
import torch

from stackformer import GPT_2, gemma_1_2b, llama_1

BATCH = 2
SEQ = 6
EMB = 8
HEADS = 2
VOCAB = 32


@pytest.mark.parametrize("model_cls", [GPT_2, gemma_1_2b, llama_1])
def test_generation_numerical_stability_single_and_batch(model_cls):
    torch.manual_seed(0)
    x = torch.randint(0, VOCAB, (BATCH, SEQ))

    model = model_cls(
        vocab_size=VOCAB,
        num_layers=1,
        embed_dim=EMB,
        num_heads=HEADS,
        seq_len=SEQ,
        hidden_dim=16,
        dropout=0.0,
    )

    logits = model(x)
    assert torch.isfinite(logits).all()

    single = model.generate(x[0], max_context_len=SEQ, max_new_tokens=2)
    batch = model.generate(x, max_context_len=SEQ, max_new_tokens=2)

    assert single.shape == (1, SEQ + 2)
    assert batch.shape == (BATCH, SEQ + 2)
    assert torch.isfinite(single.float()).all()
    assert torch.isfinite(batch.float()).all()
