import pytest
import torch

from stackformer import GPT_1, GPT_2, gemma_1_2b, llama_1
from stackformer.models.Meta import llama_2


@pytest.mark.parametrize("model_name", ["gpt1", "gpt2", "gemma", "llama1", "llama2"])
def test_model_save_and_load_roundtrip(tmp_path, model_name):
    torch.manual_seed(0)
    x = torch.randint(0, 32, (2, 6))

    if model_name == "gpt1":
        model = GPT_1(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
        new_model = GPT_1(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
    elif model_name == "gpt2":
        model = GPT_2(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
        new_model = GPT_2(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
    elif model_name == "gemma":
        model = gemma_1_2b(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
        new_model = gemma_1_2b(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
    elif model_name == "llama1":
        model = llama_1(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
        new_model = llama_1(vocab_size=32, num_layers=1, embed_dim=8, num_heads=2, seq_len=6, dropout=0.0, hidden_dim=16)
    else:
        model = llama_2(vocab_size=32, num_layers=1, embed_dim=8, num_query_heads=2, num_kv_heads=1, batch_size=2, kv_seq_len=6, hidden_dim=16, dropout=0.0)
        new_model = llama_2(vocab_size=32, num_layers=1, embed_dim=8, num_query_heads=2, num_kv_heads=1, batch_size=2, kv_seq_len=6, hidden_dim=16, dropout=0.0)

    model.eval()
    new_model.eval()

    before = model(x)
    path = tmp_path / f"{model_name}.pt"
    torch.save(model.state_dict(), path)
    new_model.load_state_dict(torch.load(path, map_location="cpu"))
    after = new_model(x)

    assert before.shape == after.shape
    assert torch.isfinite(after).all()
    assert torch.allclose(before, after)
