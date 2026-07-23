from tests._test_utils import _checkpoint


def test_import_stackformer_and_public_api():
    _checkpoint("test_import_stackformer_and_public_api starting import check")
    import stackformer

    required = [
        "GPT_1",
        "GPT_2",
        "gemma_1_2b",
        "llama_1",
        "llama_2",
        "Transformer",
        "Multi_Head_Attention",
        "RoPE",
        "text_generate",
    ]
    for name in required:
        _checkpoint("Checking stackformer export", symbol=name)
        assert hasattr(stackformer, name), f"Missing public API export: {name}"
