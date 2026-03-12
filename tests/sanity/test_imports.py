def test_import_stackformer_and_public_api():
    import stackformer

    required = [
        "GPT_1",
        "GPT_2",
        "gemma_1_2b",
        "llama_1",
        "llama_2",
        "transformer",
        "Multi_Head_Attention",
        "RoPE",
        "text_generate",
    ]
    for name in required:
        assert hasattr(stackformer, name)
