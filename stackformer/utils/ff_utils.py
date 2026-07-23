def hidden_dim_helper(
    embed_dim: int,
    ffn_multiplier: float = 4.0,
    multiple_of: int = 256,
) -> int:
    """
    Compute the recommended hidden dimension for SwiGLU/GeGLU FFNs.

    Gated feed-forward networks use three projection matrices instead of two.
    To keep the parameter count approximately equal to a standard Transformer
    FFN with ``hidden_dim = ffn_multiplier * embed_dim``, the hidden dimension
    is reduced by a factor of ``2/3``.

    Args:
        embed_dim (int):
            Model embedding dimension.
        ffn_multiplier (float, optional):
            Standard FFN expansion ratio. Default is ``4.0``.
        multiple_of (int, optional):
            Round the result up to a multiple of this value.
            Modern LLMs commonly use 64, 128, or 256 for better hardware
            utilization. Set to ``1`` to disable rounding.

    Returns:
        int:
            Recommended hidden dimension.

    Example:
        >>> swiglu_hidden_dim(4096)
        11008

        >>> swiglu_hidden_dim(768)
        2048

        >>> swiglu_hidden_dim(1024, multiple_of=1)
        2730
    """
    hidden_dim = int((2.0 / 3.0) * ffn_multiplier * embed_dim)

    if multiple_of > 1:
        hidden_dim = (
            (hidden_dim + multiple_of - 1)
            // multiple_of
            * multiple_of
        )

    return hidden_dim