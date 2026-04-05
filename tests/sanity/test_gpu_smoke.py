import pytest
import torch


@pytest.mark.gpu
def test_cuda_tensor_addition() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    a = torch.ones(8, device="cuda")
    b = torch.full((8,), 2.0, device="cuda")
    result = a + b

    assert result.is_cuda
    assert torch.allclose(result.cpu(), torch.full((8,), 3.0))
