import pytest

from pfl.internal.ops import get_pytorch_major_version

if get_pytorch_major_version() > 0:
    import torch

    from pfl.internal.ops import pytorch_ops


@pytest.mark.skipif(not get_pytorch_major_version(),
                    reason='PyTorch not installed')
@pytest.mark.parametrize('amp_dtype', ['bfloat16', 'float16', 'float32', None])
@pytest.mark.parametrize('grad_scaling', [True, False])
def test_setup_amp(amp_dtype, grad_scaling):
    if isinstance(amp_dtype, str):
        amp_dtype = getattr(torch, amp_dtype)
    amp_context, grad_scaler = pytorch_ops.setup_amp(amp_dtype, grad_scaling)
    assert grad_scaler is None  # only enable on cuda
    if (amp_dtype is None or amp_dtype == torch.float32
            or amp_dtype == torch.float16):
        # float16 is not available on cpu
        assert amp_context is None
    else:
        assert amp_context.fast_dtype == amp_dtype
