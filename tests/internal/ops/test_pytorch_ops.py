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


@pytest.mark.skipif(not get_pytorch_major_version(),
                    reason='PyTorch not installed')
@pytest.mark.parametrize('local_rank,expected', [(None, 'cuda'),
                                                 ('0', 'cuda:0'),
                                                 ('1', 'cuda:1')])
def test_get_default_device_binds_each_rank_to_its_own_gpu(
        local_rank, expected, monkeypatch):
    """
    `torch.device('cuda')` is device 0 in every process, so without `LOCAL_RANK`
    a `torchrun` launch puts the whole job on one GPU while still reporting a
    world size that says otherwise.
    """
    monkeypatch.delenv('PFL_PYTORCH_DEVICE', raising=False)
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    if local_rank is not None:
        monkeypatch.setenv('LOCAL_RANK', local_rank)
    monkeypatch.setattr(pytorch_ops, 'is_pytest_running', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert str(pytorch_ops.get_default_device()) == expected


@pytest.mark.skipif(not get_pytorch_major_version(),
                    reason='PyTorch not installed')
def test_get_default_device_prefers_the_explicit_override(monkeypatch):
    monkeypatch.setenv('PFL_PYTORCH_DEVICE', 'cpu')
    monkeypatch.setenv('LOCAL_RANK', '1')
    monkeypatch.setattr(pytorch_ops, 'is_pytest_running', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    assert str(pytorch_ops.get_default_device()) == 'cpu'
