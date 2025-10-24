# Copyright © 2023-2024 Apple Inc.
import numpy as np
import pytest
from pytest_lazy_fixtures import lf

from pfl.internal.ops.common_ops import get_pytorch_major_version, get_tf_major_version
from pfl.model.ema import CentralExponentialMovingAverage
from pfl.stats import MappedVectorStatistics

from ..conftest import ModelSetup


def _to_numpy(variable, variable_to_numpy_fn):
    if isinstance(variable, np.ndarray):
        return variable
    return variable_to_numpy_fn(variable)


def _get_variable_map(model):
    assert hasattr(model, "variable_map")
    return model.variable_map


def _cmp_model_and_ema(model, ema, variable_to_numpy_fn, cmp_fn):
    for variable_name in ema.shadow_variable_map:
        variable = _get_variable_map(model)[variable_name]
        ema_variable = ema.shadow_variable_map[variable_name]
        assert np.all(
            cmp_fn(_to_numpy(variable, variable_to_numpy_fn),
                   _to_numpy(ema_variable, variable_to_numpy_fn)))


def _get_fake_update(model, variable_to_numpy_fn, value):
    return MappedVectorStatistics({
        variable_name:
        np.ones_like(_to_numpy(variable, variable_to_numpy_fn) * value)
        for variable_name, variable in _get_variable_map(model).items()
    })


@pytest.mark.parametrize('setup', [
    pytest.param(lf('pytorch_model_setup'),
                 marks=[
                     pytest.mark.skipif(not get_pytorch_major_version(),
                                        reason='PyTorch not installed')
                 ],
                 id='pytorch'),
    pytest.param(lf('tensorflow_model_setup'),
                 marks=[
                     pytest.mark.skipif(get_tf_major_version() < 2,
                                        reason='not tf>=2')
                 ],
                 id='tensorflow'),
])
class TestCentralExponentialMovingAverage:

    def test_assign(self, setup: ModelSetup):
        model = setup.model
        ema = CentralExponentialMovingAverage(model, 0.9)
        model.apply_model_update(
            _get_fake_update(model, setup.variable_to_numpy_fn, value=1.0))
        _cmp_model_and_ema(model, ema, setup.variable_to_numpy_fn,
                           np.not_equal)
        ema.assign()
        _cmp_model_and_ema(model, ema, setup.variable_to_numpy_fn, np.equal)

    @pytest.mark.parametrize('decay', [0.5, 0.8, 0.9, 0.99, 0.999, 0.9999])
    @pytest.mark.parametrize('dynamic_decay_rate', [False, True])
    def test_update(self, setup: ModelSetup, decay, dynamic_decay_rate):
        model = setup.model
        ema = CentralExponentialMovingAverage(model, decay, dynamic_decay_rate)
        expected_ema_values = {
            variable_name: _to_numpy(variable, setup.variable_to_numpy_fn)
            for variable_name, variable in _get_variable_map(model).items()
        }

        # fake model updates
        model_update_to_apply = [
            _get_fake_update(model, setup.variable_to_numpy_fn, value=1.0),
            _get_fake_update(model, setup.variable_to_numpy_fn, value=2.0),
            _get_fake_update(model, setup.variable_to_numpy_fn, value=4.0),
            _get_fake_update(model, setup.variable_to_numpy_fn, value=8.0),
            _get_fake_update(model, setup.variable_to_numpy_fn, value=16.0),
        ]

        for model_update in model_update_to_apply:
            model.apply_model_update(model_update)
            decay = ema.decay
            ema.update()
            for variable_name, variable in _get_variable_map(model).items():
                ema_value = expected_ema_values[variable_name]
                value = _to_numpy(variable, setup.variable_to_numpy_fn)
                # ema updates by interpolating with current model
                expected_ema_values[variable_name] = decay * ema_value + (
                    1 - decay) * value
                ema_variable = ema.shadow_variable_map[variable_name]
                assert np.allclose(expected_ema_values[variable_name],
                                   setup.variable_to_numpy_fn(ema_variable))
            # test ema assign with model get and set parameters.
            state = model.get_parameters()
            ema.assign()
            _cmp_model_and_ema(model, ema, setup.variable_to_numpy_fn,
                               np.equal)
            model.set_parameters(state)
            _cmp_model_and_ema(model, ema, setup.variable_to_numpy_fn,
                               np.not_equal)
