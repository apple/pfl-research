# Copyright © 2023-2024 Apple Inc.
"""
Test stateful differentiable models that yield MappedVectorStatistics.
"""

from typing import cast

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.common_types import Population
from pfl.hyperparam import NNTrainHyperParams
from pfl.internal.bridge import FrameworkBridgeFactory as bridges
from pfl.internal.ops.common_ops import check_mlx_installed, get_pytorch_major_version, get_tf_major_version
from pfl.internal.ops.selector import get_framework_module as get_ops
from pfl.metrics import MetricName, MetricValue, Weighted
from pfl.stats import MappedVectorStatistics

from ..conftest import ModelSetup

pytorch_pytest_param = pytest.param(lazy_fixture('pytorch_model_setup'),
                                    marks=[
                                        pytest.mark.skipif(
                                            not get_pytorch_major_version(),
                                            reason='PyTorch not installed')
                                    ],
                                    id='pytorch')
tf_pytest_param = pytest.param(
    lazy_fixture('tensorflow_model_setup'),
    marks=[pytest.mark.skipif(get_tf_major_version() < 2, reason='not tf>=2')],
    id='tensorflow')

mlx_pytest_param = pytest.param(lazy_fixture('mlx_model_setup'),
                                marks=[
                                    pytest.mark.skipif(
                                        not check_mlx_installed(),
                                        reason='MLX not installed')
                                ],
                                id='mlx')


@pytest.mark.parametrize('setup', [
    pytorch_pytest_param,
    tf_pytest_param,
    mlx_pytest_param,
])
class TestModel:
    """
    Contains all tests that are common to all models.

    The model_update test cases were generated manually using the snippet:
    ```
w = np.array([[2.,4.], [3.,5.]])
w_orig = w.copy()
x = np.array([[1.,0.], [0.,2.]])
t = np.array([[4.,6.], [8.,12.]])
lr = 1.0
epochs = 1
local_batch_size = 1
i_start = 0
for _ in range(epochs):
    x_batch = x[i_start:i_start + local_batch_size]
    t_batch = t[i_start:i_start + local_batch_size]
    i_start = (i_start + local_batch_size) % len(x)
    print(x_batch)
    y = np.dot(x_batch,w)
    print(y, t_batch)
    g_loss = np.ones_like(y)
    g_loss[y < t_batch] = -1 # Gradient of MAE
    w = w - lr * np.dot(x_batch.T,g_loss)

print('delta', w - w_orig)
print('final weight', w)
    ```
    """

    def _stats_to_ndarrays(self, stats):
        return list(stats.apply_elementwise(get_ops().to_numpy).values())

    def test_save_and_load_model(self, setup: ModelSetup,
                                 check_save_and_load_model_impl):
        check_save_and_load_model_impl(setup, setup.load_model_path)

    def test_get_and_set_parameters(self, setup: ModelSetup,
                                    check_equal_stats):

        def perform_sgd():
            # Random training to modify variables.
            bridges.sgd_bridge().do_sgd(
                setup.model, setup.user_dataset,
                NNTrainHyperParams(local_learning_rate=1,
                                   local_num_epochs=1,
                                   local_batch_size=None))

        model = setup.model
        stats_before_sgd = model.get_parameters()

        perform_sgd()

        stats_after_sgd = model.get_parameters()

        # Params should not be equal to stats_before_sgd
        with np.testing.assert_raises(AssertionError):
            check_equal_stats(stats_before_sgd, model.get_parameters(),
                              setup.variable_to_numpy_fn)
        model.set_parameters(stats_before_sgd)
        # Now params should be equal to stats_before_sgd
        check_equal_stats(stats_before_sgd, model.get_parameters(),
                          setup.variable_to_numpy_fn)

        # Params should not be equal for stats_after_sgd
        # because now the model stats have been set to stats_before_sgd
        with np.testing.assert_raises(AssertionError):
            check_equal_stats(stats_after_sgd, model.get_parameters(),
                              setup.variable_to_numpy_fn)
        model.set_parameters(stats_after_sgd)
        # Now params should be equal for stats_after_sgd
        check_equal_stats(stats_after_sgd, model.get_parameters(),
                          setup.variable_to_numpy_fn)

    def test_backup_and_model_diff_op(self, setup: ModelSetup):
        model = setup.model

        state = model.get_parameters()
        model_diff = model.get_model_difference(state)

        model_diff = cast(MappedVectorStatistics, model_diff)
        # Model difference should be zero because there was no training.
        assert len(model_diff) == 1
        assert np.sum(self._stats_to_ndarrays(model_diff)[0]) == 0

    def test_model_diff_clone(self, setup: ModelSetup):
        model = setup.model

        initial_state = model.get_parameters()
        model_diff1 = model.get_model_difference(initial_state)
        model.set_parameters(initial_state.apply_elementwise(lambda w: w + 1))

        model_diff2 = model.get_model_difference(initial_state, clone=True)

        model_diff1 = cast(MappedVectorStatistics, model_diff1)
        model_diff2 = cast(MappedVectorStatistics, model_diff2)
        np.testing.assert_array_equal(model_diff1[setup.variable_names[0]],
                                      np.zeros((2, 2)))
        np.testing.assert_array_equal(model_diff2[setup.variable_names[0]],
                                      np.ones((2, 2)))

    def test_restore(self, setup: ModelSetup, check_equal_stats):
        model = setup.model
        variables = setup.variables

        state = model.get_parameters()
        placeholders = model.get_parameters()
        backed_up_variable_values = [
            setup.variable_to_numpy_fn(v) for v in variables
        ]

        # modify variables.
        setup.model.set_parameters(
            setup.model.get_parameters().apply_elementwise(lambda x: x * 0))

        # Should get new state in-place and put in placeholders.
        placeholders = model.get_parameters(placeholders)
        check_equal_stats(placeholders, model.get_parameters(),
                          setup.variable_to_numpy_fn)

        # Restore and check if the same values as during the backup.
        model.set_parameters(state)
        restored_variable_values = [
            setup.variable_to_numpy_fn(v) for v in variables
        ]
        for backed_up_var, restored_var in zip(backed_up_variable_values,
                                               restored_variable_values):
            np.testing.assert_array_equal(backed_up_var, restored_var)

    def test_train_and_difference(self, setup: ModelSetup):

        # Some frameworks, e.g. Keras, averages the loss internally.
        divider = 2 if setup.reports_average_loss else 1

        parameters_to_test = [
            (0, 1.0, None, None, [np.zeros((2, 2))]),
            (1, 1.0, None, None, [np.array([[1, 1], [2, 2]]) / divider]),
            (1, .1, None, None, [np.array([[0.1, 0.1], [0.2, 0.2]]) / divider
                                 ]),
            # Learning rate needs to be a float without
            # imprecision to be able to test convergence.
            (10, .125 * divider, None, None,
             [np.array([[1.25, 1.25], [1., 1.]])]),
            # Batch size 1, should divide by 1 always.
            (1, 1.0, 1, None, [np.array([[1, 1], [2, 2]])]),
            # Batch size 1, num_steps 1.
            (None, 1.0, 1, 1, [np.array([[1, 1], [0, 0]])]),
        ]

        model = setup.model
        state = model.get_parameters()

        for (epochs, lr, local_batch_size, local_num_steps,
             expected_model_diff) in parameters_to_test:
            bridges.sgd_bridge().do_sgd(
                model, setup.user_dataset,
                NNTrainHyperParams(local_learning_rate=lr,
                                   local_num_epochs=epochs,
                                   local_batch_size=local_batch_size,
                                   local_num_steps=local_num_steps))
            model_diff = model.get_model_difference(state)

            model_diff = cast(MappedVectorStatistics, model_diff)
            for weight_diff, expected_weight_diff in zip(
                    self._stats_to_ndarrays(model_diff), expected_model_diff):
                np.testing.assert_array_almost_equal(weight_diff,
                                                     expected_weight_diff)

            # Restore before next test case.
            model.set_parameters(state)

    def test_evaluate(self, setup: ModelSetup):
        model = setup.model
        format_fn = lambda n: MetricName(n, Population.TEST)  # pytype: disable=wrong-arg-count # pylint: disable=line-too-long

        for local_batch_size in [None, 1, 2]:

            config = NNTrainHyperParams(local_learning_rate=1,
                                        local_num_epochs=1,
                                        local_batch_size=local_batch_size)

            metrics = model.evaluate(setup.user_dataset, format_fn, config)

            loss = cast(Weighted, metrics[format_fn('loss')])

            if isinstance(loss, Weighted):
                # The loss should be 4 (value 8 with a weight of 2).
                assert loss.weighted_value == 8
                assert loss.weight == 2
            assert loss.overall_value == 4

            summed_loss = cast(MetricValue, metrics[format_fn('loss2')])
            assert summed_loss.overall_value == 4
            # Not present for TensorFlowModel.
            if format_fn('user_avg_loss') in metrics:
                user_avg_loss = cast(Weighted,
                                     metrics[format_fn('user_avg_loss')])
                assert user_avg_loss.weighted_value == 4
                assert user_avg_loss.weight == 1
                assert user_avg_loss.overall_value == 4

    def test_apply_model_update(self, setup: ModelSetup,
                                check_apply_model_update_impl):
        check_apply_model_update_impl(setup)


@pytest.mark.parametrize(
    'setup', [pytorch_pytest_param, tf_pytest_param, mlx_pytest_param])
@pytest.mark.parametrize('local_max_grad_norm', [0.001, 0.01, 0.1, 1.0, 10.0])
def test_local_gradient_clipping(setup: ModelSetup,
                                 local_max_grad_norm: float):
    """ Helper function for testing local gradient clipping. """
    model = setup.model
    state = model.get_parameters()
    # Test with FedSGD and learning rate set to 1 so that the model
    # difference will be the same as the gradients.
    bridges.sgd_bridge().do_sgd(
        model, setup.user_dataset,
        NNTrainHyperParams(local_learning_rate=1,
                           local_num_epochs=1,
                           local_batch_size=None,
                           local_max_grad_norm=local_max_grad_norm))

    local_grad = model.get_model_difference(state)
    local_grad = cast(MappedVectorStatistics, local_grad)
    local_grad_norm = get_ops().to_numpy(get_ops().global_norm(
        local_grad.get_weights()[1], order=2))
    # Confirm local gradients are smaller than the norm bound.
    # Adding np.isclose because implementation of gradient clipping
    # in Tensorflow/PyTorch could result in gradient norm a bit larger than
    # the clipping bound
    assert local_grad_norm <= local_max_grad_norm or np.isclose(
        local_grad_norm, local_max_grad_norm, atol=1e-6)
