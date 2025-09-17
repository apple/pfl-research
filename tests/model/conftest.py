# Copyright © 2023-2024 Apple Inc.
"""
See the following pages for information about conftest.py files:
* https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions # pylint: disable=line-too-long
* https://docs.pytest.org/en/2.7.3/plugins.html?highlight=re
"""
from pathlib import Path

import numpy as np
import pytest

from pfl.hyperparam import NNTrainHyperParams
from pfl.internal.bridge import FrameworkBridgeFactory as bridges
from pfl.metrics import StringMetricName, get_overall_value


@pytest.fixture
def check_save_and_load_model_impl():

    def _check_save_and_load_model_impl(setup, load_model_path: Path):
        """ helper function for testing saving and loading of a model. """

        original_var_values = {
            k: setup.variable_to_numpy_fn(v)
            for k, v in setup.model.variable_map.items()
        }
        setup.model.save(str(setup.save_model_path))

        # modify variables.
        setup.model.set_parameters(
            setup.model.get_parameters().apply_elementwise(lambda x: x * 0))

        changed_var_values = {
            k: setup.variable_to_numpy_fn(v)
            for k, v in setup.model.variable_map.items()
        }

        # Make sure variables were changed before testing loading model.
        for name in setup.model.variable_map:
            assert np.any(
                np.not_equal(original_var_values[name],
                             changed_var_values[name]))

        setup.model.load(str(load_model_path))

        for name in setup.model.variable_map:
            np.testing.assert_array_equal(
                original_var_values[name],
                setup.variable_to_numpy_fn(setup.model.variable_map[name]))

    return _check_save_and_load_model_impl


@pytest.fixture
def check_save_and_load_central_optimizer_impl():

    def _check_save_and_load_central_optimizer_impl(setup):
        # Random central step.
        def one_central_optimizer_step():
            state = setup.model.get_parameters()
            bridges.sgd_bridge().do_sgd(
                setup.model, setup.user_dataset,
                NNTrainHyperParams(local_learning_rate=1,
                                   local_num_epochs=1,
                                   local_batch_size=None))
            model_update = setup.model.get_model_difference(state)
            setup.model.set_parameters(state)
            setup.model.apply_model_update(model_update)

        one_central_optimizer_step()
        setup.model.save(str(setup.save_model_path))

        # Make sure variables in central optimizer is initialized
        assert setup.model.central_optimizer_variable_map is not None
        original_optimizer_var_values = {
            k: setup.variable_to_numpy_fn(v)
            for k, v in setup.model.central_optimizer_variable_map.items()
        }

        one_central_optimizer_step()
        changed_optimizer_var_values = {
            k: setup.variable_to_numpy_fn(v)
            for k, v in setup.model.central_optimizer_variable_map.items()
        }
        # Make sure central optimizer variables were changed before testing loading.
        for k in setup.model.central_optimizer_variable_map:
            if k == 'learning_rate':
                continue
            assert np.any(
                np.not_equal(original_optimizer_var_values[k],
                             changed_optimizer_var_values[k]))

        setup.model.load(str(setup.load_model_path))
        # Make sure central optimizer variables were the same after loading.
        for k, v in setup.model.central_optimizer_variable_map.items():
            np.testing.assert_array_equal(original_optimizer_var_values[k],
                                          setup.variable_to_numpy_fn(v))
        one_central_optimizer_step()

    return _check_save_and_load_central_optimizer_impl


@pytest.fixture
def check_apply_model_update_impl():

    def _check_apply_model_update_impl(setup):
        """ Helper function for testing apply model updates to model. """
        # Some frameworks, e.g. Keras, averages the loss internally.
        divider = 2.0 if setup.reports_average_loss else 1.0

        base_weight = np.array([[2., 4.], [3., 5.]])

        apply_model_update_parameters_to_test = [
            (0, 1.0, [base_weight]),
            (1, 1.0, [base_weight + np.array([[1, 1], [2, 2]]) / divider]),
            (1, .1,
             [base_weight + np.array([[0.1, 0.1], [0.2, 0.2]]) / divider]),
            # Learning rate needs to be a float without inprecision
            # to be able to test convergence.
            (10, .125 * divider,
             [base_weight + np.array([[1.25, 1.25], [1., 1.]])]),
        ]

        model = setup.model
        state = model.get_parameters()

        for epochs, lr, expected_weights in apply_model_update_parameters_to_test:

            bridges.sgd_bridge().do_sgd(
                model, setup.user_dataset,
                NNTrainHyperParams(local_learning_rate=lr,
                                   local_num_epochs=epochs,
                                   local_batch_size=None))

            average_model_update = model.get_model_difference(state)
            model.set_parameters(state)

            # Confirm model_update need to be applied to weights.
            for name, expected_weight in zip(average_model_update.keys(),
                                             expected_weights):

                manually_updated_weight = setup.variable_to_numpy_fn(
                    model.variable_map[name]) + setup.variable_to_numpy_fn(
                        average_model_update[name])
                np.testing.assert_array_almost_equal(manually_updated_weight,
                                                     expected_weight)
            (_, metrics) = model.apply_model_update(average_model_update)

            assert get_overall_value(
                metrics[StringMetricName('learning rate')]) == 1

            # Confirm model_update has been applied to weights.
            for name, expected_weight in zip(average_model_update.keys(),
                                             expected_weights):
                np.testing.assert_array_almost_equal(
                    setup.variable_to_numpy_fn(model.variable_map[name]),
                    expected_weight)

            # Restore before next test case.
            model.set_parameters(state)

    return _check_apply_model_update_impl
