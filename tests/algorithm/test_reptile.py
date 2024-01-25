# Copyright © 2023-2024 Apple Inc.
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.algorithm.base import PersonalizedNNAlgorithmParams
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.ops.common_ops import get_pytorch_major_version, get_tf_major_version
from pfl.metrics import TrainMetricName, get_overall_value


@pytest.fixture
def nn_algorithm_params_dict(request):
    kwargs = request.param if hasattr(request, 'param') else {}
    return kwargs


@pytest.fixture
def nn_eval_params():
    return NNEvalHyperParams(local_batch_size=8)


@pytest.fixture(scope='function')
def reptile_setup(new_event_loop, nn_algorithm_params_dict,
                  make_hyperparameter):
    from pfl.algorithm.reptile import Reptile
    algo = Reptile()
    algo_params = PersonalizedNNAlgorithmParams(
        **{
            'central_num_iterations': 3,
            'evaluation_frequency': 2,
            'train_cohort_size': make_hyperparameter(4),
            'val_cohort_size': 5,
            'val_split_fraction': 0.2,
            'min_train_size': 1,
            'min_val_size': 1,
            **nn_algorithm_params_dict,
        })
    seeds = iter(range(1337, 2000))
    # Apparently, RandomState.randint can't be mocked.
    # It is a read-only attribute.
    with patch.object(algo, '_random_state',
                      MagicMock(randint=lambda a, b, dtype: next(seeds))):
        yield {
            'algorithm': algo,
            'algorithm_params': algo_params,
        }


@pytest.mark.parametrize('model_setup', [
    pytest.param(lazy_fixture('pytorch_model_setup'),
                 marks=[
                     pytest.mark.skipif(not get_pytorch_major_version(),
                                        reason='PyTorch not installed')
                 ],
                 id='pytorch'),
    pytest.param(lazy_fixture('tensorflow_model_setup'),
                 marks=[
                     pytest.mark.skipif(get_tf_major_version() < 2,
                                        reason='not tf>=2')
                 ],
                 id='tensorflow')
])
class TestReptile:
    """
    The model update test cases were generated manually using the code:
    (assuming full-batch GD)
    ```
w = np.array([[2.,4.], [3.,5.]])
w_orig = w.copy()
x = np.array([[1.,0.]])
t = np.array([[4.,6.]])
lr = 0.25
epochs = 10
for _ in range(epochs):
    y = np.dot(x,w)
    loss_before = np.sum(np.abs(np.dot(x,w)-t))/len(x)
    g_loss = np.ones_like(y)
    g_loss[y < t] = -1 # Gradient of MAE
    # The 2nd term is the change from FedProx
    w = w - lr * np.dot(x.T,g_loss)
    loss_after = np.sum(np.abs(np.dot(x,w)-t))/len(x)
    print(f'delta={w - w_orig} final_weight={w}')
    print(f'losses=[{loss_before},{loss_after}]')
    ```
    """

    @pytest.mark.parametrize('do_evaluation,population',
                             [(True, Population.VAL), (True, Population.TRAIN),
                              (True, Population.VAL)])
    def test_simulate_one_user(self, do_evaluation, population, user_dataset,
                               reptile_setup, model_setup):

        nn_train_params = NNTrainHyperParams(local_num_epochs=10,
                                             local_learning_rate=0.25,
                                             local_batch_size=None,
                                             local_max_grad_norm=None)

        federated_averaging = reptile_setup['algorithm']
        central_context = CentralContext(
            current_central_iteration=0,
            do_evaluation=do_evaluation,
            cohort_size=5,
            population=population,
            model_train_params=nn_train_params,
            model_eval_params=NNEvalHyperParams(local_batch_size=8),
            algorithm_params=reptile_setup['algorithm_params'].static_clone(),
            seed=1338)

        model_update, metrics = federated_averaging.simulate_one_user(
            model_setup.model, user_dataset, central_context)

        # pytype: disable=duplicate-keyword-argument,wrong-arg-count
        if central_context.population == Population.TRAIN:
            # Should be training.
            assert len(model_update) == 1
            assert model_update.weight == 1
            np.testing.assert_array_equal(
                np.array([[2., 2.], [0., 0.]]),
                model_setup.variable_to_numpy_fn(
                    model_update[model_setup.variable_names[0]]))
            if central_context.do_evaluation:
                # Evaluation metrics from both before and after training.
                assert get_overall_value(
                    metrics[TrainMetricName('loss',
                                            central_context.population,
                                            after_training=False)]) == 4
                assert get_overall_value(
                    metrics[TrainMetricName('loss',
                                            central_context.population,
                                            after_training=True)]) == 0.0
                assert get_overall_value(metrics[TrainMetricName(
                    'loss',
                    central_context.population,
                    after_training=False,
                    local_partition='val')]) == 4
                assert get_overall_value(metrics[TrainMetricName(
                    'loss',
                    central_context.population,
                    after_training=True,
                    local_partition='val')]) == 4
        else:
            # No model update for val iteration.
            assert model_update is None
            if central_context.do_evaluation:
                assert get_overall_value(
                    metrics[TrainMetricName('loss',
                                            central_context.population,
                                            after_training=False)]) == 4
                assert get_overall_value(metrics[TrainMetricName(
                    'loss',
                    central_context.population,
                    after_training=False,
                    local_partition='val')]) == 4

        # pytype: enable=duplicate-keyword-argument,wrong-arg-count

        if not central_context.do_evaluation:
            assert len(metrics) == 0
