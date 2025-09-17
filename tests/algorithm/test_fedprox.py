# Copyright © 2023-2024 Apple Inc.
import numpy as np
import pytest
from pytest_lazy_fixtures import lf

from pfl.algorithm.fedprox import AdaptMuOnMetricCallback, FedProx, FedProxParams
from pfl.common_types import Population
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.ops.common_ops import get_pytorch_major_version, get_tf_major_version
from pfl.metrics import Metrics, TrainMetricName, get_overall_value
from pfl.stats import MappedVectorStatistics


@pytest.fixture
def nn_eval_params():
    return NNEvalHyperParams(local_batch_size=8)


@pytest.fixture
def nn_train_params():
    return NNTrainHyperParams(local_num_epochs=2,
                              local_learning_rate=0.5,
                              local_batch_size=None,
                              local_max_grad_norm=None)


@pytest.fixture
def make_fedprox_setup():

    def make(mu):
        fedprox = FedProx()

        fedprox_params = FedProxParams(central_num_iterations=2,
                                       evaluation_frequency=2,
                                       train_cohort_size=2,
                                       val_cohort_size=None,
                                       mu=mu)
        return fedprox, fedprox_params

    return make


@pytest.fixture
def make_adafedprox_setup():

    def make(mu):
        fedprox = FedProx()

        adaptive_mu = AdaptMuOnMetricCallback(metric_name='meatballs',
                                              adapt_frequency=2,
                                              initial_value=mu)
        fedprox_params = FedProxParams(
            mu=adaptive_mu,
            central_num_iterations=2,
            evaluation_frequency=2,
            train_cohort_size=2,
            val_cohort_size=None,
        )
        return fedprox, fedprox_params

    return make


model_setup_marks = [
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
                 id='keras_v2')
]


@pytest.mark.parametrize('model_setup', model_setup_marks)
class TestFedProx:
    """
    The model update test cases were generated manually using the code:
    (assuming full-batch GD)
    ```
w = np.array([[2.,4.], [3.,5.]])
w_orig = w.copy()
x = np.array([[1.,0.], [0.,2.]])
t = np.array([[4.,6.], [8.,12.]])
lr = 0.5
mu = 1.0
epochs = 2
for _ in range(epochs):
    y = np.dot(x,w)
    loss_before = np.sum(np.abs(np.dot(x,w)-t))/len(x)
    g_loss = np.ones_like(y)
    g_loss[y < t] = -1 # Gradient of MAE
    # The 2nd term is the change from FedProx
    w = w - lr * (1/2*np.dot(x.T,g_loss) + mu*np.abs(w-w_orig))
    loss_after = np.sum(np.abs(np.dot(x,w)-t))/len(x)
    print(f'delta={w - w_orig} final_weight={w}'
    print(f'losses=[{loss_before},{loss_after}]')
    ```
    """

    @pytest.mark.parametrize('make_algorithm_setup', [
        lf('make_fedprox_setup'),
        lf('make_adafedprox_setup')
    ])
    @pytest.mark.parametrize('mu,expected_weight_diff, expected_loss', [
        (0.0, np.array([[0.5, 0.5], [1., 1.]]), 1.5),
        (1.0, np.array([[0.375, 0.375], [0.75, 0.75]]), 2.125),
    ])
    @pytest.mark.parametrize('do_evaluation,population',
                             [(True, Population.VAL), (True, Population.TRAIN),
                              (True, Population.VAL)])
    def test_simulate_one_user(self, model_setup, do_evaluation, population,
                               mu, expected_weight_diff, expected_loss,
                               user_dataset, nn_train_params,
                               check_equal_stats, make_algorithm_setup):

        fedprox, fedprox_params = make_algorithm_setup(mu)
        central_context = fedprox.get_next_central_contexts(
            model_setup.model, 0, fedprox_params, nn_train_params,
            NNEvalHyperParams(local_batch_size=8))[0][0]

        model_update, metrics = fedprox.simulate_one_user(
            model_setup.model, user_dataset, central_context)

        # pytype: disable=duplicate-keyword-argument,wrong-arg-count
        if central_context.population == Population.TRAIN:
            # Should be training.
            assert isinstance(model_update, MappedVectorStatistics)
            assert len(model_update) == 1
            assert model_update.weight == 1
            if central_context.do_evaluation:
                # Evaluation metrics from both before and after training.
                assert get_overall_value(
                    metrics[TrainMetricName('loss',
                                            central_context.population,
                                            after_training=False)]) == 4
                assert get_overall_value(metrics[TrainMetricName(
                    'loss', central_context.population,
                    after_training=True)]) == expected_loss

            check_equal_stats(
                model_update.apply_elementwise(
                    model_setup.variable_to_numpy_fn),
                {model_setup.variable_names[0]: expected_weight_diff})
        else:
            # No model update for val iteration.
            assert model_update is None
            if central_context.do_evaluation:
                assert get_overall_value(
                    metrics[TrainMetricName('loss',
                                            central_context.population,
                                            after_training=False)]) == 4

        # pytype: enable=duplicate-keyword-argument,wrong-arg-count

        if not central_context.do_evaluation:
            assert len(metrics) == 0


@pytest.fixture
def testcase_skip_iteration():

    adaptive_mu = AdaptMuOnMetricCallback(metric_name='meatballs',
                                          adapt_frequency=2,
                                          initial_value=1.0)

    return {
        'adaptive_mu': adaptive_mu,
        'metric_values': [0.5],
        'expected_mus': [1.0]
    }


@pytest.fixture
def testcase_decrease_increase_flat():
    adaptive_mu = AdaptMuOnMetricCallback(metric_name='meatballs',
                                          adapt_frequency=1,
                                          initial_value=1.0)
    return {
        'adaptive_mu':
        adaptive_mu,
        'metric_values':
        [5, 5, 5, 4, 3, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 3, 4, 5],
        'expected_mus': [
            0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0,
            0.0, 0.1, 0.2, 0.3
        ]
    }


@pytest.fixture
def testcase_decrease_increase_small_step():
    adaptive_mu = AdaptMuOnMetricCallback(metric_name='meatballs',
                                          adapt_frequency=1,
                                          initial_value=1.0,
                                          step_size=0.01)
    return {
        'adaptive_mu':
        adaptive_mu,
        'metric_values':
        [5, 5, 5, 4, 3, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9, 1.8, 1.7, 3, 4, 5],
        'expected_mus': [
            0.99, 1.0, 1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92,
            0.91, 0.90, 0.89, 0.90, 0.91, 0.92
        ]
    }


@pytest.fixture
def testcase_decrease_mu_after_consecutive_improvements():
    adaptive_mu = AdaptMuOnMetricCallback(
        metric_name='meatballs',
        adapt_frequency=1,
        initial_value=1.0,
        decrease_mu_after_consecutive_improvements=3)
    return {
        'adaptive_mu': adaptive_mu,
        'metric_values': [9, 8, 7, 6, 5, 6, 5, 4, 3],
        'expected_mus': [1.0, 1.0, 0.9, 0.8, 0.7, 0.8, 0.8, 0.8, 0.7]
    }


@pytest.mark.parametrize('model_setup', model_setup_marks)
class TestAdaFedProx:

    @pytest.mark.parametrize('setup', [
        lf('testcase_skip_iteration'),
        lf('testcase_decrease_increase_flat'),
        lf('testcase_decrease_increase_small_step'),
        lf('testcase_decrease_mu_after_consecutive_improvements'),
    ])
    def test_process_aggregated_statistics(self, setup, model_setup):
        for metric_value, mu in zip(setup['metric_values'],
                                    setup['expected_mus']):

            _, metrics_after_adapt = setup[
                'adaptive_mu'].after_central_iteration(Metrics([
                    ('meatballs', metric_value)
                ]),
                                                       model_setup.model,
                                                       central_iteration=1)
            assert mu == pytest.approx(metrics_after_adapt['mu'])
