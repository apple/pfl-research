# Copyright © 2023-2024 Apple Inc.
import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.algorithm.scaffold import SCAFFOLD, SCAFFOLDParams
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.user_state import InMemoryUserStateStorage
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.ops.common_ops import get_pytorch_major_version
from pfl.metrics import Metrics, StringMetricName, TrainMetricName, get_overall_value
from pfl.stats import MappedVectorStatistics


@pytest.fixture
def nn_eval_params():
    return NNEvalHyperParams(local_batch_size=8)


@pytest.fixture
def nn_train_params():
    return NNTrainHyperParams(local_num_epochs=1,
                              local_learning_rate=0.1,
                              local_batch_size=None,
                              local_max_grad_norm=None)


# pylint: disable=protected-access
class TestSCAFFOLD:
    """
    The model update test cases were generated manually using the code:
    (assuming full-batch GD)
    ```
w = np.array([[2.,4.], [3.,5.]])
w_orig = w.copy()
x = np.array([[1.,0.], [0.,2.]])
t = np.array([[4.,6.], [8.,12.]])
lr = 0.1
epochs = 1
for _ in range(epochs):
    y = np.dot(x,w)
    loss_before = np.sum(np.abs(np.dot(x,w)-t))/len(x)
    g_loss = np.ones_like(y)
    g_loss[y < t] = -1 # Gradient of MAE
    w = w - lr * 1/2*np.dot(x.T,g_loss)
    loss_after = np.sum(np.abs(np.dot(x,w)-t))/len(x)
    print(f'delta={w - w_orig} final_weight={w}'
    print(f'losses=[{loss_before},{loss_after}]')
    ```
    """

    @pytest.mark.parametrize(
        'model_setup',
        [
            pytest.param(lazy_fixture('pytorch_model_setup'),
                         marks=[
                             pytest.mark.skipif(
                                 not get_pytorch_major_version(),
                                 reason='PyTorch not installed')
                         ],
                         id='pytorch'),
            # not yet implemented for tf.
            #pytest.param(lazy_fixture('tensorflow_model_setup'),
            #             marks=[
            #                 pytest.mark.skipif(get_tf_major_version() < 2,
            #                                    reason='not tf>=2')
            #             ],
            #             id='keras_v2')
        ])
    @pytest.mark.parametrize(
        'initial_c,expected_weight_diff,expected_local_c, '
        'expected_loss, use_gradient_as_control_variate', [
            (0.0, np.array([[0.05, 0.05], [0.1, 0.1]]),
             np.array([[0.5, 0.5], [1.0, 1.0]]), 3.75, False),
            (0.1, np.array([[0.04, 0.04], [0.09, 0.09]]),
             np.array([[0.3, 0.3], [0.8, 0.8]]), 3.78, False),
            (0.0, np.array([[0.05, 0.05], [0.1, 0.1]]),
             np.array([[0.5, 0.5], [1.0, 1.0]]), 3.75, True),
            (0.1, np.array([[0.04, 0.04], [0.09, 0.09]]),
             np.array([[0.5, 0.5], [1.0, 1.0]]), 3.78, True),
        ])
    @pytest.mark.parametrize('do_evaluation,population',
                             [(True, Population.VAL), (True, Population.TRAIN),
                              (True, Population.VAL)])
    def test_simulate_one_user(self, model_setup, do_evaluation, population,
                               initial_c, expected_weight_diff,
                               expected_local_c, expected_loss,
                               use_gradient_as_control_variate, user_dataset,
                               nn_train_params, check_equal_stats):
        scaffold = SCAFFOLD()

        scaffold_params = SCAFFOLDParams(
            central_num_iterations=2,
            evaluation_frequency=2,
            train_cohort_size=2,
            val_cohort_size=None,
            population=100,
            use_gradient_as_control_variate=use_gradient_as_control_variate,
            user_state_storage=InMemoryUserStateStorage())

        # This needs to be called first to init.
        ((context, ), _,
         metrics) = scaffold.get_next_central_contexts(model_setup.model, 0,
                                                       scaffold_params,
                                                       nn_train_params, None)
        assert len(metrics) == 0
        assert context.current_central_iteration == 0
        assert context.do_evaluation is True
        assert context.cohort_size == 2
        assert context.population == Population.TRAIN

        central_context = CentralContext(
            current_central_iteration=0,
            do_evaluation=do_evaluation,
            cohort_size=5,
            population=population,
            model_train_params=nn_train_params,
            model_eval_params=NNEvalHyperParams(local_batch_size=8),
            algorithm_params=scaffold_params,
            seed=1338)

        scaffold._server_c = scaffold._server_c.apply_elementwise(
            lambda w: w + initial_c)
        model_update, metrics = scaffold.simulate_one_user(
            model_setup.model, user_dataset, central_context)

        if central_context.population == Population.TRAIN:
            # Should be training.
            assert isinstance(model_update, MappedVectorStatistics)
            assert len(model_update) == 2
            assert model_update.weight == 1
            if central_context.do_evaluation:
                # Evaluation metrics from both before and after training.
                assert get_overall_value(
                    metrics[TrainMetricName('loss',
                                            central_context.population,
                                            after_training=False)]) == 4
                assert get_overall_value(metrics[TrainMetricName(
                    'loss', central_context.population,
                    after_training=True)]) == pytest.approx(expected_loss)

            check_equal_stats(
                model_update.apply_elementwise(
                    model_setup.variable_to_numpy_fn), {
                        f'model_update/{model_setup.variable_names[0]}':
                        expected_weight_diff,
                        f'c/{model_setup.variable_names[0]}': expected_local_c
                    })
        else:
            # No model update for val iteration.
            assert model_update is None
            if central_context.do_evaluation:
                assert get_overall_value(
                    metrics[TrainMetricName('loss',
                                            central_context.population,
                                            after_training=False)]) == 4

        if not central_context.do_evaluation:
            assert len(metrics) == 0

    def test_process_aggregated_statistics(self, mock_model, user_dataset,
                                           nn_train_params, check_equal_stats,
                                           check_equal_metrics):
        scaffold = SCAFFOLD()

        scaffold_params = SCAFFOLDParams(
            central_num_iterations=2,
            evaluation_frequency=2,
            train_cohort_size=2,
            val_cohort_size=None,
            population=100,
            use_gradient_as_control_variate=False,
            user_state_storage=InMemoryUserStateStorage())

        scaffold._server_c = MappedVectorStatistics(
            {'weight': np.ones((2, 2)) * 2})

        central_context = CentralContext(
            current_central_iteration=0,
            do_evaluation=True,
            cohort_size=5,
            population=Population.TRAIN,
            model_train_params=nn_train_params,
            model_eval_params=NNEvalHyperParams(local_batch_size=8),
            algorithm_params=scaffold_params,
            seed=1338)

        model_update = MappedVectorStatistics({
            'model_update/weight':
            np.ones((2, 2)),
            'c/weight':
            np.ones((2, 2)) * 0.5
        })
        _, model_update_metrics = scaffold.process_aggregated_statistics(
            central_context, Metrics(), mock_model, model_update)

        check_equal_stats(
            mock_model.apply_model_update.call_args_list[0][0][0],
            MappedVectorStatistics({'weight': np.array([[1., 1.], [1., 1.]])}))
        check_equal_metrics(
            model_update_metrics,
            Metrics([(StringMetricName('applied_stats'), 1),
                     (StringMetricName('server_c norm'), 1)]))
        check_equal_stats(
            scaffold._server_c,
            MappedVectorStatistics({'weight': np.ones((2, 2)) * 2.025}))
