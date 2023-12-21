# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import math
import pathlib
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.algorithm import FederatedAveraging, NNAlgorithmParams
from pfl.callback import ModelCheckpointingCallback
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.ops.common_ops import get_pytorch_major_version, get_tf_major_version
from pfl.metrics import Metrics, StringMetricName, TrainMetricName, get_overall_value
from pfl.stats import MappedVectorStatistics


@pytest.fixture
def nn_algorithm_params_dict(request):
    kwargs = request.param if hasattr(request, 'param') else {}
    return kwargs


@pytest.fixture
def nn_eval_params():
    return NNEvalHyperParams(local_batch_size=8)


@pytest.fixture(scope='function')
def fedavg_setup(new_event_loop, nn_algorithm_params_dict,
                 make_hyperparameter):
    algo = FederatedAveraging()
    algo_params = NNAlgorithmParams(
        **{
            'central_num_iterations': 3,
            'evaluation_frequency': 2,
            'train_cohort_size': make_hyperparameter(4),
            'val_cohort_size': 5,
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


class TestFederatedAveraging:

    @pytest.mark.parametrize('total_weight,num_devices,should_error', [
        (2, 2, False),
        (4, 2, False),
        (4, 2, True),
        (2, 2, True),
    ])
    def test_process_aggregated_statistics(self, total_weight, num_devices,
                                           should_error, mock_model,
                                           fedavg_setup):
        raw_model_updates = {
            'w1': np.ones((2, 3)),
            'var2': np.ones((3, 1)) * 2
        }
        model_statistics = MappedVectorStatistics(
            {
                k: v.copy()
                for k, v in raw_model_updates.items()
            },
            weight=total_weight)
        metrics = Metrics()
        mock_central_context = MagicMock(spec=CentralContext)

        try:
            (_, processed_metrics
             ) = fedavg_setup['algorithm'].process_aggregated_statistics(
                 mock_central_context, metrics, mock_model, model_statistics)
            assert processed_metrics[StringMetricName('applied_stats')] == 1

            applied_model_update = mock_model.apply_model_update.mock_calls[
                0].args[0]
            assert len(applied_model_update) == 2
            for name in applied_model_update:
                np.testing.assert_array_equal(
                    applied_model_update[name],
                    raw_model_updates[name] / total_weight)
        except NotImplementedError:
            assert should_error

    def test_get_next_central_contexts(self, fedavg_setup, nn_train_params,
                                       nn_eval_params):
        algo = fedavg_setup['algorithm']
        expected_model_train_params = NNTrainHyperParams(
            local_num_epochs=6,
            local_learning_rate=0.1,
            local_batch_size=7,
            local_max_grad_norm=0.5)
        expected_model_eval_params = NNEvalHyperParams(local_batch_size=8)
        (train_iteration0,
         val_iteration0), _, _ = algo.get_next_central_contexts(
             MagicMock(),
             iteration=0,
             algorithm_params=fedavg_setup['algorithm_params'],
             model_train_params=nn_train_params,
             model_eval_params=nn_eval_params)
        assert train_iteration0 == CentralContext(
            current_central_iteration=0,
            do_evaluation=True,
            population=Population.TRAIN,
            cohort_size=4,
            algorithm_params=fedavg_setup['algorithm_params'].static_clone(),
            model_train_params=expected_model_train_params,
            model_eval_params=expected_model_eval_params,
            seed=1337)
        assert val_iteration0 == CentralContext(
            current_central_iteration=0,
            do_evaluation=True,
            population=Population.VAL,
            cohort_size=5,
            algorithm_params=fedavg_setup['algorithm_params'].static_clone(),
            model_train_params=expected_model_train_params,
            model_eval_params=expected_model_eval_params,
            seed=1338)
        (train_iteration1, ), _, _ = algo.get_next_central_contexts(
            MagicMock(),
            iteration=1,
            algorithm_params=fedavg_setup['algorithm_params'].static_clone(),
            model_train_params=nn_train_params,
            model_eval_params=nn_eval_params)
        assert train_iteration1 == CentralContext(
            current_central_iteration=1,
            do_evaluation=False,
            population=Population.TRAIN,
            cohort_size=4,
            algorithm_params=fedavg_setup['algorithm_params'].static_clone(),
            model_train_params=NNTrainHyperParams(local_num_epochs=6,
                                                  local_learning_rate=0.1,
                                                  local_batch_size=7,
                                                  local_max_grad_norm=0.5),
            model_eval_params=NNEvalHyperParams(local_batch_size=8),
            seed=1339)

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
    @pytest.mark.parametrize('do_evaluation,population',
                             [(True, Population.VAL), (True, Population.TRAIN),
                              (True, Population.VAL)])
    def test_simulate_one_user(self, do_evaluation, population, user_dataset,
                               fedavg_setup, model_setup):

        nn_train_params = NNTrainHyperParams(local_num_epochs=10,
                                             local_learning_rate=0.25,
                                             local_batch_size=None,
                                             local_max_grad_norm=None)

        federated_averaging = fedavg_setup['algorithm']
        central_context = CentralContext(
            current_central_iteration=0,
            do_evaluation=do_evaluation,
            cohort_size=5,
            population=population,
            model_train_params=nn_train_params,
            model_eval_params=NNEvalHyperParams(local_batch_size=8),
            algorithm_params=fedavg_setup['algorithm_params'].static_clone(),
            seed=1338)

        model_update, metrics = federated_averaging.simulate_one_user(
            model_setup.model, user_dataset, central_context)

        # pytype: disable=duplicate-keyword-argument,wrong-arg-count
        if central_context.population == Population.TRAIN:
            # Should be training.
            assert len(model_update) == 1
            assert model_update.weight == 1
            # Same calculation as in
            # test_model::TestModel::test_train_and_difference
            np.testing.assert_array_equal(
                np.array([[1.25, 1.25], [1., 1.]]),
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
                                            after_training=True)]) == 0.75
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

    @patch('pfl.algorithm.base.get_platform')
    @patch('pfl.internal.platform.selector.get_platform')
    @pytest.mark.parametrize('nn_algorithm_params_dict',
                             ({
                                 'central_num_iterations': 1,
                                 'evaluation_frequency': 1,
                             }, {
                                 'central_num_iterations': 10,
                                 'evaluation_frequency': 2
                             }),
                             indirect=True)
    @pytest.mark.parametrize('send_metrics_to_platform', (True, False))
    def test_run(self, mock_get_platform_algo, mock_get_platform_cb,
                 send_metrics_to_platform, fedavg_setup, mock_backend,
                 mock_callback, mock_model, nn_algorithm_params_dict,
                 nn_eval_params, nn_train_params, tmp_path):
        mock_platform = MagicMock()
        mock_get_platform_algo.return_value = mock_platform
        mock_get_platform_cb.return_value = mock_platform
        mock_platform.create_checkpoint_directories.return_value = [tmp_path]

        central_num_iterations = nn_algorithm_params_dict[
            'central_num_iterations']
        evaluation_frequency = nn_algorithm_params_dict['evaluation_frequency']
        fedavg_setup['algorithm'].run(
            algorithm_params=fedavg_setup['algorithm_params'],
            backend=mock_backend,
            model=mock_model,
            model_train_params=nn_train_params,
            model_eval_params=nn_eval_params,
            callbacks=[
                mock_callback,
                ModelCheckpointingCallback(str(tmp_path)),
            ],
            send_metrics_to_platform=send_metrics_to_platform)

        assert (mock_backend.async_gather_results.call_count ==
                central_num_iterations +
                int(math.ceil(central_num_iterations / evaluation_frequency)))
        assert (
            mock_model.apply_model_update.call_count == central_num_iterations)
        if send_metrics_to_platform:
            assert (mock_platform.consume_metrics.call_count ==
                    central_num_iterations)
        else:
            assert (mock_platform.consume_metrics.call_count == 0)

        expected_model_train_params = NNTrainHyperParams(
            local_num_epochs=6,
            local_learning_rate=0.1,
            local_batch_size=7,
            local_max_grad_norm=0.5)
        mock_backend.async_gather_results.assert_has_calls([
            call(model=mock_model,
                 training_algorithm=fedavg_setup['algorithm'],
                 central_context=CentralContext(
                     current_central_iteration=0,
                     do_evaluation=True,
                     population=Population.VAL,
                     cohort_size=5,
                     algorithm_params=fedavg_setup['algorithm_params'].
                     static_clone(),
                     model_train_params=expected_model_train_params,
                     model_eval_params=nn_eval_params,
                     seed=1338)),
            call(model=mock_model,
                 training_algorithm=fedavg_setup['algorithm'],
                 central_context=CentralContext(
                     current_central_iteration=0,
                     do_evaluation=True,
                     population=Population.TRAIN,
                     cohort_size=4,
                     algorithm_params=fedavg_setup['algorithm_params'].
                     static_clone(),
                     model_train_params=expected_model_train_params,
                     model_eval_params=nn_eval_params,
                     seed=1337)),
        ],
                                                           any_order=True)

        # Assert the model update called with.
        applied_model_update = mock_model.apply_model_update.mock_calls[
            0].args[-1]
        np.testing.assert_array_equal(applied_model_update['var1'],
                                      np.ones((2, 3)) / 4)

        if send_metrics_to_platform:
            reported_metrics = (mock_platform.consume_metrics.mock_calls[0].
                                args[0].to_simple_dict())
            # Metrics reported should contain:
            # * train loss from calling gather_results on train population
            # * train loss after from calling gather_results on train population
            # * val loss from calling gather_results on val population
            # * num parameters from stat always added
            # * Should have added metrics from callback
            assert len(reported_metrics) == 8
            assert reported_metrics[
                'Train population | loss before local training'] == 1337
            assert reported_metrics[
                'Train population | loss after local training'] == 0.1
            assert reported_metrics[
                'Val population | loss before local training'] == 1337
            assert reported_metrics['Train population | total weight'] == 4
            assert reported_metrics['Val population | total weight'] == 5
            assert reported_metrics['Called_callback'] == 1
            assert reported_metrics['Called_begin'] == 1

        assert (mock_platform.create_checkpoint_directories.call_count == 1)
        mock_model.save.assert_called_with(pathlib.Path(tmp_path))

        # Callback should have been called once on all its hooks
        mock_callback.on_train_begin.assert_called_once_with(model=mock_model)
        assert (mock_callback.after_central_iteration.call_count ==
                central_num_iterations)
        for i, callback_call in enumerate(
                mock_callback.after_central_iteration.call_args_list):
            assert callback_call.args[0][StringMetricName(
                'applied_stats')] == 1
            assert callback_call.args[1] is mock_model
            assert callback_call.kwargs['central_iteration'] == i

        mock_callback.on_train_end.assert_called_once_with(model=mock_model)

    @patch('pfl.internal.platform.selector.get_platform')
    def test_run_stop_signal(self, mock_get_platform, fedavg_setup,
                             mock_backend, mock_callback, mock_model,
                             nn_eval_params, nn_train_params, tmp_path):
        mock_platform = MagicMock()
        mock_get_platform.return_value = mock_platform

        def should_stop_training(metrics, model, central_iteration):
            should_stop = central_iteration == 2
            return should_stop, Metrics()

        mock_callback.after_central_iteration.side_effect = should_stop_training
        fedavg_setup['algorithm'].run(
            algorithm_params=fedavg_setup['algorithm_params'],
            backend=mock_backend,
            model=mock_model,
            model_train_params=nn_train_params,
            model_eval_params=nn_eval_params,
            callbacks=[mock_callback])

        mock_callback.on_train_end.assert_called_once_with(model=mock_model)
