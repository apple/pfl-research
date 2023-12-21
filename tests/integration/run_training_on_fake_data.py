# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import argparse
import atexit
import signal
import sys

import dill
import numpy as np

from pfl.aggregate.simulate import SimulatedBackend
from pfl.algorithm import NNAlgorithmParams, algorithm_utils
from pfl.algorithm.federated_averaging import FederatedAveraging
from pfl.callback import CentralEvaluationCallback
from pfl.data.dataset import Dataset
from pfl.data.federated_dataset import ArtificialFederatedDataset, FederatedDataset
from pfl.data.sampling import get_data_sampler, get_user_sampler
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.ops.common_ops import get_pytorch_major_version
from pfl.internal.ops.selector import get_framework_module as get_ops
from pfl.metrics import Metrics, Weighted, get_overall_value
from pfl.privacy import (
    CentrallyApplicablePrivacyMechanism,
    CentrallyAppliedPrivacyMechanism,
    GaussianMechanism,
    LaplaceMechanism,
    NoPrivacy,
    NormClippingOnly,
    PrivUnitMechanism,
    PLDPrivacyAccountant,
)


class store_bool(argparse.Action):

    def __init__(self, option_strings, dest, **kwargs):
        argparse.Action.__init__(self, option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        false_values = {'false', 'no'}
        true_values = {'true', 'yes'}

        values = values.lower()

        if values not in (false_values | true_values):
            raise argparse.ArgumentError(
                self, 'Value must be either "true" or "false"')
        value = (values in true_values)

        setattr(namespace, self.dest, value)


def add_live_train_privacy_arguments(argument_parser):

    argument_parser.add_argument(
        '--local_privacy_clipping_bound',
        type=float,
        default=1.0,
        help=
        'Bound for norm clipping so that the local privacy mechanism can be '
        'applied.')

    argument_parser.add_argument(
        '--central_epsilon',
        type=float,
        default=1.,
        help=
        'Bound on the privacy loss (often called ε) for the central privacy '
        'mechanism.')

    argument_parser.add_argument(
        '--central_delta',
        type=float,
        default=1e-05,
        help='Bound on the probability that the privacy loss '
        'is greater than ε (often called δ) for the central privacy mechanism.'
    )

    argument_parser.add_argument(
        '--central_privacy_clipping_bound',
        type=float,
        default=1.0,
        help='Bound for norm clipping for each user, so '
        'that the central privacy mechanism can be applied.')

    return argument_parser


def add_mechanism_arguments(argument_parser):

    # Arguments that govern the local privacy mechanism.
    argument_parser.add_argument(
        '--local_privacy_mechanism',
        choices=[
            'none', 'gaussian', 'privunit', 'laplace', 'norm_clipping_only',
            'separated', 'local_dp_separated'
        ],
        default='none',
        help='The type of privacy mechanism to apply for each user.')

    argument_parser.add_argument(
        '--local_epsilon',
        type=float,
        default=1.,
        help='Bound on the privacy loss (often called ε) for the local privacy '
        'mechanism.')

    argument_parser.add_argument(
        '--local_delta',
        type=float,
        default=1e-05,
        help='Bound on the probability that the privacy loss '
        'is greater than ε (often called δ) for the local privacy mechanism.')

    argument_parser.add_argument(
        '--local_order',
        type=float,
        help='Order of the lp-norm for local norm clipping only.')

    argument_parser.add_argument(
        '--local_reconstruction_probability',
        type=float,
        default=1e-9,
        help='The Separated Differential Privacy parameter. '
        'Probability of reconstructing the data within a Euclidean distance of '
        '1.2.')

    argument_parser.add_argument(
        '--local_rho',
        type=float,
        default=10.0,
        help='The SDP parameter for the magnitude of a weight vector.')

    argument_parser.add_argument(
        '--local_precision',
        type=float,
        default=1e-14,
        help='Privacy parameter for Separated DP. Precision for drawing from a '
        'Conditional Beta Distribution')

    argument_parser.add_argument(
        '--local_min_cd_product',
        type=float,
        default=1e-14,
        help='Privacy parameter for Separated DP. '
        'Change in Continued fraction estimate of CDF of a '
        'Conditional Beta Distribution.')

    argument_parser.add_argument(
        '--local_max_rejection',
        type=int,
        default=200,
        help='Privacy parameter for Separated DP. '
        'Refers to tries for recurrence relation for continued fraction '
        'estimate. '
        'We stop if there have been more than local_max_rejection tries and '
        'value is greater than minDCProduct.')

    # Arguments that govern the central privacy mechanism.
    argument_parser = add_live_train_privacy_arguments(argument_parser)

    argument_parser.add_argument(
        '--central_privacy_mechanism',
        choices=[
            'none', 'gaussian', 'gaussian_privacy_accountant', 'laplace',
            'norm_clipping_only'
        ],
        default='none',
        help='The type of privacy mechanism to apply for each model update.')

    argument_parser.add_argument(
        '--central_order',
        type=float,
        help='Order of the lp-norm for central norm clipping only.')

    return argument_parser


def parse_mechanism(mechanism_name,
                    clipping_bound=None,
                    epsilon=None,
                    delta=None,
                    order=None,
                    cohort_size=None,
                    num_epochs=None,
                    reconstruction_probability=None,
                    population=None,
                    rho=None,
                    precision=None,
                    min_cd_product=None,
                    max_rejection=None,
                    is_central=False):
    if mechanism_name == 'none':
        mechanism = NoPrivacy()

    elif mechanism_name == 'gaussian':
        assert clipping_bound is not None
        assert epsilon is not None and delta is not None
        mechanism = GaussianMechanism.construct_single_iteration(
            clipping_bound, epsilon, delta)

    elif mechanism_name == 'privunit':
        assert clipping_bound is not None
        assert epsilon is not None
        mechanism = PrivUnitMechanism(clipping_bound, epsilon)

    elif mechanism_name == 'gaussian_privacy_accountant':
        assert clipping_bound is not None
        assert epsilon is not None
        assert delta is not None
        assert cohort_size is not None
        assert num_epochs is not None
        assert population is not None
        accountant = PLDPrivacyAccountant(num_compositions=num_epochs,
                                          sampling_probability=cohort_size /
                                          population,
                                          epsilon=epsilon,
                                          delta=delta)
        mechanism = GaussianMechanism.from_privacy_accountant(
            accountant=accountant, clipping_bound=clipping_bound)

    elif mechanism_name == 'laplace':
        assert clipping_bound is not None
        assert epsilon is not None
        mechanism = LaplaceMechanism(clipping_bound, epsilon)

    elif mechanism_name == 'norm_clipping_only':
        assert clipping_bound is not None
        assert order is not None
        mechanism = NormClippingOnly(order, clipping_bound)

    else:
        raise AssertionError(
            "Please specify `mechanism_name`. If you don't want to         use any privacy, specify 'none'."
        )

    if is_central:
        assert isinstance(mechanism, CentrallyApplicablePrivacyMechanism), (
            '`is_central=True` will wrap the mechanism into a central '
            f'mechanism, but {mechanism} is not centrally applicable')
        mechanism = CentrallyAppliedPrivacyMechanism(mechanism)

    return mechanism


def dump_statistics_to_disk(backend, algorithm, args, model, central_data,
                            algorithm_params, model_train_params,
                            model_eval_params):
    """
    Given a backend that produces model updates, call to get the updates and
    dump them along with metrics to disk in a pickle.

    :param backend:
        The backend to get model updates from.
    :param algorithm:
        Algorithm to use for gathering results.
    :param args:
        An argparse namespace that contains settings for
        `backend.gather_results`.
        Also, args.output_path is the path to the pickle file to dump.
        If `None`, print out the statistics and metrics instead.
    :param model:
        The model to train.
    """

    central_contexts, _, _ = algorithm.get_next_central_contexts(
        model, 0, algorithm_params, model_train_params, model_eval_params)
    assert central_contexts is not None

    (statistics,
     train_metrics), (_, val_metrics) = algorithm_utils.run_train_eval(
         algorithm=algorithm,
         backend=backend,
         model=model,
         central_contexts=central_contexts)
    metrics = train_metrics | val_metrics
    statistics = statistics.apply(
        lambda tensors: [get_ops().to_numpy(t) for t in tensors])

    distribute_evaluation = (args.backend_framework == 'tf_model'
                             and not args.use_metric_spec)
    model_eval_params = central_contexts[1].model_eval_params
    cb = CentralEvaluationCallback(central_data,
                                   model_eval_params,
                                   distribute_evaluation=distribute_evaluation)
    metrics |= cb.after_central_iteration(Metrics(),
                                          model,
                                          central_iteration=0)[1]

    # Convert KerasMetricValue to its float value because they are not
    # picklable or dillable in TF.9.
    # Generalize this if we ever have more non-picklable metric values.
    metrics = Metrics([(k, get_overall_value(v)) for k, v in metrics])

    if args.output_path is not None:
        if get_ops().distributed.world_size > 1:
            with open(
                    f'{args.output_path}.{get_ops().distributed.global_rank}',
                    'wb') as f:
                dill.dump((statistics, metrics), f)
        else:
            with open(args.output_path, 'wb') as f:
                dill.dump((statistics, metrics), f)
    else:
        print(statistics)
        print(metrics)


def _make_keras_model_impl(pfl_model_cls, **kwargs):
    import tensorflow as tf  # type: ignore
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(tf.keras.optimizers.SGD(0.1),
                  loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC()])
    model.build(input_shape=(10, 2))

    model = pfl_model_cls(model=model,
                          central_optimizer=tf.keras.optimizers.SGD(1.0),
                          **kwargs)

    return model


def _make_tensorflow_model():
    import tensorflow as tf  # type: ignore
    try:
        # TF>2.10
        tf.keras.utils.set_random_seed(1)
    except AttributeError:
        tf.random.set_seed(1)
    from pfl.model.tensorflow import TFModel

    metrics = {'auc': tf.keras.metrics.AUC()}
    model = _make_keras_model_impl(TFModel, metrics=metrics)
    assert model.allows_distributed_evaluation is True
    return model


def _make_pytorch_model():
    import torch  # type: ignore
    torch.manual_seed(1)
    from pfl.model.pytorch import PyTorchModel

    class TestModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.w1 = torch.nn.parameter.Parameter(
                torch.tensor(np.random.normal(scale=0.1, size=(2, 10)),
                             dtype=torch.float32,
                             device='cpu'))
            self.b1 = torch.nn.parameter.Parameter(
                torch.tensor(np.random.normal(scale=0.1, size=(10, )),
                             dtype=torch.float32,
                             device='cpu'))
            self.w2 = torch.nn.parameter.Parameter(
                torch.tensor(np.random.normal(scale=0.1, size=(10, 1)),
                             dtype=torch.float32,
                             device='cpu'))
            self.b2 = torch.nn.parameter.Parameter(
                torch.tensor(np.random.normal(scale=0.1, size=(1, )),
                             dtype=torch.float32,
                             device='cpu'))

        def forward(self, x):  # pylint: disable=arguments-differ
            a1 = torch.nn.functional.sigmoid(
                torch.matmul(x, self.w1) + self.b1)
            a2 = torch.nn.functional.sigmoid(
                torch.matmul(a1, self.w2) + self.b2)
            return a2

        def loss(self, x, y, eval=False):
            if eval:
                self.eval()
            else:
                self.train()
            l1loss = torch.nn.BCELoss(reduction='sum')
            return l1loss(self(torch.FloatTensor(x)), torch.FloatTensor(y))

        def metrics(self, x, y):
            loss_value = self.loss(x, y, eval=True)
            num_samples = len(y)
            return {'loss': Weighted(loss_value, num_samples)}

    pytorch_model = TestModel()
    model = PyTorchModel(model=pytorch_model,
                         local_optimizer_create=torch.optim.SGD,
                         central_optimizer=torch.optim.SGD(
                             pytorch_model.parameters(), lr=1.0))
    return model


def _generate_feddata_generic(slices):
    sampler = get_user_sampler('random', list(slices.keys()))
    return FederatedDataset.from_slices(slices, sampler)


if get_pytorch_major_version():
    import torch

    # This class needs to be in global scope to allow
    # PyTorch's multiprocessing to pickle it.
    class MyData(torch.utils.data.Dataset):

        def __init__(self, slices):
            super().__init__()
            self._slices = slices

        def __getitem__(self, i):
            return self._slices[i]

        def __len__(self):
            return len(self._slices)


def _generate_feddata_pytorch(slices):
    from pfl.data.pytorch import PyTorchFederatedDataset
    dataset = MyData(slices)
    sampler = get_user_sampler('random', list(slices.keys()))
    return PyTorchFederatedDataset(dataset, sampler)


def _generate_feddata_tensorflow(slices):
    import tensorflow as tf

    from pfl.data.tensorflow import TFFederatedDataset

    def make_dataset_fn(user_index):
        return list(slices.values())[user_index]

    def make_tf_dataset_fn(data):
        return data.map(lambda i: tuple(
            tf.py_function(make_dataset_fn, [i], [tf.float32, tf.float32])))

    sampler = get_user_sampler('random', range(len(slices)))

    return TFFederatedDataset(make_tf_dataset_fn, sampler)


def _generate_federated_dataset(backend_framework, use_framework_dataset):
    # Generate 20 user datasets
    slices = {
        f'user_{i}': [
            np.random.normal(0, 1, size=(10, 2)).astype(np.float32),
            np.random.randint(0, 2, size=(10, 1)).astype(np.float32)
        ]
        for i in range(20)
    }
    if not use_framework_dataset:
        return _generate_feddata_generic(slices)
    elif backend_framework == 'pytorch':
        return _generate_feddata_pytorch(slices)
    elif backend_framework == 'tensorflow':
        return _generate_feddata_tensorflow(slices)
    else:
        raise AssertionError(
            f'{backend_framework} with data API not testable yet')


def _generate_simulated_federated_dataset(backend_framework,
                                          use_framework_dataset):
    assert not use_framework_dataset, 'Not testable yet'

    # Generate 20 user datasets
    dataset_len = 100
    slices = [
        np.random.normal(0, 1, size=(dataset_len, 2)),
        np.random.randint(0, 2, size=(dataset_len, 1)).astype(float)
    ]

    data_sampler = get_data_sampler('random', dataset_len)
    sample_dataset_len = lambda: 10
    return ArtificialFederatedDataset.from_slices(slices, data_sampler,
                                                  sample_dataset_len)


def _generate_central_dataset():
    dataset_len = 100
    return Dataset((np.random.normal(0, 1, size=(dataset_len, 2)),
                    np.random.randint(0, 2,
                                      size=(dataset_len, 1)).astype(float)))


if __name__ == '__main__':

    # Run atexit manually to kill any further subprocesses because it is
    # not run if this script itself is launched by a subprocess.
    # https://stackoverflow.com/a/34507557
    signal.signal(signal.SIGTERM, lambda _, __: atexit._run_exitfuncs())  # pylint: disable=protected-access

    np.random.seed(1)

    argument_parser = argparse.ArgumentParser(
        description='Run training on fake data with the specified components')

    argument_parser.add_argument('--output_path',
                                 default=None,
                                 help='Path to a pickle file to dump to disk')

    argument_parser.add_argument('--backend_framework',
                                 choices=['tensorflow', 'pytorch'],
                                 default='tensorflow',
                                 help='Which model should be used')

    argument_parser.add_argument(
        '--use_metric_spec',
        action=store_bool,
        default=False,
        help='Initialize model with metric_spec populated.')

    argument_parser.add_argument('--dataset_type',
                                 choices=['simulated-federated', 'federated'],
                                 default='federated',
                                 help='Which type of dataset to generate')

    argument_parser.add_argument(
        '--use_framework_dataset',
        action=store_bool,
        default=False,
        help=('Use dataset API of `backend_framework` to '
              'construct the federated dataset.'))

    argument_parser.add_argument(
        '--local_learning_rate',
        type=float,
        default=0.1,
        help='Learning rate for training on the client.')

    argument_parser.add_argument(
        '--local_num_epochs',
        type=int,
        default=5,
        help='Number of epochs for local training of one user.')

    argument_parser.add_argument(
        '--val_cohort_size',
        type=int,
        default=200,
        help='The target number of users for distributed evaluation.')

    argument_parser.add_argument(
        '--cohort_size',
        type=int,
        default=1000,
        help='The target number of users for one iteration '
        'of training.')

    argument_parser.add_argument(
        '--central_num_iterations',
        type=int,
        default=200,
        help='Number of iterations of global training, i.e. '
        'the number of minibatches.')

    argument_parser = add_mechanism_arguments(argument_parser)
    arguments = argument_parser.parse_args()

    local_privacy = parse_mechanism(
        mechanism_name=arguments.local_privacy_mechanism,
        clipping_bound=arguments.local_privacy_clipping_bound,
        epsilon=arguments.local_epsilon,
        delta=arguments.local_delta,
        order=arguments.local_order,
        reconstruction_probability=arguments.local_reconstruction_probability,
        rho=arguments.local_rho,
        precision=arguments.local_precision,
        min_cd_product=arguments.local_min_cd_product,
        max_rejection=arguments.local_max_rejection)
    central_privacy = parse_mechanism(
        mechanism_name=arguments.central_privacy_mechanism,
        clipping_bound=arguments.central_privacy_clipping_bound,
        epsilon=arguments.central_epsilon,
        delta=arguments.central_delta,
        order=arguments.central_order,
        cohort_size=arguments.cohort_size,
        num_epochs=arguments.central_num_iterations,
        population=int(1e7),
        is_central=True)
    assert isinstance(central_privacy, CentrallyAppliedPrivacyMechanism)

    # Too complicated to use the ModelSetup in test/nn/conftest.py
    # because of pickle. Define new models.
    if arguments.backend_framework == 'tensorflow':
        model = _make_tensorflow_model()
    else:
        assert arguments.backend_framework == 'pytorch'
        model = _make_pytorch_model()

    if arguments.dataset_type == 'simulated-federated':
        training_federated_dataset = _generate_simulated_federated_dataset(
            arguments.backend_framework, arguments.use_framework_dataset)
        val_federated_dataset = _generate_simulated_federated_dataset(
            arguments.backend_framework, arguments.use_framework_dataset)
    else:
        training_federated_dataset = _generate_federated_dataset(
            arguments.backend_framework, arguments.use_framework_dataset)
        val_federated_dataset = _generate_federated_dataset(
            arguments.backend_framework, arguments.use_framework_dataset)

    central_dataset = _generate_central_dataset()

    algorithm: FederatedAveraging = FederatedAveraging()
    algorithm_params = NNAlgorithmParams(
        central_num_iterations=arguments.central_num_iterations,
        evaluation_frequency=1,
        train_cohort_size=arguments.cohort_size,
        val_cohort_size=arguments.val_cohort_size)

    model_train_params = NNTrainHyperParams(
        local_num_epochs=arguments.local_num_epochs,
        local_learning_rate=arguments.local_learning_rate,
        local_batch_size=None)
    model_eval_params = NNEvalHyperParams(local_batch_size=None)

    backend = SimulatedBackend(training_data=training_federated_dataset,
                               val_data=val_federated_dataset,
                               postprocessors=[local_privacy, central_privacy])

    dump_statistics_to_disk(backend, algorithm, arguments, model,
                            central_dataset, algorithm_params,
                            model_train_params, model_eval_params)
    sys.exit(0)
