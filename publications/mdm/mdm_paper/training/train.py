import os
import argparse
import datetime

import numpy as np
import torch
import joblib

from pfl.internal.ops import pytorch_ops
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.internal.ops.selector import set_framework_module
from pfl.internal.platform.selector import get_platform

from publications.mdm.mdm_utils.datasets import make_cifar10_datasets
from publications.mdm.mdm_utils.utils import (
    add_dataset_args, add_experiment_args, add_mle_args,
    add_init_algorithm_args, add_algorithm_args, add_histogram_algorithm_args,
    add_user_visualisation_args)

from publications.mdm.mdm_paper.training.mle import solve_polya_mixture_mle


def get_arguments():
    parser = argparse.ArgumentParser()
    add_experiment_args(parser)
    add_dataset_args(parser)
    add_mle_args(parser)
    add_init_algorithm_args(parser)
    add_algorithm_args(parser)
    add_histogram_algorithm_args(parser)
    add_user_visualisation_args(parser)
    return parser.parse_args()


set_framework_module(pytorch_ops)
arguments = get_arguments()
np.random.seed(arguments.seed)
torch.random.manual_seed(arguments.seed)

# Solve MLE using only CPU
os.environ['RAMSAY_PYTORCH_DEVICE'] = 'cpu'

# Create data of live users
num_classes = 10
input_shape = (32, 32, 3)

# check arguments for mixture components
print('arguments.num_mixture_components', arguments.num_mixture_components,
      type(arguments.num_mixture_components))
print('arguments.component_mean_user_dataset_length',
      arguments.component_mean_user_dataset_length)
print('arguments.component_phi', arguments.component_phi,
      type(arguments.component_phi))

assert arguments.num_mixture_components == len(
    arguments.component_mean_user_dataset_length) == len(
        arguments.component_phi)
if len(arguments.component_alphas) == arguments.num_mixture_components:
    # one alpha for all classes for each mixture component
    alphas = np.array(arguments.component_alphas).reshape(
        -1, 1) * np.ones(num_classes)
    print('alphas', alphas)
else:
    # must have length num mixture components * num_classes
    print('len(arguments.component_alphas)', len(arguments.component_alphas))
    print('arguments.num_mixture_components * num_classes',
          arguments.num_mixture_components * num_classes)
    assert len(arguments.component_alphas
               ) == arguments.num_mixture_components * num_classes
    # assumes alphas are ordered each class, each mixture component
    print('arguments.component_alphas', arguments.component_alphas)
    alphas = np.array(arguments.component_alphas).reshape(
        arguments.num_mixture_components, num_classes)
    print('alphas', alphas.shape, alphas)

print('arguments.component_phi', arguments.component_phi)
phi = np.array(arguments.component_phi)
# default values for samplers are tuple
# arguments.component_mean_user_dataset_length (50, 30)
# but these default values can be overwritten at run time of lambda fn.
samplers = [
    lambda x=x: x for x in arguments.component_mean_user_dataset_length
]
print('samplers', [s() for s in samplers])
print('true phi', phi)
print('true alphas', alphas)

# option to create artificial_federated_dataset or federated_dataset
live_training_data, live_val_data, central_val_data = make_cifar10_datasets(
    dataset_type='artificial_federated_dataset',
    data_dir=arguments.data_dir,
    user_dataset_len_samplers=samplers,
    numpy_to_tensor=get_ops().to_tensor,
    phi=phi,
    alphas=alphas)

# TODO support modelling federated dataset using mixture-polya
# or uniform distribution

# If running simulations then compute phi, alphas and num_samples_histos
# either by solving the polya MLE or just computing the histogram for
# uniform simulations
print('simulated_dirichlet_mixture experiment')
if arguments.precomputed_parameter_filepath is None:
    print('learn simulated_dirichlet_mixture parameters')
    dir_path = get_platform().create_checkpoint_directories(
        [arguments.mle_param_dirname])[0]
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M")
    save_dir = (
        #f'cifar10_{arguments.num_mixture_components}_mixture_{timestamp}')
        f'cifar10_{arguments.num_mixture_components}_mixture_{arguments.dirname}'
    )
    save_path = os.path.join(dir_path, save_dir)

    dir_path_histogram = get_platform().create_checkpoint_directories(
        ['num_samples_distribution'])[0]
    save_path_histogram = os.path.join(dir_path_histogram,
                                       save_dir + '.joblib')

    # Solve polya-mixture MLE
    # TODO use mle arguments for cohort size, num iterations, etc.
    # TODO what to do about num_samples_histos?
    phi, alphas, num_samples_distributions = solve_polya_mixture_mle(
        arguments=arguments,
        training_federated_dataset=live_training_data,
        val_federated_dataset=None,
        num_components=arguments.num_mixture_components,
        num_categories=num_classes,
        save_path=save_path,
        save_path_histogram=save_path_histogram)

    phi = phi.numpy() if isinstance(phi, torch.Tensor) else phi
    alphas = alphas.numpy() if isinstance(alphas, torch.Tensor) else alphas
    num_samples_distributions = num_samples_distributions.numpy()
    print('phi', phi)
    print('alphas', alphas)
    print('num_samples_distributions', num_samples_distributions)

else:
    params = joblib.load(arguments.precomputed_parameter_filepath)

    phi = np.array(params['phi'])
    phi /= phi.sum()
    alphas = np.array(params['alphas'])
    num_samples_distributions = np.array(params['num_samples_distributions'])
    num_samples_distributions /= np.sum(num_samples_distributions,
                                        axis=1,
                                        keepdims=True)
