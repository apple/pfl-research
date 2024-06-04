import argparse
import os

import joblib
import numpy as np
import torch

from pfl.internal.ops import pytorch_ops
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.internal.ops.selector import set_framework_module
from pfl.internal.platform.selector import get_platform
from publications.mdm.mdm_paper.training.mle import solve_polya_mixture_mle
from publications.mdm.mdm_utils.datasets import make_femnist_datasets
from publications.mdm.mdm_utils.utils import (
    add_algorithm_args,
    add_dataset_preprocessing_args,
    add_experiment_args,
    add_histogram_algorithm_args,
    add_init_algorithm_args,
    add_mle_args,
    add_user_visualisation_args,
)


def get_arguments():
    parser = argparse.ArgumentParser()
    add_experiment_args(parser)
    add_mle_args(parser)
    add_init_algorithm_args(parser)
    add_algorithm_args(parser)
    add_histogram_algorithm_args(parser)
    add_user_visualisation_args(parser)
    add_dataset_preprocessing_args(parser)
    return parser.parse_args()


set_framework_module(pytorch_ops)
arguments = get_arguments()
np.random.seed(arguments.seed)
torch.random.manual_seed(arguments.seed)

# Solve MLE using only CPU
os.environ['RAMSAY_PYTORCH_DEVICE'] = 'cpu'

# Create data of live users
num_classes = 62
input_shape = (28, 28, 1)

live_training_data, live_val_data, central_val_data = make_femnist_datasets(
    arguments.data_dir,
    digits_only=False,
    numpy_to_tensor=get_ops().to_tensor,
    dataset_type=arguments.dataset_type,
    filter_method=arguments.filter_method,
    sample_fraction=arguments.sample_fraction,
    start_idx=arguments.start_idx,
    end_idx=arguments.end_idx,
    include_sampled=arguments.include_sampled)

add_DP = False

# If running simulations then compute phi, alphas and num_samples_histos
# either by solving the polya MLE or just computing the histogram for
# uniform simulations
print('simulated_dirichlet_mixture experiment')
if arguments.precomputed_parameter_filepath is None:
    print('learn simulated_dirichlet_mixture parameters')
    dir_path = get_platform().create_checkpoint_directories(
        [arguments.mle_param_dirname])[0]
    save_dir = (
        f'femnist_{arguments.dataset_type}_{arguments.num_mixture_components}_mixture_{arguments.filter_method}_filter_method'
    )
    if arguments.filter_method is not None:
        if arguments.filter_method == 'index':
            save_dir += f'_{arguments.start_idx}_start_idx_{arguments.end_idx}_end_idx'
        elif arguments.filter_method == 'sample':
            save_dir += f'_{arguments.sample_fraction}_sample_fraction_{arguments.include_sampled}_include_sampled'
        else:
            raise ValueError(
                f'Invalid value for filter_method: {arguments.filter_method}')

    if add_DP:
        save_dir += '_DP'
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
        save_path_histogram=save_path_histogram,
        add_DP=add_DP)

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
