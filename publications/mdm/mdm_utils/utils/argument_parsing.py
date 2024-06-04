import argparse


class store_bool(argparse.Action):

    def __init__(self, option_strings, dest, **kwargs):
        argparse.Action.__init__(self, option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        false_values = set(['false', 'no'])
        true_values = set(['true', 'yes'])

        values = values.lower()

        if not values in (false_values | true_values):
            raise argparse.ArgumentError(
                self, 'Value must be either "true" or "false"')
        value = (values in true_values)

        setattr(namespace, self.dest, value)


def add_experiment_args(parser):
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dirname', type=str)
    parser.add_argument('--mle_param_dirname', type=str, default='publications/mdm/mle_params')
    parser.add_argument(
        '--precomputed_parameter_filepath',
        type=str,
        default=None,
        # default='saved_mle_params/femnist_2_mixture.pkl',
        help='If given then the inferred dirichlet mixture'
        'params will be loaded from here, do not specify file extension')
    parser.add_argument('--dataset_type',
                        type=str,
                        default='original',
                        choices=[
                            'original',
                            'original_labels_uniform_datapoints',
                            'polya_mixture_federated',
                            'polya_mixture_artificial_federated',
                            'uniform_federated',
                            'uniform_artificial_federated',
                        ])
    return parser


def add_dataset_preprocessing_args(parser):
    parser.add_argument('--filter_method',
                        type=str,
                        default=None,
                        choices=['index', 'sample'])
    parser.add_argument('--sample_fraction', type=float, default=1.0)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--include_sampled', action=store_bool, default=True)
    return parser


def float_list(arg):
    try:
        float_values = [float(val) for val in arg.split()]
        return float_values
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid float values in the list")


def int_list(arg):
    try:
        int_values = [int(val) for val in arg.split()]
        return int_values
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid int values in the list")


def add_dataset_args(parser):
    # Args that are used to create a dirichlet mixture partition of cifar10
    parser.add_argument(
        '--component_mean_user_dataset_length',
        type=int_list,
        default=50,
        help="Mean number of samples per user in each component")
    parser.add_argument('--component_phi',
                        type=float_list,
                        default=1.0,
                        help='True mixture component weights')
    parser.add_argument(
        '--component_alphas',
        type=float_list,
        default=0.1,
        help='The alpha value for each mixture component '
        '(all entries of vector/all categories take same value)')
    return parser


def add_init_algorithm_args(parser):
    parser.add_argument('--cohort_size_init_algorithm', type=int)
    parser.add_argument('--max_num_samples_mixture_component_init_algorithm',
                        type=int)
    parser.add_argument('--strategy', type=str, default='random')
    parser.add_argument('--central_num_iterations_init_algorithm',
                        type=int,
                        default=1)
    return parser


def add_algorithm_args(parser):
    parser.add_argument('--cohort_size_algorithm', type=int)
    parser.add_argument('--max_num_samples_mixture_component_algorithm',
                        type=int)
    parser.add_argument('--central_num_iterations_algorithm',
                        type=int,
                        default=10)
    return parser


def add_histogram_algorithm_args(parser):
    parser.add_argument('--cohort_size_histogram_algorithm', type=int)
    parser.add_argument('--central_num_iterations_histogram_algorithm',
                        type=int,
                        default=1)
    parser.add_argument('--num_bins_histogram', type=int, default=500)
    return parser


def add_mle_args(parser):
    # Add the arguments related to the solving of the Polya Mixture MLE
    parser.add_argument(
        '--num_mixture_components',
        type=int,
        default=1,
        help='Number of Polya mixture components to try to infer')
    return parser


def add_user_visualisation_args(parser):
    parser.add_argument('--cohort_size_visualization', type=int, default=100)
    parser.add_argument('--num_iterations_visualization', type=int, default=20)
    return parser


def add_flair_visualisation_args(parser):
    parser.add_argument('--use_fine_grained_labels',
                        action=store_bool,
                        default=False,
                        help='Whether to use fine-grained label taxonomy.')

    parser.add_argument('--max_num_user_images',
                        type=int,
                        default=100,
                        help='Maximum number of images per user')

    return parser
