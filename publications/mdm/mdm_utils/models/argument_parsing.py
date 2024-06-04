import argparse
from typing import Optional, Tuple


def add_model_arguments(
        parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add `model_name` argument to parser and add
    model-specific arguments depending on the model specified in
    `model_name` argument.
    """

    parser.add_argument(
        '--model_name',
        choices=[
            'simple_cnn', 'simple_dnn', 'resnet18', 'lm_lstm',
            'lm_transformer', 'multi_label_cnn'
        ],
        default='simple_cnn',
        help='Which model to train. See models.py for definitions.')

    # Get the value of `model_name` argument and dynamically add
    # arguments depending on which model is chosen.
    known_args, _ = parser.parse_known_args()

    if known_args.model_name in {'lm_lstm', 'lm_transformer'}:
        parser.add_argument("--embedding_size",
                            type=int,
                            required=True,
                            help='Number of dimensions in embedding layer.')

    if known_args.model_name == 'lm_lstm':
        parser.add_argument("--num_cell_states",
                            type=int,
                            required=True,
                            help='Number of cell states in each LSTM.')

        parser.add_argument("--num_lstm_layers",
                            type=int,
                            required=True,
                            help='Number of stacked LSTM layers.')

    if known_args.model_name == 'lm_transformer':
        parser.add_argument("--hidden_size",
                            type=int,
                            required=True,
                            help='Number of hidde states in each Transformer.')

        parser.add_argument(
            "--num_heads",
            type=int,
            required=True,
            help='Number of heads in multi-head attention layers.')

        parser.add_argument(
            '--feedforward_size',
            type=int,
            required=True,
            help='Number of feed forward hidden states in each Transformer.')

        parser.add_argument("--num_transformer_layers",
                            type=int,
                            required=True,
                            help='Number of stacked Transformer layers.')

        parser.add_argument(
            '--dropout_rate',
            type=float,
            default=0.1,
            help='Dropout rate applied in the Transformer model.')

    if known_args.model_name == 'multi_label_cnn':
        _torchvision_architectures = [
            'alexnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
            'wide_resnet50_2', 'wide_resnet101_2', 'vgg11', 'vgg11_bn',
            'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',
            'squeezenet1_0', 'squeezenet1_1', 'inception_v3', 'densenet121',
            'densenet169', 'densenet201', 'densenet161', 'googlenet',
            'mobilenet_v2', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0',
            'mnasnet1_3', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
            'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
        ]

        parser.add_argument('--model_type',
                            choices=_torchvision_architectures,
                            help='Model architecture.')

    return parser


def _get_model_dims_for_dataset(
        dataset_name: str) -> Tuple[Optional[Tuple[int, ...]], Optional[int]]:
    """
    Get the correct input shape and number of outputs for the
    specified dataset.
    """
    if dataset_name == 'femnist':
        input_shape = (28, 28, 1)
        num_outputs = 62
    elif dataset_name == 'femnist_digits':
        input_shape = (28, 28, 1)
        num_outputs = 10
    elif dataset_name in ['cifar10', 'cifar10_iid']:
        input_shape = (32, 32, 3)
        num_outputs = 10
    else:
        input_shape = None
        num_outputs = None

    return input_shape, num_outputs


def get_model_tf2(args: argparse.Namespace):
    """
    Initialize the TensorFlow v2 model specified by ``args.model_name`` with
    other required arguments also available in ``args``.
    Use ``add_model_arguments`` to dynamically add arguments required by
    the selected model.
    """
    assert 'model_name' in vars(args)
    from . import tf2

    input_shape, num_outputs = _get_model_dims_for_dataset(args.dataset)

    model_name = args.model_name.lower()
    if model_name == 'dnn':
        model = tf2.dnn(input_shape, args.hidden_dims, num_outputs)
    elif model_name == 'simple_dnn':
        model = tf2.simple_dnn(input_shape, num_outputs)
    elif model_name == 'simple_cnn':
        model = tf2.simple_cnn(input_shape, num_outputs)
    elif model_name == 'resnet18':
        model = tf2.resnet18(input_shape, num_outputs)
    elif model_name == 'lm_lstm':
        model = tf2.lm_lstm(args.embedding_size, args.num_cell_states,
                            args.num_lstm_layers, args.vocab_size)
    elif model_name == 'lm_transformer':
        model = tf2.lm_transformer(args.embedding_size, args.hidden_size,
                                   args.num_heads, args.feedforward_size,
                                   args.num_transformer_layers,
                                   args.vocab_size, args.max_sequence_length,
                                   args.dropout_rate)
    else:
        raise TypeError(f'Model {model_name} not implemented for TF2.')

    return model


def get_model_pytorch(args: argparse.Namespace):
    """
    Initialize the PyTorch model specified by ``args.model_name`` with
    other required arguments also available in ``args``.
    Use ``add_model_arguments`` to dynamically add arguments required by
    the selected model.
    """
    assert 'model_name' in vars(args)
    from . import pytorch

    input_shape, num_outputs = _get_model_dims_for_dataset(args.dataset)

    model_name = args.model_name.lower()

    if model_name == 'dnn':
        model = pytorch.dnn(input_shape, args.hidden_dims, num_outputs)
    elif model_name == 'simple_dnn':
        model = pytorch.simple_dnn(input_shape, num_outputs)
    elif model_name == 'simple_cnn':
        model = pytorch.simple_cnn(input_shape, num_outputs)
    elif model_name == 'lm_lstm':
        model = pytorch.lm_lstm(args.embedding_size, args.num_cell_states,
                                args.num_lstm_layers, args.vocab_size,
                                args.pad_symbol, args.unk_symbol)
    elif model_name == 'lm_transformer':
        model = pytorch.lm_transformer(
            args.embedding_size, args.hidden_size, args.num_heads,
            args.feedforward_size, args.num_transformer_layers,
            args.vocab_size, args.max_sequence_length, args.pad_symbol,
            args.unk_symbol, args.dropout_rate)
    elif model_name == 'multi_label_cnn':
        model = pytorch.multi_label_cnn(args.model_type, args.num_classes,
                                        args.channel_mean,
                                        args.channel_stddevs, args.pretrained)
    else:
        raise TypeError(f'Model {model_name} not implemented for PyTorch.')
    return model
