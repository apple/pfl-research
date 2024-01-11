# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import argparse
import os

from utils.argument_parsing import (add_dnn_training_arguments,
                                    add_mechanism_arguments, store_bool)


def add_lm_arguments(argument_parser):
    argument_parser = add_dnn_training_arguments(argument_parser)
    argument_parser = add_mechanism_arguments(argument_parser)

    argument_parser.add_argument('--central_optimizer',
                                 choices=['sgd', 'adam'],
                                 default='sgd',
                                 help='Optimizer for central updates')

    known_args, _ = argument_parser.parse_known_args()
    if known_args.central_optimizer == 'adam':
        argument_parser.add_argument(
            '--adaptivity_degree',
            type=float,
            default=0.01,
            help='Degree of adaptivity (eps) in adaptive server optimizer.')

    argument_parser.add_argument('--local_lr_decay',
                                 action=store_bool,
                                 default=False,
                                 help='Use linear local learning rate decay')

    argument_parser.add_argument(
        '--central_lr_num_warmup_iterations',
        type=int,
        default=0,
        help='Number of iterations to warmup central learning rate.')

    argument_parser.add_argument(
        '--fedsgd_after_amount_trained',
        type=float,
        default=None,
        help=
        ('Fine-tune with FedSGD after training this fraction of the total '
         'number of central iterations. Don\'t fine-tune with FedSGD if None.'
         ))

    argument_parser.add_argument(
        '--use_tensorboard',
        action=store_bool,
        default=False,
        help='If enabled, write TensorBoard logs to disk '
        'and host a TensorBoard server.')

    return argument_parser
