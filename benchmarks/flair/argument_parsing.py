# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.


def add_flair_training_arguments(argument_parser):
    argument_parser.add_argument('--central_optimizer',
                                 choices=['adam', 'sgd'],
                                 default='adam',
                                 help='Central optimizer.')

    argument_parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay coefficient to avoid over-fitting')

    return argument_parser
