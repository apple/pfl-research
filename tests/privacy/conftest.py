# Copyright © 2023-2024 Apple Inc.

import math

import pytest
from scipy.special import erf


def gaussian_mechanism_minimum_delta(epsilon, sigma, sensitivity):
    '''
    Compute the minimum delta for the Gaussian mechanism (one-shot) from the
    epsilon, sigma, and sensitivity.
    This parameterisation is only useful to check correctness, which is the
    point here.
    This exactly implements equation (6) from Balle and Wang (2018),
    "Improving the Gaussian Mechanism for Differential Privacy",
    https://arxiv.org/abs/1805.06530.
    '''

    def phi(t):
        '''
        The CDF of a unit Gaussian.
        '''
        return (1 + erf(t / math.sqrt(2.))) / 2

    # Compare the probabilities of the mechanism on two adjacent
    # databases yielding an output greater than epsilon*sigma/sensitivity.
    probability_database_1 = phi(sensitivity / (2 * sigma) -
                                 epsilon * sigma / sensitivity)
    probability_database_2 = phi(-sensitivity / (2 * sigma) -
                                 epsilon * sigma / sensitivity)
    scaled_probability_database_2 = math.exp(epsilon) * probability_database_2
    return probability_database_1 - scaled_probability_database_2


def pytest_configure():
    # Make this functions available to all tests in this directory.
    pytest.gaussian_mechanism_minimum_delta = gaussian_mechanism_minimum_delta
