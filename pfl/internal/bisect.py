# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
Perform bisection on a function whose range is not known.
"""

from typing import Callable, Tuple


def _sign(x: float) -> int:
    if x < 0:
        return -1
    elif x > 0:
        return +1
    return 0


def bisect_range(function: Callable[[float], float], start: float, end: float,
                 num_steps: int) -> Tuple[float, float]:
    """
    Use bisection to find where the output of a function crosses 0, and return
    a lower bound and a higher bound.

    :param function:
        The function for which the root is to be found.
    :param start:
        A point on one side of the root.
    :param end:
        A point on the other side of the root.
    :param num_steps:
        The number of bisection steps to take before returning.

    :return:
        A tuple of the final lower and upper bounds.
    """
    start_sign = _sign(function(start))
    end_sign = _sign(function(end))
    assert start_sign * end_sign == -1
    for _ in range(num_steps):
        middle = .5 * start + .5 * end
        middle_sign = _sign(function(middle))
        if start_sign != middle_sign:
            end = middle
            end_sign = middle_sign
        else:
            start = middle
            start_sign = middle_sign

    return start, end


def bisect_automatic_range(function: Callable[[float], float],
                           start: float,
                           initial_distance: float,
                           num_steps: int = 50) -> Tuple[float, float]:
    """
    Use bisection to find where the output of a function crosses 0.
    But first, increase the step size to find the search area.

    :param function:
        The function for which the root is to be found.
    :param start:
        A point on one side of the root.
    :param initial_distance:
        A distance from ``start`` which is an initial guess of where the sign of
        the function might be different.
        If the sign is not different between ``start`` and
        ``start+initial_distance``, then increase the distance exponentially
        until the signs are different.
    :param num_steps:
        The number of bisection steps to take before returning.

    :return:
        A tuple of the final lower and upper bounds.
    """
    start_value = function(start)
    start_sign = _sign(start_value)

    distance = initial_distance
    while start_sign * _sign(function(start + distance)) != -1:
        distance *= 2

    end = start + distance

    return bisect_range(function, start, end, num_steps)
