# Copyright © 2023-2024 Apple Inc.
"""
Test log_float.
"""

import math
import operator

import numpy as np  # type: ignore
import pytest  # type: ignore

from pfl.internal.distribution import LogFloat


class TestLogFloat:

    def setup_class(self):
        self.zero = LogFloat(+1, -math.inf)
        self.example_log_values = [
            math.inf, 0.01, .5, .9, 1., 1.1, 1.5, 2.0, 3.0, 7.0, 33.
        ]
        self.examples = [self.zero] + [
            LogFloat(+1, log_value) for log_value in self.example_log_values
        ] + [LogFloat(-1, log_value) for log_value in self.example_log_values]
        self.nan = LogFloat(+1, float('nan'))

    def test_value(self):
        reference_values = [0] + [
            math.exp(log_value) for log_value in self.example_log_values
        ] + [-math.exp(log_value) for log_value in self.example_log_values]
        example_values = [example.value for example in self.examples]
        # Note that exact equality is expected here.
        assert example_values == reference_values

    def test_from_value(self):
        assert LogFloat.from_value(0) == self.zero
        assert LogFloat.from_value(1) == LogFloat(+1, 0)
        assert LogFloat.from_value(-1) == LogFloat(-1, 0)

    def test_equality(self):
        for sign in [-1, +1]:
            for value in [-math.inf, -10., -1., 0., +1, +math.inf]:
                l1 = LogFloat(sign, value)
                l2 = LogFloat(sign, value)
                negative = LogFloat(-sign, value)
                other_value = value + 1.
                other = LogFloat(sign, other_value)

                assert l1 == l2
                assert hash(l1) == hash(l2)

                if value == -math.inf:
                    continue

                assert l1 != negative
                assert hash(l1) != hash(negative)

                if other_value == value:
                    continue
                assert LogFloat(sign, value) != LogFloat(sign, other_value)
                assert LogFloat(sign, value) != LogFloat(-sign, other_value)
                assert l1 != other
                assert l1 != -other
                assert hash(l1) != hash(other)
                assert hash(l1) != hash(-other)

        # NaN is strange.
        assert self.nan != self.nan

    def test_comparison(self):
        for operation in [operator.lt, operator.le, operator.gt, operator.ge]:
            for left in self.examples:
                for right in self.examples:
                    assert (operation(left, right) == operation(
                        left.value, right.value))

    def test_negation(self):
        assert self.zero == -self.zero

        # Zero and NaN always have positive sign.
        assert (-self.zero).sign == +1
        assert (-self.nan).sign == +1

        for sign in [-1, +1]:
            for value in [-10., -1., 0., +1, +math.inf]:
                assert -LogFloat(sign, value) == LogFloat(-sign, value)

    def test_addition(self):
        for left in [*self.examples, self.nan]:
            for right in [*self.examples, self.nan]:
                result = left + right
                reference_value = left.value + right.value
                if np.isnan(reference_value):
                    assert np.isnan(result.value)
                else:
                    assert result.value == pytest.approx(reference_value)

    def test_subtraction(self):
        one = LogFloat.from_value(1)
        assert one - one == self.zero

        for left in [*self.examples, self.nan]:
            for right in [*self.examples, self.nan]:
                result = left - right
                reference_value = left.value - right.value
                if np.isnan(reference_value):
                    assert np.isnan(result.value)
                else:
                    assert result.value == pytest.approx(reference_value)

    def test_multiplication(self):
        for left in [*self.examples, self.nan]:
            for right in [*self.examples, self.nan]:
                result = left * right
                reference_value = left.value * right.value
                if np.isnan(reference_value):
                    assert np.isnan(result.value)
                else:
                    assert result.value == pytest.approx(reference_value)

    def test_division(self):
        for left in [*self.examples, self.nan]:
            for right in [*self.examples, self.nan]:
                if right == self.zero:
                    continue
                result = left / right
                reference_value = left.value / right.value
                if np.isnan(reference_value):
                    assert np.isnan(result.value)
                else:
                    assert result.value == pytest.approx(reference_value)

    def test_pow(self):
        for exponent in self.examples:
            for power in [0, 1, 2, -1, -5, -2.5, -.3, +.2, +1.6, +3.4]:

                try:
                    reference_value = exponent.value**power
                except ZeroDivisionError:
                    # Generalised division by zero.
                    with pytest.raises(ZeroDivisionError):
                        result = exponent**power
                    continue

                if round(power) != power and exponent < self.zero:
                    # Unimplemented so don't try this.
                    continue
                else:
                    result = exponent**power
                    assert result.value == pytest.approx(reference_value)
