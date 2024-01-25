# Copyright © 2023-2024 Apple Inc.
"""
Represent real value by their logarithms in floating-point format.
"""

import math


def log(value):
    """
    :return: The natural logarithm of ``value``, or -math.inf if ``value==0``.
    """
    if value == 0:
        return -math.inf
    else:
        return math.log(value)


class LogFloat:
    """
    A real number represented by its logarithm in floating-point format, and a
    sign.
    The sign is always either -1 or +1.
    If the value represented is 0, then the sign is always +1.

    This can deal with a much larger dynamic range than a standard float.
    This is useful when computing likelihoods of high-dimensional data, such as
    sequences: it prevents underflow (or overflow).
    Various mathematical functions return natural logarithms to prevent
    overflow.
    See e.g. ``scipy.special.gammaln``.
    """

    def __init__(self, sign: int, log_value: float):
        if log_value == -math.inf or math.isnan(log_value):
            sign = +1
        self._sign: int = sign
        self._log_value: float = log_value

    @classmethod
    def from_value(cls, value: float):
        """
        Construct a ``LogFloat`` from its value as a floating-point number.
        """
        sign = +1 if (value >= 0) else -1
        return cls(sign, log(abs(value)))

    @property
    def sign(self) -> int:
        """
        :return: The sign (-1 or +1) of the value.
        """
        return self._sign

    @property
    def log_value(self) -> float:
        """
        :return: The logarithm of the absolute value.
        """
        return self._log_value

    @property
    def value(self) -> float:
        """
        :return:
            The value contained, converted to a plain floating-point
            representation.
        """
        return self.sign * math.exp(self._log_value)

    def __repr__(self) -> str:
        return f'LogFloat({self.sign}, exp({self.log_value}))'

    def __str__(self) -> str:
        return f'{self.value}'

    def __hash__(self):
        return hash((self._sign, self._log_value))

    def __eq__(self, other):
        if isinstance(other, LogFloat):
            return (self.sign == other.sign
                    and self.log_value == other.log_value)
        return self.value == other

    def __lt__(self, other):
        if isinstance(other, LogFloat):
            if self.sign == -1:
                if other.sign == +1:
                    return True
                return self.log_value > other.log_value
            else:
                assert self.sign == +1
                if other.sign == -1:
                    return False
                return self.log_value < other.log_value
        return self.value < other

    def __le__(self, other):
        return self == other or self < other

    def __neg__(self):
        if self.log_value == -math.inf:
            # -0 == 0; the sign for zero must be positive (arbitrarily).
            return self
        return LogFloat(-self.sign, self.log_value)

    def _perform_addition(self, sign_a: int, log_a: float, sign_b: int,
                          log_b: float) -> 'LogFloat':
        # Without loss of generality, assume a > b.
        # log(a + b) in terms of log_a and log_b:
        # log(exp(log_a) + exp(log_b))
        # = log(exp(log_a) * (1 + exp(log_b)/exp(log_a)))
        # = log(exp(log_a)) + log(1 + exp(log_b-log_a)))
        # = log_a + log1p(exp(log_b-log_a))
        #
        # Similarly:
        # log(a-b) = log_a + logp1(- exp(log_b-log_a))
        #
        # log(b-a) : log(a-b) with a minus sign.

        if math.isnan(log_a) or math.isnan(log_b):
            # Catch NaN: return NaN.
            return LogFloat(+1, log_a + log_b)

        # Make sure that log_a >= log_b
        if log_a < log_b:
            return self._perform_addition(  # pylint: disable=arguments-out-of-order
                sign_b, log_b, sign_a, log_a)

        assert log_a >= log_b

        def log_add(log_a: float, log_b: float) -> float:
            assert log_a >= log_b
            return log_a + math.log1p(math.exp(log_b - log_a))

        def log_subtract(log_a: float, log_b: float) -> float:
            assert log_a >= log_b
            quotient = math.exp(log_b - log_a)
            if quotient == 1:
                return -math.inf
            return log_a + math.log1p(-quotient)

        # Special case: +- 0.
        if log_b == -math.inf:
            return LogFloat(sign_a, log_a)
        # Special case: math.inf + math.inf.
        if sign_a == sign_b and log_a == log_b == +math.inf:
            return LogFloat(sign_a, log_a)

        log_value = log_add(
            log_a, log_b) if sign_a == sign_b else log_subtract(log_a, log_b)

        # log_a dominates.
        sign = sign_a
        return LogFloat(sign, log_value)

    def __add__(self, other: 'LogFloat') -> 'LogFloat':
        return self._perform_addition(self.sign, self.log_value, other.sign,
                                      other.log_value)

    def __sub__(self, other: 'LogFloat') -> 'LogFloat':
        return self._perform_addition(self.sign, self.log_value, -other.sign,
                                      other.log_value)

    def __mul__(self, other: 'LogFloat') -> 'LogFloat':
        return LogFloat(self.sign * other.sign,
                        self.log_value + other.log_value)

    def __truediv__(self, other: 'LogFloat') -> 'LogFloat':
        return LogFloat(self.sign * other.sign,
                        self.log_value - other.log_value)

    def __pow__(self, power: float) -> 'LogFloat':
        if power == 0:
            # Return 1, even for 0**0 (by usual definition in computer
            # languages).
            return LogFloat(+1, 0)
        if self.log_value == -math.inf and power < 0:
            raise ZeroDivisionError('0 cannot be raised to a negative power')
        if self.sign == -1 and round(power) != power:
            raise NotImplementedError()

        # log(sign * exp(loga)**p) == log(sign**p * exp(p*loga))
        return LogFloat(int(self.sign**power), self.log_value * power)
