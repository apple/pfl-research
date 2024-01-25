# Copyright © 2023-2024 Apple Inc.
import math
from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class QuestionGenerator(ABC):
    """
    Abstract base class for generating questions to be used in training a GBDT
    with a federated algorithm.

    A question is defined as a "threshold" value, which is used to separate
    data points into two subsets: (1) where datapoint[feature] <= threshold;
    and (2) where  datapoint[feature] > threshold.

    The questions must be returned as an increasing sorted list.
    """

    @abstractmethod
    def generate(self, min_val: Union[float, int], max_val: Union[float, int],
                 num_questions: int) -> List[float]:
        """
        Generate questions to be asked for a feature in the federated
        algorithm to train a GBDT.

        Questions generated should be in the range [`min_val`, `max_val`]. The
        number of questions generated should be equal to `num_questions`.
        """
        pass


class FloatEquidistantQuestionGenerator(QuestionGenerator):
    """
    Generate questions for a feature of type float. The questions should be
    spaced equidistantly between `min_val` and `max_val`.
    """

    def generate(self, min_val: Union[float, int], max_val: Union[float, int],
                 num_questions: int) -> List[float]:
        return np.linspace(min_val, max_val, num_questions + 2).tolist()[1:-1]


class IntEquidistantQuestionGenerator(FloatEquidistantQuestionGenerator):
    """
    Generate questions for a feature of type int. The questions should be
    spaced equidistantly between `min_val` and `max_val`. For features of type
    `int`, there should only be one question for the same integral part.
    """

    def generate(self, min_val: Union[float, int], max_val: Union[float, int],
                 num_questions: int) -> List[float]:
        min_val, max_val = math.ceil(min_val), math.floor(max_val)
        if min_val >= max_val:
            return []
        num_questions = min(num_questions, max_val - min_val)
        return super().generate(min_val, max_val, num_questions)


class FloatRandomQuestionGenerator(QuestionGenerator):
    """
    Generate questions for a feature of type float. The questions are randomly
    chosen in the range [`min_val+offset`, `max_val-offset`].
    """

    def __init__(self, offset_fraction: float = 0.02):
        self._offset_fraction = offset_fraction

    def generate(self, min_val: Union[float, int], max_val: Union[float, int],
                 num_questions: int) -> List[float]:
        offset = self._offset_fraction * (max_val - min_val)
        return np.sort(
            np.random.uniform(min_val + offset, max_val - offset,
                              num_questions)).tolist()


class IntRandomQuestionGenerator(IntEquidistantQuestionGenerator):
    """
    Generate questions for a feature of type int. The questions are randomly
    selected from the range[`min_val`, `max_val`], and there should only be one
    question for the same integral part.
    """

    def generate(self, min_val: Union[float, int], max_val: Union[float, int],
                 num_questions: int) -> List[float]:
        min_val_ceiled, max_val_floored = math.ceil(min_val), math.floor(
            max_val)
        feature_range = max_val_floored - min_val_ceiled
        questions = super().generate(min_val_ceiled, max_val_floored,
                                     feature_range)
        if num_questions < len(questions):
            return np.random.choice(questions, num_questions,
                                    replace=False).tolist()
        return questions


class BoolQuestionGenerator(QuestionGenerator):
    """
    Generate a single question for a Boolean variable. Only one question will
    be generated, and this will lie at the midpoint of the range [`false_val`,
    `true_val`] for the Boolean variable.
    """

    def __init__(self, false_val, true_val):
        self._test_existing_split = lambda min_val, max_val: bool(
            min_val > false_val or max_val < true_val)
        self._question = [false_val + (true_val - false_val) / 2]

    def generate(self,
                 min_val: Union[float, int] = 0,
                 max_val: Union[float, int] = 1,
                 num_questions=1) -> List[float]:
        """
        If a split already exists on a Boolean feature, don't generate a new
        question, since it will not be able to further separate data. An
        existing split will be identified when `min_val` > `self._false_val`,
        and/or `max_val` > `self._true_val`.
        """
        if self._test_existing_split(min_val, max_val):
            return []
        return self._question
