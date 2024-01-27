# Copyright © 2023-2024 Apple Inc.
import math

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.internal.tree import (
    BoolQuestionGenerator,
    FloatEquidistantQuestionGenerator,
    FloatRandomQuestionGenerator,
    IntEquidistantQuestionGenerator,
    IntRandomQuestionGenerator,
)


@pytest.fixture()
def float_equidistant_question_generator(scope='module'):
    return FloatEquidistantQuestionGenerator()


@pytest.fixture()
def int_equidistant_question_generator(scope='module'):
    return IntEquidistantQuestionGenerator()


@pytest.fixture()
def float_random_question_generator(scope='module'):
    return FloatRandomQuestionGenerator()


@pytest.fixture()
def int_random_question_generator(scope='module'):
    return IntRandomQuestionGenerator()


@pytest.fixture()
def bool_question_generator(scope='module'):
    return BoolQuestionGenerator(false_val=0, true_val=1)


class TestQuestionGenerators:

    @pytest.mark.parametrize(
        'question_generator, min_val, max_val, num_questions, expected', [
            (lazy_fixture('float_equidistant_question_generator'), 0, 1, 9,
             [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            (lazy_fixture('float_equidistant_question_generator'), -1, 1, 7,
             [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]),
            (lazy_fixture('float_equidistant_question_generator'), -1, 1, 0,
             []),
            (lazy_fixture('int_equidistant_question_generator'), 0, 2, 20,
             [0.67, 1.33]),
            (lazy_fixture('int_equidistant_question_generator'), 0, 2, 1,
             [1.0]),
            (lazy_fixture('int_equidistant_question_generator'), 0, 2, 2,
             [0.67, 1.33]),
            (lazy_fixture('int_equidistant_question_generator'), 5, 8, 2,
             [6.0, 7.0]),
            (lazy_fixture('int_equidistant_question_generator'), 5, 8, 4,
             [5.75, 6.5, 7.25]),
            (lazy_fixture('int_equidistant_question_generator'), 5, 8, 0, []),
            (lazy_fixture('int_equidistant_question_generator'), 0, 0.6, 1,
             []),
            (lazy_fixture('int_equidistant_question_generator'), 0.1, 1, 1,
             []),
            (lazy_fixture('int_equidistant_question_generator'), 0, 1.6, 1,
             [0.5]),
        ])
    def test_equidistant_question_generator(self, question_generator, min_val,
                                            max_val, num_questions, expected):
        questions = question_generator.generate(min_val, max_val,
                                                num_questions)
        np.testing.assert_almost_equal(questions, expected, decimal=2)

    @pytest.mark.parametrize(
        'question_generator, min_val, max_val, num_questions', [
            (lazy_fixture('float_random_question_generator'), 0, 5, 20),
            (lazy_fixture('float_random_question_generator'), -1, 1, 7),
            (lazy_fixture('float_random_question_generator'), -1, 1, 0),
        ])
    def test_float_random_question_generator(self, question_generator, min_val,
                                             max_val, num_questions):
        questions = question_generator.generate(min_val, max_val,
                                                num_questions)
        assert len(questions) == num_questions
        offset = question_generator._offset_fraction * (max_val - min_val)  # pylint: disable=protected-access
        assert all(q > (min_val + offset) and q < (max_val - offset)
                   for q in questions)
        assert [type(q) == float for q in questions] or len(questions) == 0
        assert len(set(questions)) == len(questions)

    @pytest.mark.parametrize(
        (
            'question_generator, min_val, max_val, num_questions, expected_num_questions'  # pylint: disable=line-too-long
        ),
        [
            (lazy_fixture('int_random_question_generator'), 0, 5, 20, 5),
            (lazy_fixture('int_random_question_generator'), -1, 1, 7, 2),
            (lazy_fixture('int_random_question_generator'), -1, 1, 0, 0),
            (lazy_fixture('int_random_question_generator'), 0.1, 0.6, 1, 0),
            (lazy_fixture('int_random_question_generator'), -0.1, 1.1, 3, 1),
            (lazy_fixture('int_random_question_generator'), 0, 1.1, 3, 1),
            (lazy_fixture('int_random_question_generator'), -1, 1.1, 3, 2),
        ])
    def test_int_random_question_generator(self, question_generator, min_val,
                                           max_val, num_questions,
                                           expected_num_questions):
        questions = question_generator.generate(min_val, max_val,
                                                num_questions)
        assert len(questions) == expected_num_questions
        assert len(set(questions)) == len(questions)  # all unique questions
        assert all(q > min_val and q < max_val for q in questions)
        assert len({math.floor(q)
                    for q in questions
                    }) == len(questions)  # one question max per whole number

    @pytest.mark.parametrize(
        'question_generator, min_val, max_val, num_questions', [
            (lazy_fixture('bool_question_generator'), 0, 1, 1),
            (lazy_fixture('bool_question_generator'), -1, 1, 7),
            (lazy_fixture('bool_question_generator'), -2, 1, 0),
        ])
    def test_bool_question_generator(self, question_generator, min_val,
                                     max_val, num_questions):
        questions = question_generator.generate(min_val, max_val,
                                                num_questions)
        assert len(questions) == 1
        assert questions[0] > min_val and questions[0] < max_val

    def test_bool_question_generator_smaller_range(self,
                                                   bool_question_generator):
        questions = bool_question_generator.generate(0, 0.5, 1)
        assert questions == []
        questions = bool_question_generator.generate(0.5, 1, 1)
        assert questions == []
