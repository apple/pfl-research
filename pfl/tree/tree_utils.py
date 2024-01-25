# Copyright © 2023-2024 Apple Inc.
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np

from pfl.internal.tree import (
    BoolQuestionGenerator,
    FloatEquidistantQuestionGenerator,
    FloatRandomQuestionGenerator,
    IntEquidistantQuestionGenerator,
    IntRandomQuestionGenerator,
)
from pfl.internal.tree.questions import QuestionGenerator
from pfl.tree.gbdt_model import NodeRecord


class Feature:
    """
    Represents a feature of the data used to train a GBDT. The algorithm used
    to train a GBDT will select feature splits which optimally partition data
    to reduce the loss function used for training.

    :param feature_index:
        Index of the feature in the data's feature vector. Must be non-negative
    :param feature_range:
        Range of feature, [`min_val`, `max_val`].
    :param feature_type:
        Type of feature: {bool, float, int} are supported.
    :param num_questions:
        Number of questions to ask for this feature during training algorithm.
        Must be >= 0.
    :param question_choice_method:
        Method used to generate questions for this feature, to find optimal
        feature splits to partition the data. Either `equidistant` or `random`
        for int or float features. Boolean features have default splitting
        method.
    """

    def __init__(self,
                 feature_index: int,
                 feature_range: Tuple[Union[int, float], Union[int, float]],
                 feature_type: Union[Type[float], Type[int], Type[bool]],
                 num_questions: int,
                 question_choice_method: Optional[str] = None):
        assert feature_index >= 0, (
            'Feature index must be >= 0.',
            f'{feature_index} is not a valid feature index.')
        self._index = feature_index
        assert len(
            feature_range) == 2 and feature_range[0] < feature_range[1], (
                f'feature_range = {feature_range}',
                f'is not valid for feature {self._index}.')
        self._feature_range = feature_range
        assert feature_type in [
            int, float, bool
        ], 'Only variables of type int, float and bool are supported.'

        self._question_generator: QuestionGenerator
        if feature_type == bool:
            self._question_generator = BoolQuestionGenerator(
                self._feature_range[0], self._feature_range[1])
        elif feature_type == float:
            if question_choice_method == 'equidistant':
                self._question_generator = FloatEquidistantQuestionGenerator()
            elif question_choice_method == 'random':
                self._question_generator = FloatRandomQuestionGenerator()
            else:
                raise ValueError(('For float features, question_choice_method',
                                  'must be one of [equidistant, random]'))
        elif feature_type == int:
            if question_choice_method == 'equidistant':
                self._question_generator = IntEquidistantQuestionGenerator()
            elif question_choice_method == 'random':
                self._question_generator = IntRandomQuestionGenerator()
            else:
                raise ValueError(('For int features, question_choice_method',
                                  'must be one of [equidistant, random]'))

        assert num_questions >= 0 and isinstance(
            num_questions, int), ('num_questions must be an integer >= 0')
        self._num_questions = num_questions

    @property
    def index(self) -> int:
        return self._index

    @property
    def num_questions(self) -> int:
        return self._num_questions

    def _get_feature_range(self, node: NodeRecord):
        min_val, max_val = self._feature_range

        for (feature, threshold, is_left) in node.decision_path:
            if int(feature) != self._index:
                continue

            if is_left:
                max_val = threshold if max_val > threshold else max_val
                min_val = threshold if min_val > threshold else min_val
            else:
                min_val = threshold if min_val < threshold else min_val
                max_val = threshold if max_val < threshold else max_val

        return [min_val, max_val]

    def generate_feature_questions(
            self,
            node: NodeRecord,
            num_questions: Optional[int] = None) -> List[Union[int, float]]:
        """
        Generate questions for feature, to be used in training a tree.

        :param node:
            The node in a tree for which questions should be generated for this
            feature.
        :param num_questions:
            Optional parameter for number of questions to generate. If not
            None, it will override self._num_questions. Can be 0.
        """
        num_questions = (num_questions
                         if num_questions is not None else self._num_questions)
        min_val, max_val = self._get_feature_range(node)
        if min_val > max_val:
            return []

        return self._question_generator.generate(min_val, max_val,
                                                 num_questions)


def choose_questions(
        nodes: List[NodeRecord],
        features: List[Feature]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Select questions to be used to train a GBDT in each iteration of training.

    Note: leaf nodes only require a single question, as leaf nodes require the
    value of the node to be determined, which can be computed from the
    gradients from any split on the data, not the optimal split.

    :param nodes:
        Nodes for which questions should be generated so optimal parameters for
        these nodes can be computed, so these nodes can be added to trees in
        GBDT.
    :param features:
        List of Feature objects used to train a GBDT. Questions will be asked
        of each Feature to identify the feature-split which achieves the best
        partition of the dataset to reduce the loss function used for training.
    :returns:
        A tuple including a list comprising questions for each node in `nodes`,
        to be used for training, and the total number of questions asked. For
        each node, a dictionary is returned, which includes the path to the
        node, with the key `decisionPath`, and a dictionary mapping feature
        indices to threshold values: these are the questions.
    """
    all_questions = []
    total_num_questions = 0

    for node in nodes:
        node_questions = {}
        if node.is_leaf:
            while True:
                random_feature = features[np.random.choice(len(features))]
                questions = random_feature.generate_feature_questions(node, 1)
                if len(questions) == 1:
                    node_questions[random_feature.index] = questions
                    total_num_questions += 1
                    break
        else:
            for feature in features:
                questions = feature.generate_feature_questions(node)
                node_questions[feature.index] = questions
                total_num_questions += len(questions)

        all_questions.append({
            'decisionPath': node.decision_path,
            'splits': node_questions
        })

    return all_questions, total_num_questions
