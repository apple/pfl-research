# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np

from pfl.algorithm import FederatedAlgorithm
from pfl.callback import TrainingProcessCallback
from pfl.common_types import Population
from pfl.context import CentralContext
from pfl.data.dataset import TabularDataset
from pfl.hyperparam import AlgorithmHyperParams, HyperParam, get_param_value
from pfl.metrics import MetricName, Metrics, TrainMetricName, Weighted, get_overall_value
from pfl.stats import MappedVectorStatistics
from pfl.tree.gbdt_model import GBDTModel, GBDTModelHyperParamsType, GBDTModelType
from pfl.tree.tree_utils import Feature, choose_questions

# pylint: disable=too-many-lines

STATISTIC_INFO_NAME = 'GBDT_training_statistics'


class GBDTClippingBound(HyperParam, TrainingProcessCallback):
    """
    Adapt clipping bound based on current layer and current index of tree being
    trained in a GBDT.

    The sensitivity of the vector of gradients gets smaller as training
    progresses due to smaller gradients as predictions improve. Adapting the
    clipping bound during training improves the SNR.

    Clipping bound is adapted as follows:
    `clipping_bound = base_value * layer_multiplier ** current_depth * tree_multiplier ** (current_tree - num_trees_in_model_at_start_of_training)`  # pylint: disable=line-too-long
    where base_value is the base clipping bound value, layer_multiplier and
    tree_multiplier are parameters controlling the impact of layer depth and
    number of trees on the clipping bound, current_depth is the depth of the
    current tree being trained, and current_tree is the id of the current tree
    being trained (starting at 0).

    .. warning::
        You must include a GBDTClippingBound() object in the list of callbacks
        input to `backend` to ensure the clipping bound is actually updated on
        each iteration that the algorithm is run.

    :param base_value:
        Default value for clipping bound
    :param layer_multiplier:
        Factor used to modify the base value of clipping bound depending on
        the layer of a tree being trained. Should be in range (0, 1].
    :param tree_multiplier:
        Factor used to modify the base value for clipping bound depending on
        the layer of a tree being trained. Should be in range (0, 1].
    :param use_tree_offset:
        Whether or not to include trees in GBDT initialization model when
        computing clipping bound. A random initialization of a GBDT has zero
        trees. By default do include all trees as gradients depend on number
        of trees in GBDT, not on number of trees currently being trained.
    """

    def __init__(self,
                 base_value: float,
                 layer_multiplier: float = 1.,
                 tree_multiplier: float = 1.,
                 use_tree_offset: bool = False):
        assert base_value > 0, (f'Invalid base_value {base_value}',
                                'for clipping bound.')
        assert layer_multiplier > 0 and layer_multiplier <= 1, (
            'Invalid layer_multiplier',
            f'{layer_multiplier} for clipping bound.')
        assert tree_multiplier > 0 and tree_multiplier <= 1, (
            'Invalid tree_multiplier',
            f'{tree_multiplier} for clipping bound.')
        self._value = base_value
        self._layer_multiplier = layer_multiplier
        self._tree_multiplier = tree_multiplier
        self._use_tree_offset = use_tree_offset
        self._tree_offset = 0

    def value(self) -> float:
        return self._value

    def _set_clipping_bound(self, model: GBDTModel) -> None:
        self._value = (
            self.base_value * self._layer_multiplier**model.current_depth *
            self._tree_multiplier**(model.current_tree - self._tree_offset))

    def on_train_begin(self, *, model: GBDTModel) -> Metrics:
        assert model.current_tree >= 0, 'Current tree of GBDTModel must be >= 0'
        self._tree_offset = model.current_tree if self._use_tree_offset else 0
        self._set_clipping_bound(model)
        return Metrics()

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: GBDTModel, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        self._set_clipping_bound(model)

        return False, Metrics()


@dataclass(frozen=True)
class GBDTAlgorithmHyperParams(AlgorithmHyperParams):
    """
    Parameters for federated GBDT algorithms.

    :param cohort_size:
        Size of cohort used for each training iteration.
        Note that initial value of cohort_size may be
        modified by FederatedGBDT algorithm during training
        of a GBDT.
    :param val_cohort_size:
        Size of val cohort used for distributed evaluation
        during training of a GBDT. If set to 0, then there will
        not be any evaluation.
    :param num_trees:
        Number of trees to train.
    :param cohort_size_per_layer_modifier_fn:
        Define function to use to modify the base value of the training cohort
        size depending on the layer of a tree being trained. Options include
        {'none', 'linear', 'power'}, which, respectively, correspond to: not
        adapting the cohort size; linearly increasing the cohort size with the
        layer being trained; exponentially increasing the cohort size with the
        layer being trained. 'power' is the default setting, as this achieves
        the correct SNR during training. See `_compute_cohort_size` for
        more details.
    :param compute_second_order_gradients:
        Boolean determining whether or not second order
        gradients should be computed and reported on device.
    :param report_gradients_both_sides_split:
        Boolean determining whether or not to report
        gradients on both sides of each question/split asked.
    :param report_node_sum_gradients:
        Boolean determining whether to report sum of
        gradients in node, or else right gradient for one
        question asked of node, if
        report_gradients_both_sides_split = False.
    :param report_per_feature_result_difference:
        Report difference in gradients of questions
        asked for each feature in each node.
    :param report_per_node_result_difference:
        Report difference in gradients for all
        questions asked for each node.
    :param leaf_nodes_reduction_factor:
        Defines the factor by which to reduce the training cohort size from
        the base value when training the maximum depth of the tree, which
        comprises only leaf nodes. The default value is 1, i.e. no reduction
        takes place. However, this can be set to
        `total_num_questions/2^(max_depth - 1)`, where `total_num_questions` is
        the sum of the number of questions specified for each feature used for
        training. `leaf_nodes_reduction_factor` must be an integer >= 1.
    """
    cohort_size: int
    val_cohort_size: int
    num_trees: int
    cohort_size_per_layer_modifier_fn: str = 'power'
    l2_regularization: float = 1
    leaf_nodes_reduction_factor: int = 1
    compute_second_order_gradients: bool = True
    report_gradients_both_sides_split: bool = True
    report_node_sum_gradients: bool = False
    report_per_feature_result_difference: bool = False
    report_per_node_result_difference: bool = False

    def __post_init__(self):
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        assert self.num_trees > 0, 'num_trees must be > 0'
        assert self.l2_regularization >= 0, 'l2 regularization should be >= 0'
        assert isinstance(
            self.cohort_size, int
        ) and self.cohort_size >= 0, 'cohort size must be an integer >= 0'
        assert isinstance(self.val_cohort_size, int)
        assert self.val_cohort_size >= 0, (
            'validation cohort size must be an integer >= 0')
        assert self.cohort_size_per_layer_modifier_fn in [
            'none', 'linear', 'power'
        ], (f'{self.cohort_size_per_layer_modifier_function} is not a',
            'valid value for cohort_size_per_layer_modifier_function')
        assert isinstance(
            self.leaf_nodes_reduction_factor,
            int) and self.leaf_nodes_reduction_factor >= 1, (
                'leaf_nodes_reduction_factor must be an integer >= 1')


@dataclass(frozen=True)
class _GBDTInternalAlgorithmHyperParams(AlgorithmHyperParams):

    cohort_size: int
    val_cohort_size: int
    num_trees: int
    cohort_size_leaf_nodes: int
    cohort_size_per_layer_modifier_fn: str = 'power'
    l2_regularization: float = 1
    leaf_nodes_reduction_factor: int = 1
    compute_first_order_gradients: bool = True
    compute_second_order_gradients: bool = True
    report_gradients_both_sides_split: bool = True
    report_node_sum_gradients: bool = False
    report_per_feature_result_difference: bool = False
    report_per_node_result_difference: bool = False

    # managed internally by the FederatedGBDT algorithm
    gbdt_questions: List = field(default_factory=list)
    weight_vector: Optional[Union[np.ndarray, List]] = None
    translate_vector: Optional[Union[np.ndarray, List]] = None

    def __post_init__(self):
        if hasattr(super(), '__post_init__'):
            super().__post_init__()
        assert self.num_trees > 0, 'num_trees must be > 0'
        assert self.l2_regularization >= 0, 'l2 regularization should be >= 0'

    @classmethod
    def from_GBDTAlgorithmHyperParams(
            cls, algorithm_params: GBDTAlgorithmHyperParams, **kwargs):
        current_params = {
            k: get_param_value(p)
            for k, p in algorithm_params.to_context_dict().items()
        }
        params = {**current_params, **kwargs}
        params['cohort_size_leaf_nodes'] = math.ceil(
            params['cohort_size'] // params['leaf_nodes_reduction_factor'])

        return _GBDTInternalAlgorithmHyperParams(**params)


GBDTAlgorithmHyperParamsType = TypeVar('GBDTAlgorithmHyperParamsType',
                                       bound=GBDTAlgorithmHyperParams)
FederatedGBDTCentralContextType = CentralContext[GBDTAlgorithmHyperParamsType,
                                                 GBDTModelHyperParamsType]
_GBDTInternalAlgorithmHyperParamsType = TypeVar(
    '_GBDTInternalAlgorithmHyperParamsType',
    bound='_GBDTInternalAlgorithmHyperParams')


class FederatedGBDT(FederatedAlgorithm[GBDTAlgorithmHyperParamsType,
                                       GBDTModelHyperParamsType, GBDTModelType,
                                       MappedVectorStatistics,
                                       TabularDataset]):

    def __init__(self, features: List[Feature]):
        super().__init__()
        self._features = features

        self._finish_tree_id = 0

    def _train_one_user(
            self, model, user_dataset, algorithm_params, train_params,
            name_formatting_fn) -> Tuple[MappedVectorStatistics, Metrics]:
        """
        Compute training statistics from one simulated user.

        Compute sum of left and right gradients on the user's dataset for
        questions asked of nodes in a tree of a GBDT to be split.

        First and/or second order gradients can be computed and reported.

        Predictions used to compute gradients are computed from all trees in
        GBDT except the latest tree if it is not completely trained.

        :param user_dataset:
            The dataset, representing one user, on which training should take
            place. Must be of type TabularDataset.
        :param train_params:
            Indicates the computation and evaluation to take place on the
            user's data.
        :param metric_name_current:
            A function that produces a ``TrainMetricName`` object from a simple
            string for metrics computed on user's data.
        :param metric_name_final:
            Not used in GBDT training, as model is not updated locally based on
            user's data.

        :returns:
            Tuple including flat vector of gradients and metrics computed on
            user's data.
        """

        assert (algorithm_params.compute_first_order_gradients
                or get_param_value(
                    algorithm_params.compute_second_order_gradients)), (
                        'At least one of 1st and 2nd order ',
                        'gradients must be computed on device')

        input_vectors = user_dataset.features
        assert isinstance(input_vectors, np.ndarray)
        targets = np.ravel(user_dataset.labels)

        # TODO evaluation_threshold should be input to model.predict() function
        # - only relevant for classification, not regression.
        predictions = model.predict(input_vectors)

        result = []

        for (flag, compute_gradient) in zip([
                algorithm_params.compute_first_order_gradients,
                get_param_value(
                    algorithm_params.compute_second_order_gradients)
        ], [
                model.compute_first_order_gradient,
                model.compute_second_order_gradient
        ]):
            if flag:
                gradients = [
                    compute_gradient(target, prediction)
                    for target, prediction in zip(targets, predictions)
                ]

                result.extend(
                    self._evaluate_result_with_gradients(
                        gradients, algorithm_params.gbdt_questions,
                        input_vectors,
                        algorithm_params.report_gradients_both_sides_split,
                        algorithm_params.report_node_sum_gradients,
                        algorithm_params.report_per_feature_result_difference,
                        algorithm_params.report_per_node_result_difference))

        translated_result = self._translate_result(
            algorithm_params.translate_vector, result)
        weighted_result = self._weight_result(algorithm_params.weight_vector,
                                              translated_result)

        metrics = Metrics()
        metrics[name_formatting_fn(
            'l1 norm of vector of statistics')] = Weighted.from_unweighted(
                np.linalg.norm(weighted_result, ord=1))
        metrics[name_formatting_fn(
            'l2 norm of vector of statistics')] = Weighted.from_unweighted(
                np.linalg.norm(weighted_result, ord=2))

        return MappedVectorStatistics(
            {STATISTIC_INFO_NAME: np.array(weighted_result)}), metrics

    def _find_node_samples(
            self, decision_paths: List[List], input_vectors: np.ndarray,
            gradients: List[float]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Determines data points which satisfy all conditions in a
        decision path, leading to a particular node in a tree.

        Returns the input_vectors and gradients of the data points
        satisfying the decision path.
        """
        node_input_vectors = []
        node_gradients = []

        for (input_vector, gradient) in zip(input_vectors, gradients):
            satisfies_decision_path = True

            for (feature, threshold, is_left) in decision_paths:
                if is_left:
                    if input_vector[feature] > threshold:
                        satisfies_decision_path = False
                        break
                else:
                    if input_vector[feature] <= threshold:
                        satisfies_decision_path = False
                        break

            if satisfies_decision_path:
                node_input_vectors.append(input_vector)
                node_gradients.append(gradient)

        return node_input_vectors, node_gradients

    def _compute_sum_gradients_left_and_right_of_split(
            self, feature: int, threshold: float,
            input_vectors: List[np.ndarray], gradients: List[float]):
        """
        Compute the sum of the gradients of all data points on the left and
        right sides of a feature-threshold split.

        A datapoint is left of a split if its value at the split feature is
        less than or equal to the split threshold value. Otherwise, a data point
        is right of a split.

        Return the sum of the gradients on the left side of the split and the
        sum of the gradients on the right side of the split.
        """
        left_sum_gradients = 0.
        right_sum_gradients = 0.

        for (input_vector, gradient) in zip(input_vectors, gradients):
            if input_vector[int(feature)] <= threshold:
                left_sum_gradients += gradient
            else:
                right_sum_gradients += gradient
        return left_sum_gradients, right_sum_gradients

    def _evaluate_result_with_gradients(
            self, gradients: List[float], questions: List,
            input_vectors: np.ndarray, report_gradients_both_sides_split: bool,
            report_node_sum_gradients: bool,
            report_per_feature_result_difference: bool,
            report_per_node_result_difference: bool):
        """
        Compute local training statistics using data points, their gradients
        and questions.

        This function returns the sum of the left and right gradients of all
        data points who reach a particular node of a tree, for all the
        questions asked of this node.

        1. Iterate over all nodes on which questions are being asked.
        2. For each node, find the data points which satisfy the decision
        path conditions to reach this node of a tree.
        3. For each question asked for a particular node, sum the gradients
        of the samples on the left and right sides of the feature-threshold
        question split.
        4. Post-process results to reduce magnitude so less DP noise needs to
        be added. This is done by reporting the difference in successive
        elements of a result vector, only reporting left gradients and total
        gradients in a node, or reporting right gradients of only one question
        in a node.

        Each decision path is an array with elements consisting of 3 items,
        ``[<feature>, <threshold>, <is_left>]``, representing the tests
        defined by the path to a particular node in a tree.

        To ensure results are reported for questions in correct order on all
        devices, questions are sorted by increasing feature value. Note that
        the order of the questions with the same feature value and different
        threshold values is fixed, since thresholds for each feature are stored
        in an array, whose order is fixed.

        :param report_gradients_both_sides_split:
            Determines whether or not to report gradients on both left and
            right sides of a split.
        :param report_node_sum_gradients:
            Determines whether to report sum of a gradients in a node. Only
            used if ``report_gradients_both_sides_split=False``.
        :param report_per_feature_result_difference:
            Determines whether or not to report difference in successive values
            in the result array for all questions for a particular feature.
            This reduces the magnitude of the result vector so less DP noise
            needs to be added.
        :param report_per_node_result_difference:
            Determines whether or not to report difference in successive values
            in the result array for all questions for a particular node. This
            reduces the magnitude of the result vector so less DP noise needs
            to be added.
        """
        result = []

        for node_questions in questions:
            decision_path = node_questions['decisionPath']
            test_splits = node_questions['splits']

            node_input_vectors, node_gradients = self._find_node_samples(
                decision_path, input_vectors, gradients)

            left_node_gradients = []
            right_node_gradients = []

            features = sorted(test_splits.keys(), key=lambda x: int(x))
            for feature in features:
                thresholds = test_splits[feature]

                left_feature_gradients = []
                right_feature_gradients = []

                for threshold in thresholds:
                    (left_sum_gradients, right_sum_gradients
                     ) = self._compute_sum_gradients_left_and_right_of_split(
                         feature, threshold, node_input_vectors,
                         node_gradients)

                    left_feature_gradients.append(left_sum_gradients)

                    if report_gradients_both_sides_split:
                        right_feature_gradients.append(right_sum_gradients)

                if report_per_feature_result_difference:
                    left_feature_gradients = self._difference_array(
                        left_feature_gradients)

                    if report_gradients_both_sides_split:
                        right_feature_gradients = self._difference_array(
                            right_feature_gradients)

                left_node_gradients.extend(left_feature_gradients)

                if report_gradients_both_sides_split:
                    right_node_gradients.extend(right_feature_gradients)

            if report_per_node_result_difference:
                left_node_gradients = self._difference_array(
                    left_node_gradients)

                if report_gradients_both_sides_split:
                    right_node_gradients = self._difference_array(
                        right_node_gradients)

            result.extend(left_node_gradients)

            if report_gradients_both_sides_split:
                result.extend(right_node_gradients)

            if not report_gradients_both_sides_split:
                if report_node_sum_gradients:
                    result.append(left_sum_gradients + right_sum_gradients)
                else:
                    result.append(right_sum_gradients)

        return result

    def _weight_result(self, weight_vector: Optional[Union[np.ndarray, List]],
                       result: np.ndarray) -> np.ndarray:
        """
        Do element-wise multiplication of the result vector with another vector
        of equal length.

        :param weight_vector:
            Vector used to weight the result.
        :param result:
            Vector of floats which is the result to submit from client to be
            aggregated.

        :returns:
            Weight result vector.
        """
        if weight_vector is None or len(weight_vector) != len(result):
            return np.array(result)

        return np.multiply(weight_vector, result)

    def _translate_result(self, translate_vector: Optional[Union[np.ndarray,
                                                                 List]],
                          result: List) -> np.ndarray:
        """
        Apply a translation to a result vector by element-wise addition with
        another vector of equal length.

        Note that no translation of the result vector takes place if the
        translate vector is not of the same length.
        """
        if translate_vector is None or len(translate_vector) != len(result):
            return np.array(result)

        return np.add(result, translate_vector)

    def _difference_array(self, arr: List) -> List:
        """
        Transform array into another array of the same shape, where each
        element of the array is the difference between the value at the current
        index of the array and the value at the previous index of the array.

        Note that the value at the first index of the array remains the same as
        in the original array.
        """
        if len(arr) < 2:
            return arr

        return np.hstack((arr[0], np.subtract(arr[1:], arr[0:-1]))).tolist()

    def simulate_one_user(
        self, model: GBDTModelType, user_dataset: TabularDataset,
        central_context: FederatedGBDTCentralContextType
    ) -> Tuple[Optional[MappedVectorStatistics], Metrics]:

        assert isinstance(user_dataset,
                          TabularDataset), 'Must use TabularDataset'

        model_train_params = central_context.model_train_params
        model_eval_params = central_context.model_train_params
        algorithm_params = central_context.algorithm_params

        def name_formatting_fn(s: str):
            return MetricName(s, central_context.population)

        metrics = Metrics()

        metrics |= model.evaluate(user_dataset, name_formatting_fn,
                                  model_eval_params)
        statistics = None

        if central_context.population == Population.TRAIN:

            statistics, training_metrics = self._train_one_user(
                model, user_dataset, algorithm_params, model_train_params,
                name_formatting_fn)

            metrics |= training_metrics

        return statistics, metrics

    def _compute_cohort_size(self, algorithm_params, depth):
        """
        Cohort size used for training each level of a tree in a GBDT.

        Default method for adapting training cohort size per layer of trees in
        GBDT is the "power" method, which ensures the SNR is not reduced when
        training deeper layers of the trees in a GBDT.

        Note that fewer results are required to compute the values of leaf
        nodes, as fewer training statistics are aggregated to compute this
        value. Consequently, the cohort size for training leaf nodes is
        reduced compared to the base value.
        """
        if algorithm_params.cohort_size_per_layer_modifier_fn == 'none':
            return algorithm_params.cohort_size

        if algorithm_params.cohort_size_per_layer_modifier_fn == 'linear':
            return int(depth * algorithm_params.cohort_size)

        if algorithm_params.cohort_size_per_layer_modifier_fn == 'power':
            return int(2**(depth - 1) * algorithm_params.cohort_size)

    def get_next_central_contexts(
        self,
        model: GBDTModelType,
        iteration: int,
        algorithm_params: GBDTAlgorithmHyperParamsType,
        model_train_params: GBDTModelHyperParamsType,
        model_eval_params: Optional[GBDTModelHyperParamsType] = None,
    ) -> Tuple[Optional[Tuple[FederatedGBDTCentralContextType, ...]],
               GBDTModelType, Metrics]:

        if iteration == 0:
            self._finish_tree_id = (model.current_tree +
                                    algorithm_params.num_trees)

        # stop condition
        if model.current_tree > self._finish_tree_id:
            return None, model, Metrics()

        # train cohort size
        if model.current_depth == model.max_depth:
            cohort_size = algorithm_params.cohort_size_leaf_nodes
        else:
            cohort_size = self._compute_cohort_size(algorithm_params,
                                                    model.current_depth)

        # only do evaluation on fully trained trees
        val_cohort_size = (algorithm_params.val_cohort_size
                           if model.current_depth == 0 else 0)
        do_evaluation = model.current_depth == 0
        compute_second_order_gradients = any(
            node.is_leaf for node in model.nodes_to_split
        ) if not algorithm_params.compute_second_order_gradients else True

        static_model_train_params = model_train_params.static_clone()
        static_model_eval_params = model_eval_params.static_clone()

        gbdt_questions, _ = choose_questions(model.nodes_to_split,
                                             self._features)
        static_train_algorithm_params = (
            _GBDTInternalAlgorithmHyperParams.from_GBDTAlgorithmHyperParams(
                algorithm_params=algorithm_params,
                cohort_size=cohort_size,
                val_cohort_size=val_cohort_size,
                compute_second_order_gradients=compute_second_order_gradients,
                gbdt_questions=gbdt_questions))

        configs = [
            CentralContext(current_central_iteration=iteration,
                           do_evaluation=do_evaluation,
                           cohort_size=get_param_value(
                               algorithm_params.cohort_size),
                           population=Population.TRAIN,
                           model_train_params=static_model_train_params,
                           model_eval_params=static_model_eval_params,
                           algorithm_params=static_train_algorithm_params,
                           seed=self._get_seed())
        ]

        if do_evaluation and val_cohort_size:
            configs.append(
                CentralContext(
                    current_central_iteration=iteration,
                    do_evaluation=do_evaluation,
                    cohort_size=val_cohort_size,
                    population=Population.VAL,
                    model_train_params=static_model_train_params,
                    model_eval_params=static_model_eval_params,
                    algorithm_params=_GBDTInternalAlgorithmHyperParams.
                    from_GBDTAlgorithmHyperParams(algorithm_params),
                    seed=self._get_seed()))

        return tuple(configs), model, Metrics()

    @staticmethod
    def postprocess_training_statistics(
        statistics: MappedVectorStatistics,
        l2_regularization: float = 1
    ) -> Tuple[int, float, float, float, float, float]:
        """
        Postprocess statistics for one node.

        :param statistics:
            Must have keys: 'questions'; 'first_order_grads_left';
            'first_order_grads_right'; 'second_order_grads_left';
            'second_order_grads_right'.
        :param l2_regularization:
            L2 regularization used to reduce the complexity of the GBDT model
            trained, and avoid the model overfitting to the data.
        :returns:
            Tuple of 6 scalar statistics representing a node in a tree in a
            GBDT: (feature, threshold, gain, value, left_child_value,
            right_child_value)
        """
        assert set(statistics.keys()) == {
            'questions', 'first_order_grads_left', 'first_order_grads_right',
            'second_order_grads_left', 'second_order_grads_right'
        }, ('Invalid statistics object. Must have the following keys:',
            "'questions', 'first_order_grads_left', 'first_order_grads_right',",
            "'second_order_grads_left', 'second_order_grads_right'")

        assert l2_regularization >= 0, ('l2_regularization must be >= 0.',
                                        'Default value = 1.')

        q = statistics['questions']
        g_l = statistics['first_order_grads_left']
        g_r = statistics['first_order_grads_right']
        h_l = statistics['second_order_grads_left']
        h_r = statistics['second_order_grads_right']

        assert isinstance(g_l, np.ndarray) and isinstance(
            g_r, np.ndarray), 'Require left and right first order gradients'
        assert (isinstance(h_l, np.ndarray)
                and isinstance(h_r, np.ndarray)) or (h_l is None
                                                     and h_r is None)

        if h_l is None or len(h_l) == 0:
            h_l = np.zeros(g_l.shape)
        else:
            h_l[h_l < 0] = 0

        if h_r is None or len(h_r) == 0:
            h_r = np.zeros(g_r.shape)
        else:
            h_r[h_r < 0] = 0

        # compute gain for each question
        def compute_gains(g, h):
            return np.divide(g**2,
                             h + l2_regularization,
                             where=(h + l2_regularization) != 0)

        gains = compute_gains(g_l, h_l) + compute_gains(g_r, h_r)

        # find best split for node
        best_split_id = np.random.choice(
            np.argwhere(gains == np.amax(gains)).reshape(-1), 1)[0]
        best_feature, best_threshold = q[best_split_id]
        best_gain = gains[best_split_id]

        # compute value of node, and child nodes, given best split
        def compute_values(g, h):
            return -g / (h + l2_regularization)

        left_child_value = compute_values(g_l[best_split_id],
                                          h_l[best_split_id])
        right_child_value = compute_values(g_r[best_split_id],
                                           h_r[best_split_id])
        node_value = compute_values(g_l[best_split_id] + g_r[best_split_id],
                                    h_l[best_split_id] + h_r[best_split_id])

        return (best_feature, best_threshold, best_gain, node_value,
                left_child_value, right_child_value)

    @staticmethod
    def _compute_num_gradients(questions, report_gradients_both_sides_split,
                               compute_first_order_gradients,
                               compute_second_order_gradients):

        def compute_num_gradients_per_question_order(num_q, order):
            return int(order) * (
                num_q * max(2 * int(report_gradients_both_sides_split), 1) +
                int(not report_gradients_both_sides_split))

        num_questions = sum([len(v) for v in questions.values()])
        num_first_order_gradients = compute_num_gradients_per_question_order(  # pylint: disable=line-too-long
            num_questions, compute_first_order_gradients)
        num_second_order_gradients = compute_num_gradients_per_question_order(  # pylint: disable=line-too-long
            num_questions, compute_second_order_gradients)

        return num_first_order_gradients, num_second_order_gradients

    @staticmethod
    def _decode_training_statistics(
        training_statistics: MappedVectorStatistics,
        algorithm_params: _GBDTInternalAlgorithmHyperParamsType
    ) -> List[MappedVectorStatistics]:
        """
        Decode result vector of aggregated statistics for training a GBDT, to
        reconstruct left and right 1st and 2nd order gradients for each
        question asked for each node currently being split in a tree.

        :returns:
            Return: list of questions asked for each node being split in this
            central iteration; and lists of left and right 2nd order gradients,
            corresponding to the flattened list of questions per node.
        """

        def determine_gradient_endpoints(
                questions,
                algorithm_params: _GBDTInternalAlgorithmHyperParams):
            num_questions_per_feature = [len(v) for v in questions.values()]

            (num_first_order_gradients, num_second_order_gradients
             ) = FederatedGBDT._compute_num_gradients(
                 questions, algorithm_params.report_gradients_both_sides_split,
                 algorithm_params.compute_first_order_gradients,
                 get_param_value(
                     algorithm_params.compute_second_order_gradients))

            return (num_questions_per_feature, num_first_order_gradients,
                    num_second_order_gradients)

        def get_first_second_order_gradients(
                training_statistics,
                algorithm_params: _GBDTInternalAlgorithmHyperParams):
            g, h = None, None
            if (algorithm_params.compute_first_order_gradients
                    and get_param_value(
                        algorithm_params.compute_second_order_gradients)):
                g = training_statistics[:len(training_statistics) // 2]
                h = training_statistics[len(training_statistics) // 2:]
            elif algorithm_params.compute_first_order_gradients:
                g = training_statistics
            elif get_param_value(
                    algorithm_params.compute_second_order_gradients):
                h = training_statistics
            else:
                raise ValueError(
                    """At least one of compute_first_order_gradients and
                compute_second_order_gradients must be True.""")

            return g, h

        def flatten_questions(questions):
            return [(int(feature), float(t))
                    for (feature, thresholds) in questions.items()
                    for t in thresholds]

        def decode_gradients(
                gradients, algorithm_params: _GBDTInternalAlgorithmHyperParams,
                feature_lengths):
            """
            Decode flattened list of gradients into first and second order,
            left and right gradients.
            """
            if gradients is None or len(gradients) == 0:
                return np.array([])

            if algorithm_params.report_gradients_both_sides_split:
                node_left_gradients = gradients[:len(gradients) // 2]
                node_right_gradients = gradients[len(gradients) // 2:]

                # reverse per-node difference
                if algorithm_params.report_per_node_result_difference:
                    node_left_gradients = np.cumsum(node_left_gradients)
                    node_right_gradients = np.cumsum(node_right_gradients)

                # reverse per-feature difference
                if algorithm_params.report_per_feature_result_difference:
                    start_f = 0
                    for feature_length in feature_lengths:
                        node_left_gradients[
                            start_f:start_f + feature_length] = np.cumsum(
                                node_left_gradients[start_f:start_f +
                                                    feature_length])
                        node_right_gradients[
                            start_f:start_f + feature_length] = np.cumsum(
                                node_right_gradients[start_f:start_f +
                                                     feature_length])
                        start_f += feature_length

            else:
                # only left gradients are reported for each question
                # (right gradients are not reported)
                node_left_gradients = gradients[:-1]

                # reverse per-node difference
                if algorithm_params.report_per_node_result_difference:
                    node_left_gradients = np.cumsum(node_left_gradients)

                # reverse per-feature difference
                if algorithm_params.report_per_feature_result_difference:
                    start_f = 0
                    for feature_length in feature_lengths:
                        node_left_gradients[
                            start_f:start_f + feature_length] = np.cumsum(
                                node_left_gradients[start_f:start_f +
                                                    feature_length])
                        start_f += feature_length

                if algorithm_params.report_node_sum_gradients:
                    # right gradient for each question in node is found by
                    # subtracting left gradient from sum of gradients in node.
                    node_sum_gradients = gradients[-1]
                    node_right_gradients = np.subtract(node_sum_gradients,
                                                       node_left_gradients)
                else:
                    # right gradient of final question in node is reported.
                    # sum of gradients in node can be computed, thereby
                    # yielding right gradient for each question in node.
                    node_sum_gradients = np.add(gradients[-1], gradients[-2])
                    node_right_gradients = np.subtract(node_sum_gradients,
                                                       node_left_gradients)

            assert len(node_left_gradients) == len(
                node_right_gradients
            ), 'number of left and right gradients must be equal.'

            if len(node_left_gradients) == 0:
                node_left_gradients, node_right_gradients = np.array(
                    []), np.array([])

            return node_left_gradients, node_right_gradients

        assert isinstance(training_statistics, MappedVectorStatistics)
        assert STATISTIC_INFO_NAME in training_statistics

        training_statistics = training_statistics[STATISTIC_INFO_NAME]

        g, h = get_first_second_order_gradients(training_statistics,
                                                algorithm_params)
        num_nodes = len(algorithm_params.gbdt_questions)

        output = []
        start_g = 0
        start_h = 0

        for node_id in range(num_nodes):
            gbdt_questions = algorithm_params.gbdt_questions[node_id]['splits']
            (num_questions_per_feature, num_first_order_gradients,
             num_second_order_gradients) = determine_gradient_endpoints(
                 gbdt_questions, algorithm_params)

            if g is not None:
                gradients = g[start_g:start_g + num_first_order_gradients]
                start_g += num_first_order_gradients

                g_l, g_r = decode_gradients(gradients, algorithm_params,
                                            num_questions_per_feature)
            else:
                g_l, g_r = np.array([]), np.array([])

            if h is not None:
                gradients = h[start_h:start_h + num_second_order_gradients]
                start_h += num_second_order_gradients

                h_l, h_r = decode_gradients(gradients, algorithm_params,
                                            num_questions_per_feature)
            else:
                h_l, h_r = np.array([]), np.array([])

            statistics = {
                'questions': flatten_questions(gbdt_questions),
                'first_order_grads_left': g_l,
                'first_order_grads_right': g_r,
                'second_order_grads_left': h_l,
                'second_order_grads_right': h_r
            }
            output.append(MappedVectorStatistics(statistics))

        return output

    def _process_metrics(self, metrics: Metrics, population: Population):
        """
        Compute average accuracy and average MSE on population metrics and add
        to metrics.
        """
        new_metrics = Metrics()

        metrics_format_fn = lambda n: TrainMetricName(
            n, population, after_training=False)
        sum_accurate_predictions_name = metrics_format_fn(
            'SumAccuratePredictions')
        sum_abs_prediction_error_name = metrics_format_fn(
            'SumAbsolutePredictionError')
        num_training_records_name = metrics_format_fn('NumberTrainingRecords')

        if (sum_accurate_predictions_name in metrics
                and sum_abs_prediction_error_name in metrics
                and num_training_records_name in metrics
                and metrics_format_fn('Accuracy') not in metrics
                and metrics_format_fn('MSE') not in metrics):
            new_metrics[metrics_format_fn(
                'Accuracy')] = Weighted.from_unweighted(
                    get_overall_value(metrics[sum_accurate_predictions_name]) /
                    get_overall_value(metrics[num_training_records_name]))
            new_metrics[metrics_format_fn('MSE')] = Weighted.from_unweighted(
                get_overall_value(metrics[sum_abs_prediction_error_name]) /
                get_overall_value(metrics[num_training_records_name]))

        return new_metrics

    def process_aggregated_statistics(
            self, central_context: FederatedGBDTCentralContextType,
            aggregate_metrics: Metrics, model: GBDTModelType,
            statistics: MappedVectorStatistics
    ) -> Tuple[GBDTModelType, Metrics]:
        """
        The aggregated statistics are the aggregated first and, optionally,
        second order gradients for each of the questions asked of nodes to be
        split. For each question, a gain and a value are computed from the
        gradients for that question. The question with the highest gain for
        each node to be split is selected as the split question for that node,
        if the node is a branch node, and will be included in the tree in the
        GBDT for that node. The value will be assigned as a leaf weight to the
        node if the node is a leaf node.

        This function also calls for the model to apply the update, by adding
        branch or leaf nodes with the optimal splits or values as appropriate.
        """
        intermediate_statistics_all_nodes = (self._decode_training_statistics(
            statistics, central_context.algorithm_params))

        model_update_statistics: MappedVectorStatistics = MappedVectorStatistics(
        )
        for node_id, intermediate_statistics_single_node in enumerate(
                intermediate_statistics_all_nodes):
            model_update_statistics[
                f'node_{node_id}'] = self.postprocess_training_statistics(
                    intermediate_statistics_single_node,
                    central_context.algorithm_params.l2_regularization)

        model, metrics = model.apply_model_update(model_update_statistics)

        metrics |= self._process_metrics(aggregate_metrics,
                                         central_context.population)

        return model, metrics
