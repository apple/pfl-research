# Copyright © 2023-2024 Apple Inc.

import json
import os
import typing
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Tuple, TypeVar

import numpy as np
import xgboost as xgb

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import ModelHyperParams
from pfl.internal.ops.selector import get_default_framework_module, set_framework_module
from pfl.internal.tree import GBDTClassifier, GBDTRegressor, Node
from pfl.internal.tree.gbdt import GBDT
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.model.base import EvaluatableModel
from pfl.stats import MappedVectorStatistics

GBDTKind = TypeVar('GBDTKind', bound=GBDT)
GBDTModelType = TypeVar('GBDTModelType', bound='GBDTModel')


@dataclass(frozen=True)
class NodeRecord:
    """
    Node of a GBDT.

    :param parent:
        Parent node. None for root.
    :param decision_path:
        List of triples [feature, threshold, is_left] defining the tests
        along the path to this node and the results of these tests.
        ``is_left`` is True if and only if the specific feature is
        less than or equal to the threshold, so the decision path
        goes left.
    :param is_left:
        Whether this node hangs to the left of its parent. The value in
        the root node is True by default but it does not matter.
    :param value:
        Value of this node as determined by gradients aggregated for
        this node. For the root, it's None.
    :param is_leaf:
        True if the node is a leaf, False otherwise.
    """
    parent: Optional[Node]
    decision_path: List[List]
    is_left: bool
    value: Optional[float]
    is_leaf: bool


@dataclass(frozen=True)
class GBDTModelHyperParams(ModelHyperParams):
    pass


GBDTModelHyperParamsType = TypeVar('GBDTModelHyperParamsType',
                                   bound='GBDTModelHyperParams')


@dataclass(frozen=True)
class GBDTClassificationModelHyperParams(ModelHyperParams):
    evaluation_threshold: float

    def __post__init__(self):
        super().__post_init__()
        assert (self.evaluation_threshold >= 0 and self.evaluation_threshold
                <= 1), ('Evaluation threshold for binary classification'
                        'must be in range [0,1]')


class GBDTModel(EvaluatableModel, Generic[GBDTKind]):
    """
    Gradient Boosted Decision Tree (GBDT) model base class.

    :param num_features:
        Number of features of data used to train GBDT.
    :param max_depth:
        Maximum depth of any tree trained in a GBDT.
    :param learning_rate:
        Learning rate used during boosting. Used to control by how much each
        tree should try to reduce error in predictions to zero.
    :param minimum_split_loss:
        Minimum reduction in loss achieved by splitting a node into two child
        nodes required before a node is split in a GBDT.
    :param base_prediction:
        Default prediction of a GBDT on datapoints before any trees are added
        to the GBDT.
    :param alpha:
        Optionally used when computing value of a node. If None, the value of a
        node is computed from statistics aggregated for that node. If real-
        valued and greater than 0, the value of a node is the weighted sum of
        statistics aggregated for the current node and for the parent node:
        ``a*parent_value + (1-a)*value``, where ``a = alpha``.
    """
    set_framework_module(get_default_framework_module())

    def __init__(self,
                 num_features: int,
                 max_depth: int = 3,
                 learning_rate: float = 0.9,
                 minimum_split_loss: float = 0,
                 base_prediction: float = 0,
                 alpha: Optional[float] = None):

        super().__init__()

        self._num_features = num_features
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._minimum_split_loss = minimum_split_loss
        self._base_prediction = base_prediction

        self._gbdt = self._initialize_gbdt()
        self._nodes_to_split = [self._root_node()]

        self._alpha = alpha
        if self._alpha:
            assert (self._alpha >= 0.0) and (self._alpha <= 1.0), (
                'alpha determining fraction of leaf value apportioned to ',
                'parent value must be in range [0.0, 1.0]')

    @property
    def current_depth(self) -> int:
        """
        Current depth of tree being trained, starting at 0, meaning no nodes
        have yet been added to the tree.
        """
        if len(self._gbdt.trees) == 0 or (
                len(self._gbdt.trees) > 0
                and self._gbdt.trees[-1].training_complete()):
            return 0
        else:
            return self._gbdt.trees[-1].max_depth()

    @property
    def current_tree(self) -> int:
        """
        Current tree being trained for GBDT. Starts at 0.
        """
        if self.current_depth > 0:
            return len(self._gbdt.trees) - 1
        else:
            return len(self._gbdt.trees)

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def nodes_to_split(self) -> List[NodeRecord]:
        return self._nodes_to_split

    @abstractmethod
    def _initialize_gbdt(self) -> GBDTKind:
        """
        Return empty GBDT object, for classification or regression.
        """

    @abstractmethod
    def compute_first_order_gradient(self, target: float, prediction: float):
        """
        Compute first order gradient.
        """

    @abstractmethod
    def compute_second_order_gradient(self, target: float, prediction: float):
        """
        Compute second order gradient.
        """

    def __str__(self):
        return ''.join([
            f'GBDT Model:\nNum features: {self._num_features}\n',
            f'Max tree depth: {self._max_depth}\n',
            f'Learning rate: {self._learning_rate}\n',
            f'Minimum split loss: {self._minimum_split_loss}\n',
            f'{self._gbdt}'
        ])

    def _root_node(self) -> NodeRecord:
        """
        Make a ``NodeRecord`` for a root node of a tree in a GBDT.
        """
        decision_path: List = []
        parent, is_left, value = None, True, None
        is_leaf = self._max_depth == 1
        return NodeRecord(parent, decision_path, is_left, value, is_leaf)

    def save(self, dir_path: str) -> None:
        """
        Save a GBDT model to disk.

        :param dir_path:
            Path to which to save GBDT model will be saved as "gbdt.json"
        """
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        with open(os.path.join(dir_path, 'gbdt.json'), 'w') as f:
            json.dump(self._gbdt.to_serialized_xgboost(), f)

    def load(self, path: str) -> None:  # pylint: disable=arguments-renamed
        """
        Load a GBDT model from disk.

        The model can be loaded from a saved XGBoost model or from a json
        serialisation of a GBDT model, which uses the XGBoost serialisation
        format:
        https://xgboost.readthedocs.io/en/latest/python/python_api.html?highlight=dump_model#xgboost.Booster.dump_model

        An XGBoost model must has a ".model" extension, e.g. "xgboost.model".
        A json serialisation of the model must have a ".json" extension, e.g.
        "gbdt.json".

        Note that the following parameters are not saved to disk along with a
        GBDT model: l2 regularization; number of features; minimum split loss;
        maximum number of trees; maximum depth of any tree.

        Further training of the GBDT model loaded using this function will
        result in new trees being trained and added to the GBDT model. Further
        training will not adapt the existing trees in the GBDT loaded using
        this function.

        :param path:
            Path to a XGBoost model or a json serialisation of a GBDT model.
            Extension of file must be either .model or .json.
        """
        _, file_extension = os.path.splitext(path)
        if file_extension == '.model':  # load from xgboost model
            xgboost_model = xgb.Booster()
            xgboost_model.load_model(path)
            model = xgboost_model.get_dump(dump_format='json')
            model = [json.loads(tree) for tree in model]
        elif file_extension == '.json':  # load from json model serialisation
            with open(path) as f:
                model = json.load(f)
        else:
            raise ValueError(
                f'File type {path} not supported for loading. Only XGBoost models '
                'with ".model" extension or a json serialisation of the model '
                'with ".json" extension are supported.')

        self._gbdt = self._initialize_gbdt()

        for tree in model:
            self._gbdt.add_tree(Node.from_serialized_xgboost(tree))

        self._nodes_to_split = [self._root_node()]

    def _combine_values(self,
                        value: float,
                        parent_value: Optional[float] = None) -> float:
        """
        Incorporate parent with current observation to determine value of node.

        The parent value is computed from statistics aggregated for the parent
        node of the node whose value is being determined. The current value is
        computed from statistics aggregated for the current node. If
        ``self._alpha`` is not ``None``, the parent and current values are
        combined using ``a * parent_value + (1 - a) * value``, where
        ``a = self._alpha``. Else, the value of the node is ``value``.

        :param value:
            Value of node as determined by gradients aggregated for this node.
        :param parent_value:
            Value of node as determined by gradients aggregated for parent node
            of this node.

        :returns:
            Value of node.
        """
        if parent_value is not None and self._alpha is not None:
            return self._alpha * parent_value + (1 - self._alpha) * value
        return value

    def _compute_node_value(self,
                            value: float,
                            parent_value: Optional[float] = None):
        """
        Compute value of node by:
        1. combining current value with the parent value for node.
        2. applying learning rate to value.

        :param value:
            Value of node as determined by gradients aggregated for this node.
        :param parent_value:
            Value of node as determined by gradients aggregated for parent node
            of this node.

        :returns:
            Adjusted value of node.
        """
        value = self._combine_values(value, parent_value)
        return value * self._learning_rate

    def apply_model_update(
            self, statistics: MappedVectorStatistics
    ) -> Tuple[GBDTModelType, Metrics]:
        """
        Update GBDT model using best splits determined from first and second
        order gradients computed on training data.

        The statistics define the node to be added to a tree in a GBDT.

        All nodes on a single level of a tree in a GBDT are updated in one
        iteration of training, corresponding to one call of this function.

        The parent nodes of new nodes to be added to the tree in a GBDT are
        defined in ``self._nodes_to_split``. The best splits for these nodes are
        computed by the ``FederatedGBDT`` algorithm, determining whether another
        branch node or leaf node should be added as a child node to the parent
        node. Any child branch nodes are added to ``self._nodes_to_split`` so
        that they can be added to the tree in the next iteration of training.

        A node can be set as a leaf node if the maximum depth of a tree has
        been reached or if the minimum split loss was not attained by the best
        split for that node.

        :param statistics:
            ``MappedVectorStatistics`` holding (feature, threshold, gain, value,
            left_child_value, right_child_value) statistics for each node to
            be added to tree. These statistics have been computed from
            gradients aggregated from user devices.
        """
        current_depth = self.current_depth
        new_nodes_to_split = []

        for i, node in enumerate(self._nodes_to_split):

            (feature, threshold, gain, value, left_child_value,
             right_child_value) = statistics[f'node_{i}']

            is_leaf = current_depth >= (
                self._max_depth -
                1) or gain <= self._minimum_split_loss or node.is_leaf

            if is_leaf:
                if node.parent:
                    node.parent.add_leaf_node(
                        node.is_left,
                        self._compute_node_value(value, node.value))
                else:  # root node is a leaf node
                    self._gbdt.add_tree(Node(value=value))

            else:
                if node.parent:
                    parent = node.parent.add_branch_node(
                        node.is_left, feature, threshold)
                else:  # root node is a branch node
                    parent = Node(feature=feature, threshold=threshold)
                    self._gbdt.add_tree(parent)

                child_is_leaf = current_depth >= (self._max_depth - 2)

                for (is_left, child_value) in zip(
                    [True, False], [left_child_value, right_child_value]):
                    child_decision_path = [
                        *node.decision_path.copy(),
                        [feature, threshold, is_left]
                    ]
                    child_node = NodeRecord(parent, child_decision_path,
                                            is_left, child_value,
                                            child_is_leaf)

                    new_nodes_to_split.append(child_node)

        self._nodes_to_split = new_nodes_to_split

        if self.current_depth > (self._max_depth - 1):
            assert len(self._nodes_to_split) == 0, (
                'The final level of the tree has been reached, but there are ',
                'still nodes to be split which should have been added to the ',
                'latest tree in the GBDT as leaf nodes.')

        if len(self._nodes_to_split) == 0:
            self._nodes_to_split = [self._root_node()]

        return typing.cast(GBDTModelType, self), Metrics()

    def predict(self, X: np.ndarray):
        """
        Make predictions for each data sample in X using GBDT.
        """
        return self._gbdt.predict(X)

    def evaluate(
            self,
            dataset: AbstractDatasetType,
            name_formatting_fn=lambda n: StringMetricName(n),
            eval_params: Optional[GBDTModelHyperParams] = None) -> Metrics:
        """
        Evaluate model performance using a dataset.

        The prediction of a datapoint used for evaluation is computed using:

        1. all the trees in a GBDT, when the most recently added tree of the
           GBDT is fully trained.
        2. All trees in the GBDT except the most recently added tree, when this
           most recently added tree is not fully trained.

        :param dataset:
            The dataset on which evaluation should take place. Assumes that
            ``dataset.raw_data`` is a list of datapoints, where each datapoint
            is represented by a tuple ``(x, y)``, where ``x`` is a vector of
            input features, and ``y`` is a scalar target value.
        :param name_formatting_fn:
            A function that produces a ``TrainMetricName`` object from a simple
            string for metrics computed on a dataset.

        :returns:
            A `Metrics` object with performance metrics evaluated on the
            dataset.
        """
        input_vectors = dataset.raw_data[0]
        targets = np.ravel(dataset.raw_data[1])

        metrics = Metrics()
        for (metric_name,
             metric_value) in self._gbdt.evaluate(input_vectors,
                                                  targets).items():
            # User-average metric
            metrics[name_formatting_fn(
                f'per-user {metric_name}')] = Weighted.from_unweighted(
                    metric_value)
            # Datapoint-average metric
            metrics[name_formatting_fn(f'overall {metric_name}')] = Weighted(
                metric_value * len(targets), len(targets))

        return metrics


class GBDTModelClassifier(GBDTModel[GBDTClassifier]):
    """
    Federated GBDT model for binary classification.

    See GBDTModel for description of parameters.
    """

    def _initialize_gbdt(self) -> 'GBDTClassifier':
        """
        Return empty GBDT object for classification.
        """
        return GBDTClassifier(self._base_prediction)

    def compute_first_order_gradient(self, target, prediction):
        """
        Compute first order gradient of log loss (cross-entropy loss for binary
        classification).
        """
        return prediction - target

    def compute_second_order_gradient(self, target, prediction):
        """
        Second order gradient of log loss for binary classification.
        """
        return prediction * (1 - prediction)

    def predict_classes(self, X: np.ndarray):
        """
        Round predictions to closest of binary values, {0,1}.
        """
        return self._gbdt.predict_classes(X)


class GBDTModelRegressor(GBDTModel[GBDTRegressor]):
    """
    Federated GBDT model for regression.

    See GBDTModel for description of parameters.
    """

    def _initialize_gbdt(self) -> 'GBDTRegressor':
        """
        Return empty GBDT object, for classification or regression.
        """
        return GBDTRegressor(self._base_prediction)

    def compute_first_order_gradient(self, target, prediction):
        """
        Compute first order gradient for squared error loss function used for
        regression.
        """
        return prediction - target

    def compute_second_order_gradient(self, target, prediction):
        """
        Compute second order gradient for squared error loss function for
        regression.
        """
        return 1
