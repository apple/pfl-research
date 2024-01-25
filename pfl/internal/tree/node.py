# Copyright © 2023-2024 Apple Inc.

from typing import Dict, List, Optional, Tuple

import numpy as np


class Node:
    """
    Represents a branch node or a leaf node in a binary decision tree.

    A branch node has a left child node and right child node, and is defined
    by an inequality: feature <= threshold. A datapoint follows the path to the
    left child node if the inequality is satisfied for this datapoint. Else,
    a datapoint follows the path to the right child node.

    A leaf node has no left or right child node. It is defined by a value.
    """

    def __init__(self,
                 feature: Optional[int] = None,
                 threshold: Optional[float] = None,
                 value: Optional[float] = None,
                 left_child: 'Optional[Node]' = None,
                 right_child: 'Optional[Node]' = None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left_child = left_child
        self.right_child = right_child

    def is_leaf(self) -> bool:
        """
        Check if node is a leaf node.

        A leaf node has a value and does not have a left or a right child node.

        If a node does not have a left or a right child node, but does not have
        a value, it is just an incomplete branch node, with no child nodes
        attached yet.

        :return:
            True if node is a leaf node, else False.
        """
        return not (self.left_child or self.right_child) and (self.value
                                                              is not None)

    def num_nodes(self) -> int:
        """
        Return number of nodes in tree, starting with current node.

        The tree does not have to be fully trained to call this function.

        Either value must be set, or feature and threshold must be set for a
        node to be counted.
        """
        # undefined node
        if self.value is None and (self.feature is None
                                   or self.threshold is None):
            return 0

        num_nodes = 1  # count this node
        if self.left_child:
            num_nodes += self.left_child.num_nodes()
        if self.right_child:
            num_nodes += self.right_child.num_nodes()
        return num_nodes

    def max_depth(self) -> int:
        """
        Return maximum depth of tree: the number of nodes along the longest
        path in the tree.

        The tree does not have to be fully trained to call this function.

        Either value must be set, or feature and threshold must be set for a
        node to be counted.

        :return:
            Maximum depth of tree from current node.
        """
        # undefined node
        if self.value is None and (self.feature is None
                                   or self.threshold is None):
            return 0

        left_depth = self.left_child.max_depth() if self.left_child else 0
        right_depth = self.right_child.max_depth() if self.right_child else 0
        return max(left_depth, right_depth) + 1

    def __str__(self) -> str:
        """
        Example:
        '
        Tree
        Number of nodes: 7
        Max depth: 3
        Finished training: True
        Structure:
        Feature 3 <= 5.5
        ---> Feature 2 <= 0.0
        ---> ---> Leaf value = 0.1
        ---> ---> Leaf value = -0.5
        ---> Feature 1 <= -0.5
        ---> ---> Leaf value = 0.5
        ---> ---> Leaf value = 0.2
        '
        """
        text = '\n'.join([
            '\nTree', f'Number of nodes: {self.num_nodes()}',
            f'Max depth: {self.max_depth()}',
            f'Finished training: {self.training_complete()}', 'Structure:'
        ])

        return text + self._node_str(0) + '\n'

    def _node_str(self, depth: int) -> str:
        if self.is_leaf():
            return '\n' + '---> ' * depth + f'Leaf value = {self.value}'

        text = '\n' + '---> ' * depth + f'Feature {self.feature} <= {self.threshold}'

        if self.left_child:
            text += self.left_child._node_str(depth + 1)  # pylint: disable=protected-access
        if self.right_child:
            text += self.right_child._node_str(depth + 1)  # pylint: disable=protected-access

        return text

    def _add_node(self, node: 'Node', is_left: bool):
        assert not self.is_leaf(
        ) and self.feature is not None and self.threshold is not None, (
            'Cannot add child node to leaf node '
            'or node without feature and threshold set')
        if is_left:
            self.left_child = node
        else:
            self.right_child = node

    def add_leaf_node(self, is_left: bool, value: float):
        """
        Add a leaf child node to this node.

        :param is_left:
            If True, leaf node is left child of parent node.
            Else, leaf node is right child of parent node.
        :param value:
            Value of leaf node, used for predictions.
        """
        self._add_node(Node(value=value), is_left)

    def add_branch_node(self, is_left: bool, feature: int,
                        threshold: float) -> 'Node':
        """
        Add a branch child node to this node.

        :param is_left:
            If True, branch node is left child of parent node.
            Else, branch node is right child of parent node.
        :param feature:
            Feature used for inequality condition at branch node.
        :param threshold:
            Threshold used for inequality condition at branch node.

        :return:
            Branch child node just added.
        """
        branch_node = Node(feature=feature, threshold=threshold)
        self._add_node(branch_node, is_left)
        return branch_node

    def training_complete(self) -> bool:
        """
        Returns whether tree, including this node and all child nodes, is
        completely trained or not.

        Tree is completely trained if it is a full binary tree: every node
        other than the leaf nodes have 2 child nodes.
        """
        # branch node
        if self.left_child and self.right_child:
            return (self.feature is not None) and (
                self.threshold
                is not None) and self.left_child.training_complete(
                ) and self.right_child.training_complete()

        # either leaf node or incomplete node
        return self.is_leaf()

    def get_leaf_values(self) -> List[float]:
        """
        Return list of values of all leaves in tree.

        Tree can be fully or partially trained when this method is called.
        """
        if self.is_leaf():
            assert self.value is not None
            return [self.value]

        leaf_values = []
        if self.left_child:
            leaf_values.extend(self.left_child.get_leaf_values())
        if self.right_child:
            leaf_values.extend(self.right_child.get_leaf_values())
        return leaf_values

    def _predict(self, datapoint: np.ndarray) -> float:
        """
        Make prediction for a single datapoint using tree.
        Only call when tree is fully trained: self.training_complete() is True.

        :param datapoint:
            A single datapoint, array-like of size (d, 1), (d,) or (1, d).
        """
        if self.is_leaf():
            assert self.value is not None
            return self.value

        if datapoint[self.feature] <= self.threshold:
            assert self.left_child is not None
            return self.left_child._predict(datapoint)  # pylint: disable=protected-access
        else:
            assert self.right_child is not None
            return self.right_child._predict(datapoint)  # pylint: disable=protected-access

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make prediction for each datapoint in X using tree.

        Each datapoint is passed through the tree, following the decision path
        arising from the inequality condition at each branch node, until each
        datapoint arrives at a leaf node, whose value is the prediction for
        this datapoint.

        If the tree is empty, return the base prediction for all datapoints.

        The tree must be fully trained to return a prediction.

        :param X:
            (N x d) array of datapoints, where N is the number of datapoints
            and d is the number of features per datapoint.

        :return:
            Prediction from tree for each datapoint in X.
        """
        assert self.training_complete(), ('Tree must be fully trained before'
                                          'calling predict()')

        predictions = []
        for datapoint in X:
            predictions.append(self._predict(datapoint))

        return np.array(predictions)

    def to_serialized_xgboost(self) -> Dict:
        """
        Emulate xgboost's dump_model() function.

        :return:
            Nested dictionary describing the nodes in a tree.
        """
        assert self.training_complete(), (
            'Tree must be fully trained before '
            'being converted to XGBoost dump_model() format.')

        return self._add_xgboost_node(0, 0)  # pylint: disable=protected-access

    def _add_xgboost_node(self, node_id: int, depth: int) -> Dict:
        """
        Assumes tree is fully trained.
        """
        if self.is_leaf():
            return {'nodeid': node_id, 'leaf': self.value}

        xgboost_node: Dict = {
            'nodeid': node_id,
            'depth': depth,
            'split': self.feature,
            'split_condition': self.threshold
        }

        left_child_id, right_child_id = 2 * node_id + 1, 2 * node_id + 2

        xgboost_node['yes'] = left_child_id
        xgboost_node['no'] = right_child_id
        assert self.left_child is not None
        assert self.right_child is not None
        xgboost_node['children'] = [
            self.left_child._add_xgboost_node(left_child_id, depth + 1),  # pylint: disable=protected-access
            self.right_child._add_xgboost_node(right_child_id, depth + 1)  # pylint: disable=protected-access
        ]

        return xgboost_node

    @staticmethod
    def from_serialized_xgboost(tree: Dict) -> 'Node':
        """
        Extra tree from serialized xgboost format.
        Use depth-first search to process all nodes in tree.

        :param tree:
            Dictionary representing tree, in serialized
            XGBoost format.
        :return:
            Return root node of tree.
        """
        current_nodes: List = [(tree, None, True)]
        while len(current_nodes) > 0:
            next_nodes = []
            for (node_dict, parent, is_left) in current_nodes:
                (feature, threshold, value) = process_xgboost_node(node_dict)
                # leaf node
                if value:
                    if parent is None:
                        root = Node(value=value)
                    else:
                        parent.add_leaf_node(is_left, value)
                # branch node
                else:
                    if parent is None:
                        root = node = Node(feature=feature,
                                           threshold=threshold)
                    else:
                        node = parent.add_branch_node(is_left, feature,
                                                      threshold)

                    for child in node_dict['children']:
                        if child['nodeid'] == node_dict['yes']:
                            is_left = True
                        elif child['nodeid'] == node_dict['no']:
                            is_left = False
                        else:
                            raise ValueError(
                                'Issue parsing model to load.',
                                'Child node has not been assigned to '
                                'left or right of branch node.')
                        next_nodes.append((child, node, is_left))

            current_nodes = next_nodes
        return root


def process_xgboost_node(
        node: Dict) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    """
    Process a node in serialized xgboost format.

    :param node
        Serialized XGBoost dictionary format for node.

    :return:
        Tuple of (feature, threshold, value) describing node.
        Note that if node is a branch node, Value will be None.
        Else if node is a leaf node, Feature and Threshold will
        be None.
    """
    # leaf node
    if 'leaf' in node:
        return None, None, node['leaf']

    # branch node
    feature = int(node['split'].lstrip('f')) if isinstance(
        node['split'], str) else node['split']
    threshold = node['split_condition']
    return feature, threshold, None
