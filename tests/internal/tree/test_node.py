# Copyright © 2023-2024 Apple Inc.

import numpy as np
import pytest
from pytest_lazy_fixtures import lf

from pfl.internal.tree.node import Node


@pytest.fixture()
def serialized_xgboost_tree_fully_trained_3_layers():
    serialized_xgboost = {
        "nodeid":
        0,
        "depth":
        0,
        "split":
        0,
        "split_condition":
        0,
        "yes":
        1,
        "no":
        2,
        "children": [{
            "nodeid":
            1,
            "depth":
            1,
            "split":
            1,
            "split_condition":
            1,
            "yes":
            3,
            "no":
            4,
            "children": [{
                "nodeid": 3,
                "leaf": 0.3
            }, {
                "nodeid": 4,
                "leaf": 0.4
            }]
        }, {
            "nodeid":
            2,
            "depth":
            1,
            "split":
            2,
            "split_condition":
            2,
            "yes":
            5,
            "no":
            6,
            "children": [{
                "nodeid": 5,
                "leaf": -0.3
            }, {
                "nodeid": 6,
                "leaf": -0.4
            }]
        }]
    }
    return serialized_xgboost


class TestNode:

    @pytest.mark.parametrize('is_left', [True, False])
    def test_add_leaf_node_success(self, branch_node, is_left,
                                   check_equal_nodes):
        value = 0.3
        branch_node.add_leaf_node(is_left, value)
        expected = Node(value=value)
        if is_left:
            check_equal_nodes(branch_node.left_child, expected)
            assert not branch_node.right_child
        else:
            check_equal_nodes(branch_node.right_child, expected)
            assert not branch_node.left_child

    @pytest.mark.xfail(raises=AssertionError, strict=True)
    @pytest.mark.parametrize('node', [lf('undefined_node'), lf('leaf_node')])
    def test_add_leaf_node_fail(self, node):
        node.add_leaf_node(is_left=True, value=0.3)

    @pytest.mark.parametrize('is_left', [True, False])
    def test_add_branch_node_success(self, branch_node, is_left,
                                     check_equal_nodes):
        feature, threshold = 0, 0.5
        branch_node.add_branch_node(is_left, feature, threshold)
        expected = Node(feature=feature, threshold=threshold)
        if is_left:
            check_equal_nodes(branch_node.left_child, expected)
            assert not branch_node.right_child
        else:
            check_equal_nodes(branch_node.right_child, expected)
            assert not branch_node.left_child

    @pytest.mark.xfail(raises=AssertionError, strict=True)
    @pytest.mark.parametrize('node', [lf('undefined_node'), lf('leaf_node')])
    def test_add_branch_node_fail(self, node):
        node.add_branch_node(True, 0, 0.5)

    @pytest.mark.xfail(raises=AssertionError, strict=True)
    @pytest.mark.parametrize(
        'node', [lf('tree_incomplete_2_layers'),
                 lf('undefined_node')])
    def test_predict_fail(self, node, gbdt_datapoints):
        node.predict(gbdt_datapoints)

    @pytest.mark.parametrize(
        'node, expected',
        [(lf('tree_fully_trained_3_layers'), np.array([0.3, 0.4, -0.3, -0.4])),
         (lf('leaf_node'), np.array([0.3, 0.3, 0.3, 0.3]))])
    def test_predict_success(self, node, gbdt_datapoints, expected):
        predictions = node.predict(gbdt_datapoints)
        np.testing.assert_array_equal(predictions, expected)

    @pytest.mark.parametrize(
        'node, expected',
        [(lf('tree_fully_trained_3_layers'), [0.3, 0.4, -0.3, -0.4]),
         (lf('leaf_node'), [0.3]), (lf('tree_incomplete_2_layers'), []),
         (lf('branch_node'), []), (lf('undefined_node'), [])])
    def test_get_leaf_values(self, node, expected):
        assert node.get_leaf_values() == expected

    @pytest.mark.parametrize('node, expected',
                             [(lf('tree_fully_trained_3_layers'), 7),
                              (lf('leaf_node'), 1),
                              (lf('tree_incomplete_2_layers'), 3),
                              (lf('branch_node'), 1),
                              (lf('undefined_node'), 0)])
    def test_num_nodes(self, node, expected):
        assert node.num_nodes() == expected

    @pytest.mark.parametrize('node, expected',
                             [(lf('tree_fully_trained_3_layers'), 3),
                              (lf('leaf_node'), 1),
                              (lf('tree_incomplete_2_layers'), 2),
                              (lf('branch_node'), 1),
                              (lf('undefined_node'), 0)])
    def test_max_depth(self, node, expected):
        assert node.max_depth() == expected

    @pytest.mark.parametrize('node, expected',
                             [(lf('tree_fully_trained_3_layers'), True),
                              (lf('leaf_node'), True),
                              (lf('tree_incomplete_2_layers'), False),
                              (lf('branch_node'), False),
                              (lf('undefined_node'), False)])
    def test_training_complete(self, node, expected):
        assert node.training_complete() == expected

    @pytest.mark.xfail(raises=AssertionError, strict=True)
    @pytest.mark.parametrize(
        'node', [lf('tree_incomplete_2_layers'),
                 lf('undefined_node')])
    def test_to_serialized_xgboost_fail(self, node):
        node.to_serialized_xgboost()

    def test_to_serialized_xgboost_success(
            self, tree_fully_trained_3_layers,
            serialized_xgboost_tree_fully_trained_3_layers):
        dump_model = tree_fully_trained_3_layers.to_serialized_xgboost()
        assert dump_model == serialized_xgboost_tree_fully_trained_3_layers

    def test_from_serialized_xgboost(
            self, serialized_xgboost_tree_fully_trained_3_layers,
            tree_fully_trained_3_layers):
        tree = Node.from_serialized_xgboost(
            serialized_xgboost_tree_fully_trained_3_layers)

        assert tree.num_nodes() == tree_fully_trained_3_layers.num_nodes()
        assert tree.max_depth() == tree_fully_trained_3_layers.max_depth()
        assert str(tree) == str(tree_fully_trained_3_layers)
