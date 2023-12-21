# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

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
    @pytest.mark.parametrize(
        'node', [lazy_fixture('undefined_node'),
                 lazy_fixture('leaf_node')])
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
    @pytest.mark.parametrize(
        'node', [lazy_fixture('undefined_node'),
                 lazy_fixture('leaf_node')])
    def test_add_branch_node_fail(self, node):
        node.add_branch_node(True, 0, 0.5)

    @pytest.mark.xfail(raises=AssertionError, strict=True)
    @pytest.mark.parametrize('node', [
        lazy_fixture('tree_incomplete_2_layers'),
        lazy_fixture('undefined_node')
    ])
    def test_predict_fail(self, node, gbdt_datapoints):
        node.predict(gbdt_datapoints)

    @pytest.mark.parametrize(
        'node, expected',
        [(lazy_fixture('tree_fully_trained_3_layers'),
          np.array([0.3, 0.4, -0.3, -0.4])),
         (lazy_fixture('leaf_node'), np.array([0.3, 0.3, 0.3, 0.3]))])
    def test_predict_success(self, node, gbdt_datapoints, expected):
        predictions = node.predict(gbdt_datapoints)
        np.testing.assert_array_equal(predictions, expected)

    @pytest.mark.parametrize(
        'node, expected',
        [(lazy_fixture('tree_fully_trained_3_layers'), [0.3, 0.4, -0.3, -0.4]),
         (lazy_fixture('leaf_node'), [0.3]),
         (lazy_fixture('tree_incomplete_2_layers'), []),
         (lazy_fixture('branch_node'), []),
         (lazy_fixture('undefined_node'), [])])
    def test_get_leaf_values(self, node, expected):
        assert node.get_leaf_values() == expected

    @pytest.mark.parametrize('node, expected',
                             [(lazy_fixture('tree_fully_trained_3_layers'), 7),
                              (lazy_fixture('leaf_node'), 1),
                              (lazy_fixture('tree_incomplete_2_layers'), 3),
                              (lazy_fixture('branch_node'), 1),
                              (lazy_fixture('undefined_node'), 0)])
    def test_num_nodes(self, node, expected):
        assert node.num_nodes() == expected

    @pytest.mark.parametrize('node, expected',
                             [(lazy_fixture('tree_fully_trained_3_layers'), 3),
                              (lazy_fixture('leaf_node'), 1),
                              (lazy_fixture('tree_incomplete_2_layers'), 2),
                              (lazy_fixture('branch_node'), 1),
                              (lazy_fixture('undefined_node'), 0)])
    def test_max_depth(self, node, expected):
        assert node.max_depth() == expected

    @pytest.mark.parametrize(
        'node, expected', [(lazy_fixture('tree_fully_trained_3_layers'), True),
                           (lazy_fixture('leaf_node'), True),
                           (lazy_fixture('tree_incomplete_2_layers'), False),
                           (lazy_fixture('branch_node'), False),
                           (lazy_fixture('undefined_node'), False)])
    def test_training_complete(self, node, expected):
        assert node.training_complete() == expected

    @pytest.mark.xfail(raises=AssertionError, strict=True)
    @pytest.mark.parametrize('node', [
        lazy_fixture('tree_incomplete_2_layers'),
        lazy_fixture('undefined_node')
    ])
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
