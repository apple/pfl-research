# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import pytest

from pfl.internal.tree.node import Node


@pytest.fixture(scope='function')
def branch_node():
    return Node(feature=0, threshold=2)


@pytest.fixture(scope='function')
def leaf_node():
    return Node(value=0.3)


@pytest.fixture(scope='function')
def undefined_node():
    return Node()


@pytest.fixture(scope='session')
def check_equal_nodes():

    def _check_equal_nodes(node_1: Node, node_2: Node):
        assert node_1.value == node_2.value
        assert node_1.feature == node_2.feature
        assert node_1.threshold == node_2.threshold
        if node_1.left_child or node_2.left_child:
            assert _check_equal_nodes(node_1.left_child, node_2.left_child)
        if node_1.right_child or node_2.right_child:
            assert _check_equal_nodes(node_1.right_child, node_2.right_child)

    return _check_equal_nodes
