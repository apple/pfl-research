# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture
from scipy.special import expit

from pfl.internal.tree.gbdt import GBDTClassifier, GBDTRegressor


def set_trees(gbdt, trees):
    gbdt._trees = trees  # pylint: disable=protected-access
    return gbdt


@pytest.fixture()
def empty_gbdt_classifier():
    return GBDTClassifier()


@pytest.fixture()
def gbdt_classifier_one_tree_trained(tree_fully_trained_3_layers):
    gbdt = GBDTClassifier()
    return set_trees(gbdt, [tree_fully_trained_3_layers])


@pytest.fixture()
def gbdt_classifier_two_trees_trained(tree_fully_trained_3_layers):
    gbdt = GBDTClassifier()
    return set_trees(
        gbdt, [tree_fully_trained_3_layers, tree_fully_trained_3_layers])


@pytest.fixture()
def gbdt_classifier_one_tree_incomplete(tree_incomplete_2_layers):
    gbdt = GBDTClassifier()
    return set_trees(gbdt, [tree_incomplete_2_layers])


@pytest.fixture()
def gbdt_classifier_two_trees_incomplete(tree_fully_trained_3_layers,
                                         tree_incomplete_2_layers):
    gbdt = GBDTClassifier()
    return set_trees(gbdt,
                     [tree_fully_trained_3_layers, tree_incomplete_2_layers])


@pytest.fixture()
def empty_gbdt_regressor():
    return GBDTRegressor()


@pytest.fixture()
def gbdt_regressor_one_tree_trained(tree_fully_trained_3_layers):
    gbdt = GBDTRegressor()
    return set_trees(gbdt, [tree_fully_trained_3_layers])


@pytest.fixture()
def gbdt_regressor_two_trees_trained(tree_fully_trained_3_layers):
    gbdt = GBDTRegressor()
    return set_trees(
        gbdt, [tree_fully_trained_3_layers, tree_fully_trained_3_layers])


@pytest.fixture()
def gbdt_regressor_one_tree_incomplete(tree_incomplete_2_layers):
    gbdt = GBDTRegressor()
    return set_trees(gbdt, [tree_incomplete_2_layers])


@pytest.fixture()
def gbdt_regressor_two_trees_incomplete(tree_fully_trained_3_layers,
                                        tree_incomplete_2_layers):
    gbdt = GBDTRegressor()
    return set_trees(gbdt,
                     [tree_fully_trained_3_layers, tree_incomplete_2_layers])


def expected_xgboost_output_one_tree_trained():
    return {
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


class TestGBDT:

    @pytest.mark.parametrize('gbdt', [
        lazy_fixture('empty_gbdt_regressor'),
        lazy_fixture('gbdt_regressor_two_trees_trained'),
        pytest.param(lazy_fixture('gbdt_regressor_one_tree_incomplete'),
                     marks=pytest.mark.xfail)
    ])
    def test_add_tree(self, gbdt, branch_node, check_equal_nodes):
        num_trees = len(gbdt.trees)
        gbdt.add_tree(branch_node)
        assert len(gbdt.trees) == num_trees + 1
        check_equal_nodes(gbdt.trees[-1], branch_node)

    @pytest.mark.parametrize(
        'gbdt, expected',
        [(lazy_fixture('gbdt_regressor_two_trees_trained'),
          [expected_xgboost_output_one_tree_trained()] * 2),
         (lazy_fixture('gbdt_regressor_two_trees_incomplete'),
          [expected_xgboost_output_one_tree_trained()]),
         (lazy_fixture('gbdt_classifier_one_tree_incomplete'), [])])
    def test_to_serialized_xgboost(self, gbdt, expected):
        assert gbdt.to_serialized_xgboost() == expected


class TestGBDTClassifier:

    @pytest.mark.parametrize(
        'gbdt, expected',
        [(lazy_fixture('empty_gbdt_classifier'),
          expit(np.array([0., 0., 0., 0.]))),
         (lazy_fixture('gbdt_classifier_one_tree_trained'),
          expit(np.array([0.3, 0.4, -0.3, -0.4]))),
         (lazy_fixture('gbdt_classifier_two_trees_trained'),
          expit(np.array([0.6, 0.8, -0.6, -0.8]))),
         (lazy_fixture('gbdt_classifier_one_tree_incomplete'),
          expit(np.array([0., 0., 0., 0.]))),
         (lazy_fixture('gbdt_classifier_two_trees_incomplete'),
          expit(np.array([0.3, 0.4, -0.3, -0.4])))])
    def test_predict(self, gbdt, expected, gbdt_datapoints):
        np.testing.assert_equal(gbdt.predict(gbdt_datapoints), expected)

    @pytest.mark.parametrize(
        'gbdt, expected',
        [(lazy_fixture('empty_gbdt_classifier'), np.array([0, 0, 0, 0])),
         (lazy_fixture('gbdt_classifier_one_tree_trained'),
          np.array([1, 1, 0, 0])),
         (lazy_fixture('gbdt_classifier_two_trees_trained'),
          np.array([1, 1, 0, 0])),
         (lazy_fixture('gbdt_classifier_one_tree_incomplete'),
          np.array([0, 0, 0, 0])),
         (lazy_fixture('gbdt_classifier_two_trees_incomplete'),
          np.array([1, 1, 0, 0]))])
    def test_predict_classes(self, gbdt, expected, gbdt_datapoints):
        np.testing.assert_almost_equal(gbdt.predict_classes(gbdt_datapoints),
                                       expected,
                                       decimal=4)

    @pytest.mark.parametrize(
        'gbdt, targets',
        [(lazy_fixture('empty_gbdt_classifier'), np.array([0, 0, 0, 0])),
         (lazy_fixture('gbdt_classifier_one_tree_trained'),
          np.array([1, 1, 0, 0])),
         (lazy_fixture('gbdt_classifier_two_trees_trained'),
          np.array([1, 1, 0, 0])),
         (lazy_fixture('gbdt_classifier_one_tree_incomplete'),
          np.array([0, 0, 0, 0])),
         (lazy_fixture('gbdt_classifier_two_trees_incomplete'),
          np.array([1, 1, 0, 0]))])
    def test_evaluate(self, gbdt, targets, gbdt_datapoints,
                      check_dictionaries_almost_equal):
        predictions = gbdt.predict_classes(gbdt_datapoints)
        expected_metrics = {'accuracy': np.mean(predictions == targets)}
        actual_metrics = gbdt.evaluate(gbdt_datapoints, targets)
        check_dictionaries_almost_equal(actual_metrics, expected_metrics)

    @pytest.mark.parametrize(
        'gbdt, expected',
        [(lazy_fixture('empty_gbdt_classifier'), (expit(0), expit(0))),
         (lazy_fixture('gbdt_classifier_one_tree_trained'),
          (expit(0.4), expit(-0.4))),
         (lazy_fixture('gbdt_classifier_two_trees_trained'),
          (expit(0.8), expit(-0.8))),
         (lazy_fixture('gbdt_classifier_one_tree_incomplete'),
          (expit(0), expit(0))),
         (lazy_fixture('gbdt_classifier_two_trees_incomplete'),
          (expit(0.4), expit(-0.4)))])
    def test_get_max_min_predictions(self, gbdt, expected):
        assert gbdt.get_max_min_predictions() == expected


class TestGBDTRegressor:

    @pytest.mark.parametrize(
        'gbdt, expected',
        [(lazy_fixture('empty_gbdt_regressor'), np.array([0, 0, 0, 0])),
         (lazy_fixture('gbdt_regressor_one_tree_trained'),
          np.array([0.3, 0.4, -0.3, -0.4])),
         (lazy_fixture('gbdt_regressor_two_trees_trained'),
          np.array([0.6, 0.8, -0.6, -0.8])),
         (lazy_fixture('gbdt_regressor_one_tree_incomplete'),
          np.array([0, 0, 0, 0])),
         (lazy_fixture('gbdt_regressor_two_trees_incomplete'),
          np.array([0.3, 0.4, -0.3, -0.4]))])
    def test_predict(self, gbdt, expected, gbdt_datapoints):
        np.testing.assert_equal(gbdt.predict(gbdt_datapoints), expected)

    @pytest.mark.parametrize(
        'gbdt, targets',
        [(lazy_fixture('empty_gbdt_regressor'), np.array([0, 0, 0, 0])),
         (lazy_fixture('gbdt_regressor_one_tree_trained'),
          np.array([1, 1, 0, 0])),
         (lazy_fixture('gbdt_regressor_two_trees_trained'),
          np.array([1, 1, 0, 0])),
         (lazy_fixture('gbdt_regressor_one_tree_incomplete'),
          np.array([0, 0, 0, 0])),
         (lazy_fixture('gbdt_regressor_two_trees_incomplete'),
          np.array([1, 1, 0, 0]))])
    def test_evaluate(self, gbdt, targets, gbdt_datapoints,
                      check_dictionaries_almost_equal):
        predictions = gbdt.predict(gbdt_datapoints)
        expected_metrics = {
            'mae': np.mean(abs(predictions - targets)),
            'mse': np.mean(np.power(predictions - targets, 2))
        }
        actual_metrics = gbdt.evaluate(gbdt_datapoints, targets)
        check_dictionaries_almost_equal(actual_metrics, expected_metrics)

    @pytest.mark.parametrize(
        'gbdt, expected',
        [(lazy_fixture('empty_gbdt_regressor'), (0, 0)),
         (lazy_fixture('gbdt_regressor_one_tree_trained'), (0.4, -0.4)),
         (lazy_fixture('gbdt_regressor_two_trees_trained'), (0.8, -0.8)),
         (lazy_fixture('gbdt_regressor_one_tree_incomplete'), (0, 0)),
         (lazy_fixture('gbdt_regressor_two_trees_incomplete'), (0.4, -0.4))])
    def test_get_max_min_predictions(self, gbdt, expected):
        assert gbdt.get_max_min_predictions() == expected
