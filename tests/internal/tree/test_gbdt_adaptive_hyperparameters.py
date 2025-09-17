# Copyright © 2023-2024 Apple Inc.
import pytest
from pytest_lazy_fixtures import lf

from pfl.internal.ops.common_ops import check_pfl_tree_installed

if check_pfl_tree_installed():
    from pfl.internal.tree.gbdt_adaptive_hyperparameters import (
        ClippingBound,
        ComputeSecondOrderGradients,
        TrainCohortSize,
        ValidationCohortSize,
    )
    from pfl.tree.gbdt_model import GBDTModelClassifier, NodeRecord


@pytest.fixture(scope='function')
def gbdt_model_classifier():
    return GBDTModelClassifier(num_features=4, max_depth=3)


@pytest.fixture(scope='function')
def gbdt_model_classifier_incomplete_branch_nodes(tree_incomplete_2_layers,
                                                  set_trees):
    model = GBDTModelClassifier(num_features=4, max_depth=3)
    return set_trees(model, [tree_incomplete_2_layers])


@pytest.fixture(scope='function')
def gbdt_model_classifier_incomplete_leaf_nodes(tree_incomplete_2_layers,
                                                set_trees):
    model = GBDTModelClassifier(num_features=4, max_depth=2)
    return set_trees(model, [tree_incomplete_2_layers])


@pytest.fixture(scope='function')
def gbdt_model_classifier_two_trees_incomplete(tree_fully_trained_3_layers,
                                               tree_incomplete_2_layers,
                                               set_trees):
    model = GBDTModelClassifier(num_features=4, max_depth=3)
    return set_trees(model,
                     [tree_fully_trained_3_layers, tree_incomplete_2_layers])


@pytest.fixture(scope='function')
def gbdt_model_classifier_leaf_node_to_split():
    model = GBDTModelClassifier(num_features=4, max_depth=1)
    model._nodes_to_split = [NodeRecord(None, [], True, None, True)]  # pylint: disable=protected-access
    return model


@pytest.mark.skipif(not check_pfl_tree_installed(),
                    reason='pfl [tree] not installed')
class TestAdaptiveHyperparameters:

    @pytest.mark.parametrize(
        'model, per_layer_modifier_function, base_value, expected, leaf_nodes_reduction_factor',  # pylint: disable=line-too-long
        [(lf('gbdt_model_classifier_incomplete_branch_nodes'),
          'none', 200, 200, 1),
         (lf('gbdt_model_classifier_incomplete_branch_nodes'),
          'linear', 200, 400, 3),
         (lf('gbdt_model_classifier_incomplete_branch_nodes'),
          'power', 200, 400, 1),
         (lf('gbdt_model_classifier_incomplete_leaf_nodes'), 'power',
          200, 40, 5)])
    def test_TrainCohortSize(self, model, per_layer_modifier_function,
                             base_value, expected,
                             leaf_nodes_reduction_factor):
        t = TrainCohortSize(base_value, per_layer_modifier_function,
                            leaf_nodes_reduction_factor)
        assert t.current_value(model) == expected

    @pytest.mark.xfail(raises=AssertionError, strict=True)
    @pytest.mark.parametrize(
        'base_value, per_layer_modifier_function, leaf_nodes_reduction_factor',
        [(1.5, 'none', 1), (-2, 'none', 1), (10, 'other', 1),
         (100, 'none', 0.1), (0, 'none', 1), (0, 'power', 0.2),
         (-1, 'power', '2')])
    def test_TrainCohortSize_fail(self, base_value,
                                  per_layer_modifier_function,
                                  leaf_nodes_reduction_factor):
        TrainCohortSize(base_value, per_layer_modifier_function,
                        leaf_nodes_reduction_factor)

    @pytest.mark.parametrize(
        'model, base_value, expected',
        [(lf('gbdt_model_classifier'), 200, 200),
         (lf('gbdt_model_classifier_incomplete_branch_nodes'), 200,
          0), (lf('gbdt_model_classifier'), 0, 0)])
    def test_ValidationCohortSize(self, model, base_value, expected):
        assert ValidationCohortSize(base_value).current_value(
            model) == expected

    @pytest.mark.xfail(raises=AssertionError, strict=True)
    @pytest.mark.parametrize('base_value', [1.5, -2, -0.01])
    def test_ValidationCohortSize_fail(self, base_value):
        ValidationCohortSize(base_value)

    @pytest.mark.parametrize('model, expected', [
        (lf('gbdt_model_classifier'), 1.5),
        (lf('gbdt_model_classifier_incomplete_branch_nodes'), 0.375),
        (lf('gbdt_model_classifier_two_trees_incomplete'), 0.234375)
    ])
    def test_ClippingBound(self, model, expected):
        base_value = 1.5
        layer_multiplier = 0.5
        tree_multiplier = 0.625
        assert ClippingBound(base_value, layer_multiplier,
                             tree_multiplier).current_value(model) == expected

    @pytest.mark.xfail(raises=AssertionError, strict=True)
    @pytest.mark.parametrize('base_value, layer_multiplier, tree_multiplier',
                             [(-1, 1, 1), (1, 0, 1), (1, 1, 0)])
    def test_ClippingBound_fail(self, base_value, layer_multiplier,
                                tree_multiplier):
        ClippingBound(base_value, layer_multiplier, tree_multiplier)

    @pytest.mark.parametrize('base_value, model, expected', [
        (False, lf('gbdt_model_classifier_incomplete_branch_nodes'),
         False),
        (True, lf('gbdt_model_classifier_incomplete_branch_nodes'),
         True),
        (False, lf('gbdt_model_classifier_leaf_node_to_split'), True)
    ])
    def test_ComputeSecondOrderGradients(self, base_value, model, expected):
        assert ComputeSecondOrderGradients(base_value).current_value(
            model) == expected
