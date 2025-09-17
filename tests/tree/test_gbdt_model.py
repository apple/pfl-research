# Copyright © 2023-2024 Apple Inc.
from typing import List, cast

import numpy as np
import pytest
from pytest_lazy_fixtures import lf
from scipy.special import expit

from pfl.internal.ops.common_ops import check_pfl_tree_installed
from pfl.metrics import Weighted
from pfl.stats import MappedVectorStatistics

if check_pfl_tree_installed():
    from pfl.tree.gbdt_model import GBDTModelClassifier, NodeRecord

regression_first_order_gradients_left = np.array(
    [-2, -2, -1, -1, 0, 0, 0, 0, -2, -2, -1, -1, -1, -1, -1, -1])
regression_first_order_gradients_right = np.array(
    [0, 0, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0])
regression_second_order_gradients_left = np.array(
    [2, 2, 1, 3, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1])
regression_second_order_gradients_right = np.array(
    [2, 2, 3, 1, 2, 2, 2, 0, 0, 0, 1, 1, 0, 0, 0, 0])

classification_first_order_gradients_left = np.array([
    -1.0, -1.0, -0.5, 0.5, 0, 0, 0, 1, -1.0, -1.0, -0.5, -0.5, -0.5, -0.5,
    -0.5, -0.5
])
classification_first_order_gradients_right = np.array(
    [1, 1, 0.5, -0.5, 1, 1, 1, 0, 0, 0, -0.5, -0.5, 0, 0, 0, 0])
classification_second_order_gradients_left = np.array([
    0.5, 0.5, 0.25, 0.75, 0, 0, 0, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.25,
    0.25
])
classification_second_order_gradients_right = np.array(
    [0.5, 0.5, 0.75, 0.25, 0.5, 0.5, 0.5, 0, 0, 0, 0.25, 0.25, 0, 0, 0, 0])

regression_first_order_gradients: List = []
regression_second_order_gradients: List = []
classification_first_order_gradients: List = []
classification_second_order_gradients: List = []
for i in range(4):
    regression_first_order_gradients.extend(
        regression_first_order_gradients_left[i * 4:i * 4 + 4])
    regression_first_order_gradients.extend(
        regression_first_order_gradients_right[i * 4:i * 4 + 4])
    regression_second_order_gradients.extend(
        regression_second_order_gradients_left[i * 4:i * 4 + 4])
    regression_second_order_gradients.extend(
        regression_second_order_gradients_right[i * 4:i * 4 + 4])
    classification_first_order_gradients.extend(
        classification_first_order_gradients_left[i * 4:i * 4 + 4])
    classification_first_order_gradients.extend(
        classification_first_order_gradients_right[i * 4:i * 4 + 4])
    classification_second_order_gradients.extend(
        classification_second_order_gradients_left[i * 4:i * 4 + 4])
    classification_second_order_gradients.extend(
        classification_second_order_gradients_right[i * 4:i * 4 + 4])

regression_first_order_gradients_one_side_only_node_sum = (
    regression_first_order_gradients_left.tolist())
regression_first_order_gradients_one_side_only_one_right_gradient = (
    regression_first_order_gradients_left.tolist())
for i in range(4):
    regression_first_order_gradients_one_side_only_node_sum.insert(
        4 * (i + 1) + i,
        regression_first_order_gradients_left[4 * (i + 1) - 1] +
        regression_first_order_gradients_right[4 * (i + 1) - 1])
    regression_first_order_gradients_one_side_only_one_right_gradient.insert(
        4 * (i + 1) + i,
        regression_first_order_gradients_right[4 * (i + 1) - 1])

regression_first_order_gradients_feature_difference = np.copy(
    regression_first_order_gradients)
regression_first_order_gradients_feature_difference[1:] = np.subtract(
    regression_first_order_gradients_feature_difference[1:],
    regression_first_order_gradients_feature_difference[0:-1])
regression_first_order_gradients_feature_difference[
    0:-1:2] = regression_first_order_gradients[0:-1:2]

regression_first_order_gradients_node_difference = np.copy(
    regression_first_order_gradients)
regression_first_order_gradients_node_difference[1:] = np.subtract(
    regression_first_order_gradients_node_difference[1:],
    regression_first_order_gradients_node_difference[0:-1])
regression_first_order_gradients_node_difference[
    0:-1:4] = regression_first_order_gradients[0:-1:4]

vector_shape = np.hstack((regression_first_order_gradients,
                          regression_second_order_gradients)).shape
weight_translate_vector = np.random.normal(0, 1, vector_shape)


@pytest.fixture()
def root_NodeRecord(scope='function'):
    return NodeRecord(None, [], True, None, False)


@pytest.mark.skipif(not check_pfl_tree_installed(),
                    reason='pfl [tree] not installed')
class TestGBDTModel:

    def test_model_init(self, gbdt_model_classifier_empty, root_NodeRecord):
        assert gbdt_model_classifier_empty.nodes_to_split == [root_NodeRecord]

    @pytest.mark.parametrize(
        'model, expected',
        [(lf('gbdt_model_classifier_empty'), 0),
         (lf('gbdt_model_classifier_one_tree_incomplete'), 2),
         (lf('gbdt_model_classifier_one_tree_complete'), 0),
         (lf('gbdt_model_regressor_two_trees_incomplete'), 2),
         (lf('gbdt_model_regressor_two_trees_complete'), 0)])
    def test_current_depth(self, model, expected):
        assert model.current_depth == expected

    @pytest.mark.parametrize('model, num_nodes_to_split', [
        (lf('gbdt_model_classifier_empty'), 1),
        (lf('gbdt_model_classifier_one_tree_incomplete'), 4),
        (lf('gbdt_model_classifier_one_tree_complete'), 1),
        (lf('gbdt_model_classifier_two_trees_incomplete'), 4),
        (lf('gbdt_model_classifier_two_trees_complete'), 1),
    ])
    def test_load_and_save(self, model, num_nodes_to_split, gbdt_datapoints,
                           num_features, max_depth, tmp_path, root_NodeRecord):
        new_model = GBDTModelClassifier(num_features=num_features,
                                        max_depth=max_depth)
        model.save(tmp_path)
        new_model.load(str(tmp_path / 'gbdt.json'))
        np.testing.assert_array_equal(model.predict(gbdt_datapoints),
                                      new_model.predict(gbdt_datapoints))
        assert len(model.nodes_to_split) == num_nodes_to_split

    def test_apply_model_update(self, gbdt_model_regressor_empty,
                                root_NodeRecord):
        (feature, threshold, gain, value, left_child_value,
         right_child_value) = (0, 1, 2, 3, 4, 5)

        # add branch root node
        statistics = MappedVectorStatistics({
            'node_0':
            np.array([
                feature, threshold, gain, value, left_child_value,
                right_child_value
            ])
        })
        model, _ = gbdt_model_regressor_empty.apply_model_update(statistics)

        assert len(model.nodes_to_split) == 2
        left_child = model.nodes_to_split[0]
        right_child = model.nodes_to_split[1]
        assert left_child.decision_path == [[feature, threshold, True]]
        assert left_child.value == left_child_value
        assert left_child.is_leaf is False
        assert right_child.decision_path == [[feature, threshold, False]]
        assert right_child.value == right_child_value
        assert right_child.is_leaf is False

        # add child nodes - two branch nodes
        statistics = MappedVectorStatistics({
            'node_0':
            np.array([
                feature, threshold, gain, value, left_child_value,
                right_child_value
            ]),
            'node_1':
            np.array([
                feature, threshold, gain, value, left_child_value,
                right_child_value
            ])
        })
        model, _ = model.apply_model_update(statistics)

        assert len(model.nodes_to_split) == 4
        left_left_child = model.nodes_to_split[0]
        left_right_child = model.nodes_to_split[1]
        right_left_child = model.nodes_to_split[2]
        right_right_child = model.nodes_to_split[3]
        assert left_left_child.decision_path == [[feature, threshold, True],
                                                 [feature, threshold, True]]
        assert left_left_child.value == left_child_value
        assert left_left_child.is_leaf is True
        assert left_right_child.decision_path == [[feature, threshold, True],
                                                  [feature, threshold, False]]
        assert left_right_child.value == right_child_value
        assert left_right_child.is_leaf is True
        assert right_left_child.decision_path == [[feature, threshold, False],
                                                  [feature, threshold, True]]
        assert right_left_child.value == left_child_value
        assert right_left_child.is_leaf is True
        assert right_right_child.decision_path == [[feature, threshold, False],
                                                   [feature, threshold, False]]
        assert right_right_child.value == right_child_value
        assert right_right_child.is_leaf is True

        # add child leaf nodes
        statistics = MappedVectorStatistics({
            'node_0':
            np.array([
                feature, threshold, gain, value, left_child_value,
                right_child_value
            ]),
            'node_1':
            np.array([
                feature, threshold, gain, value, left_child_value,
                right_child_value
            ]),
            'node_2':
            np.array([
                feature, threshold, gain, value, left_child_value,
                right_child_value
            ]),
            'node_3':
            np.array([
                feature, threshold, gain, value, left_child_value,
                right_child_value
            ])
        })
        model, _ = model.apply_model_update(statistics)

        assert model.nodes_to_split == [root_NodeRecord]

    @pytest.mark.parametrize('model, value, expected', [
        (lf('gbdt_model_classifier_empty'), 1.0, 1.0),
        (lf('gbdt_model_classifier_empty'), -1.0, -1.0),
        (lf('gbdt_model_classifier_one_tree_complete'), 1.0, 1.0),
        (lf('gbdt_model_classifier_one_tree_complete'), -1.0, -1.0),
        (lf('gbdt_model_classifier_one_tree_incomplete'), 1.0, 1.0),
        (lf('gbdt_model_classifier_one_tree_incomplete'), -1.0,
         -1.0),
        (lf('gbdt_model_classifier_two_trees_complete'), 2.0, 2.0),
        (lf('gbdt_model_classifier_two_trees_complete'), -0.5, -0.5)
    ])
    def test_compute_node_value_no_prior(self, model, value, expected,
                                         learning_rate):
        leaf_value = model._compute_node_value(value)  # pylint: disable=protected-access
        expected_leaf_value = learning_rate * expected
        np.testing.assert_almost_equal(leaf_value, expected_leaf_value)

    @pytest.mark.parametrize('model, value, expected', [
        (lf('gbdt_model_classifier_empty'), 1.0, 0.79),
        (lf('gbdt_model_classifier_empty'), -1.0, -0.61),
        (lf('gbdt_model_classifier_one_tree_complete'), 1.0, 0.79),
        (lf('gbdt_model_classifier_one_tree_complete'), -1.0, -0.61),
        (lf('gbdt_model_classifier_one_tree_incomplete'), 1.0, 0.79),
        (lf('gbdt_model_classifier_one_tree_incomplete'), -1.0,
         -0.61),
        (lf('gbdt_model_classifier_two_trees_complete'), 2.0, 1.49),
        (lf('gbdt_model_classifier_two_trees_complete'), -0.5, -0.26)
    ])
    def test_compute_node_value_with_prior(self, model, value, expected,
                                           learning_rate):
        prior_value = 0.3
        leaf_value = model._compute_node_value(value, prior_value)  # pylint: disable=protected-access
        expected_leaf_value = learning_rate * expected
        np.testing.assert_almost_equal(leaf_value, expected_leaf_value)


@pytest.mark.skipif(not check_pfl_tree_installed(),
                    reason='pfl [tree] not installed')
class TestGBDTModelClassifier:

    @pytest.mark.parametrize(
        'model, expected',
        [(lf('gbdt_model_classifier_empty'),
          expit(np.array([0., 0., 0., 0.]))),
         (lf('gbdt_model_classifier_one_tree_complete'),
          expit(np.array([0.3, 0.4, -0.3, -0.4]))),
         (lf('gbdt_model_classifier_two_trees_incomplete'),
          expit(np.array([0.3, 0.4, -0.3, -0.4])))])
    def test_predict(self, model, expected, gbdt_datapoints):
        np.testing.assert_almost_equal(model.predict(gbdt_datapoints),
                                       expected,
                                       decimal=3)

    @pytest.mark.parametrize(
        'model, expected',
        [(lf('gbdt_model_classifier_empty'), np.array([0, 0, 0, 0])),
         (lf('gbdt_model_classifier_one_tree_complete'),
          np.array([1, 1, 0, 0])),
         (lf('gbdt_model_classifier_two_trees_incomplete'),
          np.array([1, 1, 0, 0]))])
    def test_predict_classes(self, model, expected, gbdt_datapoints):
        np.testing.assert_almost_equal(model.predict_classes(gbdt_datapoints),
                                       expected,
                                       decimal=4)

    @pytest.mark.parametrize('model', [
        lf('gbdt_model_classifier_empty'),
        lf('gbdt_model_classifier_one_tree_complete'),
        lf('gbdt_model_classifier_two_trees_complete'),
        lf('gbdt_model_classifier_one_tree_incomplete'),
        lf('gbdt_model_classifier_two_trees_incomplete'),
    ])
    def test_evaluate(self, model, gbdt_user_dataset):
        predictions = model.predict_classes(gbdt_user_dataset.raw_data[0])
        targets = gbdt_user_dataset.raw_data[1]
        accuracy = np.mean(predictions == targets)
        metrics = model.evaluate(gbdt_user_dataset, lambda n: f'{n}')

        per_user_accuracy = cast(Weighted, metrics['per-user accuracy'])
        assert per_user_accuracy.weighted_value == accuracy
        assert per_user_accuracy.weight == 1
        assert per_user_accuracy.overall_value == accuracy

        overall_accuracy = cast(Weighted, metrics['overall accuracy'])
        assert overall_accuracy.weighted_value == accuracy * len(targets)
        assert overall_accuracy.weight == len(targets)
        assert overall_accuracy.overall_value == accuracy


@pytest.mark.skipif(not check_pfl_tree_installed(),
                    reason='pfl [tree] not installed')
class TestGBDTModelRegressor:

    @pytest.mark.parametrize(
        'model, expected',
        [(lf('gbdt_model_regressor_empty'), np.array([0, 0, 0, 0])),
         (lf('gbdt_model_regressor_one_tree_complete'),
          np.array([0.3, 0.4, -0.3, -0.4])),
         (lf('gbdt_model_regressor_two_trees_incomplete'),
          np.array([0.3, 0.4, -0.3, -0.4]))])
    def test_predict(self, model, expected, gbdt_datapoints):
        np.testing.assert_almost_equal(model.predict(gbdt_datapoints),
                                       expected,
                                       decimal=4)

    @pytest.mark.parametrize('model', [
        lf('gbdt_model_regressor_empty'),
        lf('gbdt_model_regressor_one_tree_complete'),
        lf('gbdt_model_regressor_two_trees_complete'),
        lf('gbdt_model_regressor_one_tree_incomplete'),
        lf('gbdt_model_regressor_two_trees_incomplete'),
    ])
    def test_evaluate(self, model, gbdt_user_dataset):
        # Only installed with [tree] install extra.
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        predictions = model.predict(gbdt_user_dataset.raw_data[0])
        targets = gbdt_user_dataset.raw_data[1]
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        metrics = model.evaluate(gbdt_user_dataset, lambda n: f'{n}')

        per_user_mae = cast(Weighted, metrics['per-user mae'])
        assert per_user_mae.weighted_value == mae
        assert per_user_mae.weight == 1
        assert per_user_mae.overall_value == mae

        overall_mae = cast(Weighted, metrics['overall mae'])
        assert overall_mae.weighted_value == mae * len(targets)
        assert overall_mae.weight == len(targets)
        assert overall_mae.overall_value == mae

        per_user_mse = cast(Weighted, metrics['per-user mse'])
        assert per_user_mse.weighted_value == mse
        assert per_user_mse.weight == 1
        assert per_user_mse.overall_value == mse

        overall_mse = cast(Weighted, metrics['overall mse'])
        assert overall_mse.weighted_value == mse * len(targets)
        assert overall_mse.weight == len(targets)
        assert overall_mse.overall_value == mse
