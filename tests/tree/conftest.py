# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import pytest

from pfl.internal.ops.common_ops import check_pfl_tree_installed

if check_pfl_tree_installed():
    from pfl.tree.gbdt_model import GBDTModelClassifier, GBDTModelRegressor


@pytest.fixture()
def bool_feature(scope='module'):
    from pfl.tree.tree_utils import Feature
    return Feature(2, (0, 1), bool, 1)


@pytest.fixture()
def int_equidistant_feature(scope='module'):
    from pfl.tree.tree_utils import Feature
    return Feature(0, (0, 5), int, 5, 'equidistant')


@pytest.fixture()
def int_random_feature(scope='module'):
    from pfl.tree.tree_utils import Feature
    return Feature(3, (2, 3), int, 3, 'random')


@pytest.fixture()
def float_equidistant_feature(scope='module'):
    from pfl.tree.tree_utils import Feature
    return Feature(1, (0, 100), float, 5, 'equidistant')


@pytest.fixture()
def float_random_feature(scope='module'):
    from pfl.tree.tree_utils import Feature
    return Feature(4, (-2, 3), float, 8, 'random')


@pytest.fixture()
def four_features(bool_feature,
                  int_equidistant_feature,
                  int_random_feature,
                  float_equidistant_feature,
                  scope='module'):

    return [
        bool_feature, int_equidistant_feature, int_random_feature,
        float_equidistant_feature
    ]


@pytest.fixture()
def num_features(scope='module'):
    return 4


@pytest.fixture()
def max_depth(scope='module'):
    return 3


@pytest.fixture()
def alpha(scope='module'):
    return 0.3


@pytest.fixture()
def learning_rate(scope='module'):
    return 0.9


@pytest.fixture(scope='function')
def gbdt_model_classifier_empty(num_features, max_depth, alpha, learning_rate):
    return GBDTModelClassifier(num_features=num_features,
                               max_depth=max_depth,
                               alpha=alpha,
                               learning_rate=learning_rate)


@pytest.fixture(scope='function')
def gbdt_model_classifier_one_tree_incomplete(num_features, max_depth,
                                              tree_incomplete_2_layers,
                                              set_trees, alpha, learning_rate):
    model = GBDTModelClassifier(num_features=num_features,
                                max_depth=max_depth,
                                alpha=alpha,
                                learning_rate=learning_rate)
    return set_trees(model, [tree_incomplete_2_layers])


@pytest.fixture(scope='function')
def gbdt_model_classifier_one_tree_complete(num_features, max_depth,
                                            tree_fully_trained_3_layers,
                                            set_trees, alpha, learning_rate):
    model = GBDTModelClassifier(num_features=num_features,
                                max_depth=max_depth,
                                alpha=alpha,
                                learning_rate=learning_rate)
    return set_trees(model, [tree_fully_trained_3_layers])


@pytest.fixture(scope='function')
def gbdt_model_classifier_two_trees_incomplete(num_features, max_depth,
                                               tree_fully_trained_3_layers,
                                               tree_incomplete_2_layers,
                                               set_trees, alpha,
                                               learning_rate):
    model = GBDTModelClassifier(num_features=num_features,
                                max_depth=max_depth,
                                alpha=alpha,
                                learning_rate=learning_rate)
    return set_trees(model,
                     [tree_fully_trained_3_layers, tree_incomplete_2_layers])


@pytest.fixture(scope='function')
def gbdt_model_classifier_two_trees_complete(num_features, max_depth,
                                             tree_fully_trained_3_layers,
                                             set_trees, alpha, learning_rate):
    model = GBDTModelClassifier(num_features=num_features,
                                max_depth=max_depth,
                                alpha=alpha,
                                learning_rate=learning_rate)
    return set_trees(
        model, [tree_fully_trained_3_layers, tree_fully_trained_3_layers])


@pytest.fixture(scope='function')
def gbdt_model_regressor_empty(num_features, max_depth, alpha, learning_rate):
    return GBDTModelRegressor(num_features=num_features,
                              max_depth=max_depth,
                              alpha=alpha,
                              learning_rate=learning_rate)


@pytest.fixture(scope='function')
def gbdt_model_regressor_one_tree_incomplete(num_features, max_depth,
                                             tree_incomplete_2_layers,
                                             set_trees, alpha, learning_rate):
    model = GBDTModelRegressor(num_features=num_features,
                               max_depth=max_depth,
                               alpha=alpha,
                               learning_rate=learning_rate)
    return set_trees(model, [tree_incomplete_2_layers])


@pytest.fixture(scope='function')
def gbdt_model_regressor_one_tree_complete(num_features, max_depth,
                                           tree_fully_trained_3_layers,
                                           set_trees, alpha, learning_rate):
    model = GBDTModelRegressor(num_features=num_features,
                               max_depth=max_depth,
                               alpha=alpha,
                               learning_rate=learning_rate)
    return set_trees(model, [tree_fully_trained_3_layers])


@pytest.fixture(scope='function')
def gbdt_model_regressor_two_trees_incomplete(num_features, max_depth,
                                              tree_fully_trained_3_layers,
                                              tree_incomplete_2_layers,
                                              set_trees, alpha, learning_rate):
    model = GBDTModelRegressor(num_features=num_features,
                               max_depth=max_depth,
                               alpha=alpha,
                               learning_rate=learning_rate)
    return set_trees(model,
                     [tree_fully_trained_3_layers, tree_incomplete_2_layers])


@pytest.fixture(scope='function')
def gbdt_model_regressor_two_trees_complete(num_features, max_depth,
                                            tree_fully_trained_3_layers,
                                            set_trees, alpha, learning_rate):
    model = GBDTModelRegressor(num_features=num_features,
                               max_depth=max_depth,
                               alpha=alpha,
                               learning_rate=learning_rate)
    return set_trees(
        model, [tree_fully_trained_3_layers, tree_fully_trained_3_layers])
