# Copyright © 2023-2024 Apple Inc.

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.internal.ops.common_ops import check_pfl_tree_installed

if check_pfl_tree_installed():
    from pfl.tree.gbdt_model import GBDTModelClassifier, NodeRecord
    from pfl.tree.tree_utils import Feature, choose_questions


@pytest.fixture()
def bool_feature(scope='module'):
    return Feature(2, (0, 1), bool, 1)


@pytest.fixture()
def int_equidistant_feature(scope='module'):
    return Feature(0, (0, 5), int, 5, 'equidistant')


@pytest.fixture()
def int_random_feature(scope='module'):
    return Feature(3, (2, 3), int, 3, 'random')


@pytest.fixture()
def float_equidistant_feature(scope='module'):
    return Feature(1, (0, 100), float, 5, 'equidistant')


@pytest.fixture()
def float_random_feature(scope='module'):
    return Feature(4, (-2, 3), float, 8, 'random')


@pytest.fixture()
def node_record(scope='module'):
    decision_path = [[0, 4, True], [1, 10, False], [2, 0.5, True]]
    is_left = True
    return NodeRecord(None, decision_path, is_left, None, False)


@pytest.fixture()
def features(bool_feature,
             int_equidistant_feature,
             int_random_feature,
             float_equidistant_feature,
             float_random_feature,
             scope='module'):

    return [
        bool_feature, int_equidistant_feature, int_random_feature,
        float_equidistant_feature, float_random_feature
    ]


@pytest.fixture(scope='function')
def gbdt_model_classifier_incomplete_branch_nodes(tree_incomplete_2_layers,
                                                  set_trees):
    model = GBDTModelClassifier(num_features=4, max_depth=3)
    return set_trees(model, [tree_incomplete_2_layers])


@pytest.mark.skipif(not check_pfl_tree_installed(),
                    reason='pfl [tree] not installed')
class TestFeature:

    @pytest.mark.parametrize('feature, expected', [
        (lazy_fixture('bool_feature'), []),
        (lazy_fixture('int_equidistant_feature'), [0.8, 1.6, 2.4, 3.2]),
        (lazy_fixture('float_equidistant_feature'),
         [25.0, 40.0, 55.0, 70.0, 85.0]),
    ])
    def test_generate_feature_questions(self, feature, node_record, expected):
        questions = feature.generate_feature_questions(node_record)
        np.testing.assert_almost_equal(questions, expected, decimal=2)


@pytest.mark.skipif(not check_pfl_tree_installed(),
                    reason='pfl [tree] not installed')
def test_choose_questions(features,
                          gbdt_model_classifier_incomplete_branch_nodes):
    nodes = gbdt_model_classifier_incomplete_branch_nodes.nodes_to_split
    questions, total_num_questions = choose_questions(nodes, features)

    assert total_num_questions == 69
    assert len(questions) == 4  # 4 nodes to split in tree
    for node_questions in questions:
        assert type(node_questions) == dict
        assert set(node_questions.keys()) == {'decisionPath', 'splits'}
        assert set(node_questions['splits'].keys()) == {0, 1, 2, 3, 4}
