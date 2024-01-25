# Copyright © 2023-2024 Apple Inc.

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from .node import Node


class GBDT(ABC):
    """
    Gradient Boosted Decision Tree base class.

    :param base_value:
        Default value for all datapoints before any trees are trained and added
        to a GBDT. When trees have been added to a GBDT, the prediction for a
        datapoint is base_value + sum of leaf values assigned to a datapoint
        from each tree in a GBDT.
    """

    def __init__(self, base_value: float = 0):
        self._base_value = base_value
        self._trees: List[Node] = []

    @property
    def trees(self) -> List[Node]:
        return self._trees

    @property
    def base_value(self) -> float:
        return self._base_value

    def __str__(self) -> str:
        """
        Example:
        '
        GBDT
        Number of trees: 1
        Base value: 0

        Tree no. 0
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
        return '\n'.join([
            '\nGBDT', f'Number of trees: {len(self.trees)}',
            f'Base value: {self.base_value}'
        ] + [f'\nTree no. {i} {tree}' for i, tree in enumerate(self.trees)])

    def add_tree(self, root: Node):
        """
        Add the root node of a fully or partially trained tree to the GBDT.

        This will be the root node of a new tree in the GBDT ensemble of trees.
        Ensure the previous tree in the GBDT is fully trained before adding a
        new tree.

        :param root:
            Root node of a fully or partially trained tree to be added to GBDT.
        """
        if len(self.trees) > 0:
            assert self.trees[-1].training_complete(), (
                'Previous tree in '
                'GBDT must be completely trained before adding another tree.')

        self.trees.append(root)

    def get_max_min_predictions(self) -> Tuple[float, float]:
        """
        Return the maximum and minimum predictions possible from a GBDT.

        Maximum prediction = sum of highest value leaf from each tree in GBDT
        plus base value. Minimum prediction = sum of lowest value leaf from
        each tree in GBDT plus base value.

        Only consider trees in GBDT which are fully trained.

        :return:
            Maximum possible prediction from GBDT,
            Minimum possible prediction from GBDT.
        """
        max_value = min_value = self.base_value

        for tree in self.trees:
            if not tree.training_complete():
                break

            leaf_weights = tree.get_leaf_values()
            max_value += np.amax(leaf_weights)
            min_value += np.amin(leaf_weights)

        return max_value, min_value

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make prediction for each datapoint in X using all fully trained trees
        in GBDT.

        The prediction of a datapoint from a GBDT is the sum of the values
        assigned to a datapoint from each tree in the GBDT plus the base value.

        If a GBDT is empty, the prediction of a datapoint is the base value.

        Only fully trained trees in a GBDT are considered during prediction.

        :param X:
            (N x d) array of datapoints, where N is the number of datapoints,
            and d is the number of features per datapoint.

        :return:
            Prediction from GBDT for each datapoint in X.
        """
        if len(self.trees) == 0 or not self.trees[0].training_complete():
            return np.array([self.base_value] * X.shape[0])

        final_index = len(self.trees) if self.trees[-1].training_complete(
        ) else len(self.trees) - 1
        predictions = np.stack(
            [tree.predict(X) for tree in self.trees[:final_index]], axis=0)
        predictions = np.sum(predictions, axis=0)
        predictions = np.add(predictions, self.base_value)

        return predictions

    def to_serialized_xgboost(self) -> List[Dict]:
        """
        Emulate output of xgboost dump_model() method.

        Returns a dictionary describing the GBDT, which looks the same as the
        model description from the xgboost dump_model() method, see
        https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.dump_model

        Only fully trained trees in GBDT will be converted to XGBoost format.

        :return:
            Dictionary describing GBDT, in the format of XGBoost's dump_model()
            function.
        """
        xgboost_gbdt = []
        for tree in self.trees:
            if not tree.training_complete():
                break
            xgboost_gbdt.append(tree.to_serialized_xgboost())

        return xgboost_gbdt

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute performance metrics for GBDT on datapoints in X, given targets
        in y.

        Only fully trained trees in GBDT will be considered during evaluation.

        :param X:
            (N x d) array of datapoints, where N is the number of datapoints,
            and d is the number of features per datapoint.
        :param y:
            Targets corresponding to datapoints in X: array-like, of shape
            (N,) or (N, 1), where N is the number of datapoints.

        :return:
            Dictionary of performance metrics, evaluating GBDT on datapoints
            in X, given targets y.
        """


class GBDTClassifier(GBDT):
    """
    Gradient Boosted Decision Tree class for binary classification.

    :param base_value:
        Default value for all datapoints before any trees are trained and added
        to a GBDT. When trees have been added to a GBDT, the prediction for a
        datapoint is base_value + sum of leaf values assigned to a datapoint
        from each tree in a GBDT.
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make prediction for each datapoint in X using all fully trained trees
        in GBDT.

        The prediction of a datapoint from a classification GBDT is the sigmoid
        of the sum of the values assigned to that datapoint from each tree in
        the GBDT plus the base value.

        If a GBDT is empty, the prediction of a datapoint is the sigmoid of the
        base value of the GBDT.

        Only fully trained trees in a GBDT are considered for prediction.

        :param X:
            (N x d) array of datapoints, where N is the number of datapoints,
            and d is the number of features per datapoint.

        :return:
            Prediction from GBDT for each datapoint in X.
        """
        predictions = super().predict(X)
        return expit(predictions)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary class prediction for each datapoint in X using all fully
        trained trees in GBDT.

        The class prediction is the prediction rounded to the nearest class of
        0 and 1. Predictions in range (-inf, 0.5] map to class 0. Predictions
        in range (0.5, +inf) map to class 1. Note that the prediction used
        to calculate the class is the sigmoid of the sum of the values assigned
        to the datapoint from each tree in the GBDT plus the base value.

        Only fully trained trees in GBDT will be considered for prediction.

        :param X:
            (N x d) array of datapoints, where N is the number of datapoints,
            and d is the number of features per datapoint.

        :return:
            Prediction from GBDT for each datapoint in X.
        """
        predictions = self.predict(X)
        return (predictions > 0.5).astype(np.int32)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute accuracy of GBDT binary classifier on data X, given targets y.

        Evaluation will only be performed on fully trained trees in GBDT.

        :param X:
            (N x d) array of datapoints, where N is the number of datapoints,
            and d is the number of features per datapoint.
        :param y:
            Targets corresponding to datapoints in X: array-like, of shape
            (N,) or (N, 1), where N is the number of datapoints.

        :return:
            Dictionary of metrics, including accuracy of GBDT on data X, with
            targets y.
        """
        predictions = self.predict_classes(X)
        return {'accuracy': accuracy_score(y, predictions)}

    def get_max_min_predictions(self) -> Tuple[float, float]:
        """
        Return the maximum and minimum predictions possible from a GBDT.

        Maximum prediction = expit of sum of highest value leaf from each tree
        in GBDT plus base value. Minimum prediction = expit of sum of lowest
        value leaf from each tree in GBDT plus base value.

        Only consider trees in GBDT which are fully trained.

        :return:
            Maximum possible prediction from GBDT,
            Minimum possible prediction from GBDT.
        """
        (max_value, min_value) = super().get_max_min_predictions()
        return expit(max_value), expit(min_value)


class GBDTRegressor(GBDT):
    """
    Gradient Boosted Decision Tree class for regression.

    :param num_features:
        Number of features of datapoints used to train GBDT.
    :param base_value:
        Default value for all datapoints before any trees are trained and added
        to a GBDT. When trees have been added to a GBDT, the prediction for a
        datapoint is base_value + sum of leaf values assigned to a datapoint
        from each tree in a GBDT.
    """

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute mean-absolute error (MAE), and mean-squared error (MSE) of
        regression GBDT, given data X and targets y.

        Evaluation will only be performed on fully trained trees in GBDT.

        :param X:
            (N x d) array of datapoints, where N is the number of datapoints,
            and d is the number of features per datapoint.
        :param y:
            Targets corresponding to datapoints in X: array-like, of shape
            (N,) or (N, 1), where N is the number of datapoints.

        :return:
            Dictionary of metrics, including MAE and MSE of GBDT on data X,
            with targets y.
        """
        predictions = self.predict(X)
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)

        return {'mae': mae, 'mse': mse}
