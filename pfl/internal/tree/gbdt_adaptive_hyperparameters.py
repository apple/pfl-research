# Copyright © 2023-2024 Apple Inc.
import math
from abc import ABC, abstractmethod

from pfl.tree.gbdt_model import GBDTModel


class GBDTAdaptiveHyperparameter(ABC):
    """
    Base class for hyperparameters which are adapted during PFL of a GBDT using
    the GBDT model.

    :param base_value:
        Default value for hyperparameter.
    """

    def __init__(self, base_value, *args, **kwargs):
        self._base_value = base_value

    @property
    def base_value(self):
        return self._base_value

    @abstractmethod
    def current_value(self, model: GBDTModel):
        """
        Current value of hyperparameter, which is a function of base value, and
        the GBDT model.
        """
        pass


class TrainCohortSize(GBDTAdaptiveHyperparameter):
    """
    Cohort size used for training each level of a tree in a GBDT.

    Default method for adapting training cohort size per layer of trees in GBDT
    is the "power" method, which ensures the signal-to-noise ratio (SNR) is not
    reduced when training deeper layers of the trees in a GBDT.

    Note that fewer results are required to compute the values of leaf nodes,
    as fewer training statistics are aggregated to compute this value.
    Consequently, the cohort size for training leaf nodes is reduced compared
    to the base value.

    :param base_value:
        Initial value for training cohort size. Gradients from this number of
        users are aggregated to train root node, at top level of a tree in a
        GBDT. This base value is selected to balance the tradeoff between final
        performance of the trained model, the DP guarantees used during
        training, and the time required to wait to aggregate this number of
        results.
    :param per_layer_modifier_function:
        Define function to use to modify the base value of the training cohort
        size depending on the layer of a tree being trained. Options include
        {'none', 'linear', 'power'}, which, respectively, correspond to: not
        adapting the cohort size; linearly increasing the cohort size with the
        layer being trained; exponentially increase the cohort size with the
        layer being trained. 'power' is the default setting, as this helps to
        ensure that SNR does not reduce with the tree depth of the tree being
        trained.
    :param leaf_nodes_reduction_factor:
        Defines the factor by which to reduce the training cohort size from
        the base value when training the maximum depth of the tree, which
        comprises only leaf nodes. The default value is 1, i.e. no reduction
        takes place. However, this can be set to
        `total_num_questions/2^(max_depth - 1)`, where `total_num_questions` is
        the sum of the number of questions specified for each feature used for
        training.
    """

    def __init__(self,
                 base_value: int,
                 per_layer_modifier_function: str = 'power',
                 leaf_nodes_reduction_factor: int = 1):
        assert isinstance(base_value, int) and base_value > 0, (
            'Base value for TrainCohortSize', 'must be an integer > 0')
        assert per_layer_modifier_function in [
            'none', 'linear', 'power'
        ], (f'{per_layer_modifier_function} is not a',
            'valid value for per_layer_modifier_function')
        assert isinstance(leaf_nodes_reduction_factor,
                          int) and leaf_nodes_reduction_factor >= 1

        self._fn_compute_value_branch_nodes = (
            self._get_fn_compute_value_branch_nodes(
                base_value, per_layer_modifier_function))
        self._cohort_size_leaf_nodes = math.ceil(base_value //
                                                 leaf_nodes_reduction_factor)

    def _get_fn_compute_value_branch_nodes(self, base_value,
                                           per_layer_modifier_function):
        if per_layer_modifier_function == 'none':
            return lambda depth: int(base_value)

        if per_layer_modifier_function == 'linear':
            return lambda depth: int(depth * base_value)

        if per_layer_modifier_function == 'power':
            return lambda depth: int(2**(depth - 1) * base_value)

    def current_value(self, model: GBDTModel) -> int:
        if model.current_depth == model.max_depth:
            return self._cohort_size_leaf_nodes

        return self._fn_compute_value_branch_nodes(model.current_depth)


class ValidationCohortSize(GBDTAdaptiveHyperparameter):
    """
    Validation is only performed once per tree being trained.

    As trees are trained layer-wise, validation is only performed while
    training the top layer of the tree. This is because the model's predictions
    will not change until an entire new tree is trained.

    Future enhancement:
    rdar://104295602 (GBDT evaluation should occur at lowest level of tree,
    when largest training cohort size is used)

    :param base_value:
        Number of users from which to gather metrics during validation
        iterations.
    """

    def __init__(self, base_value: int):
        assert isinstance(base_value,
                          int), ('Base value for ValidationCohortSize',
                                 'must be an integer.')
        assert base_value >= 0, ('Base value for ValidationCohortSize',
                                 'must be >= 0.')
        self._base_value = base_value

    def current_value(self, model: GBDTModel) -> int:
        return self._base_value if model.current_depth == 0 else 0


class ClippingBound(GBDTAdaptiveHyperparameter):
    """
    Adapt clipping bound based on current layer and current index of tree being
    trained in a GBDT.

    The sensitivity of the vector of gradients gets smaller as training
    progresses due to smaller gradients as predictions improve. Adapting the
    clipping bound during training improves the SNR.

    :param base_value:
        Default value for clipping bound
    :param layer_multiplier:
        Factor used to modify the base value for clipping bound depending on
        the layer of a tree being trained. Should be in range (0, 1].
    :param tree_multiplier:
        Factor used to modify the base value for clipping bound depending on
        the layer of a tree being trained. Should be in range (0, 1].
    """

    def __init__(self,
                 base_value: float,
                 layer_multiplier: float = 1.,
                 tree_multiplier: float = 1.):
        assert base_value > 0, (f'Invalid base_value {base_value}',
                                'for clipping bound.')
        assert layer_multiplier > 0, (
            'Invalid layer_multiplier',
            f'{layer_multiplier} for clipping bound.')
        assert tree_multiplier > 0, ('Invalid tree_multiplier',
                                     f'{tree_multiplier} for clipping bound.')
        self._base_value = base_value
        self._layer_multiplier = layer_multiplier
        self._tree_multiplier = tree_multiplier

    def current_value(self, model: GBDTModel) -> float:
        return (self.base_value * self._layer_multiplier**model.current_depth *
                self._tree_multiplier**model.current_tree)


class ComputeSecondOrderGradients(GBDTAdaptiveHyperparameter):
    """
    Decide whether or not to compute and aggregate second order gradients
    during training of a GBDT.

    Second order gradients improve the algorithm's ability to identify the
    optimal split for a branch node, or value for a leaf node in a tree.
    However, including second order gradients in the vector of gradients being
    aggregated during training increases the sensitivity of this vector, which
    means that the clipping for DP reduces the sensitivity by a greater factor
    which can result in lower SNR during training.

    Note that second order gradients must be aggregated for leaf nodes, in
    order that leaf values can be computed.

    :param base_value:
        Default setting of whether to aggregate second order gradients during
        training of a tree.
    """

    def __init__(self, base_value: bool):
        assert isinstance(
            base_value,
            bool), 'ComputeSecondOrderGradients base value must be Boolean'
        self._base_value = base_value

    def current_value(self, model: GBDTModel) -> bool:
        if self.base_value:
            return True

        return any(node.is_leaf for node in model.nodes_to_split)
