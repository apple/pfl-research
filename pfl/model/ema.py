# Copyright © 2023-2024 Apple Inc.
from typing import Any, Dict

from pfl.internal.ops.selector import get_framework_module

from .base import StatefulModel


class CentralExponentialMovingAverage:
    """
    Maintains moving averages of variables by employing an exponential decay.
    When training a model, it is often beneficial to maintain moving averages of
    the trained parameters. Evaluations that use averaged parameters sometimes
    produce better results.

    EMA update is described as in the following formula:
    `shadow_variable -= (1 - decay) * (shadow_variable - variable)`,
    where `shadow_variable` is the moving average of `variable`.

    Reasonable values for `decay` are close to 1.0, typically in the
    multiple-nines range: 0.99, 0.999, etc.

    :example:

        .. code-block:: python

            ema = CentralExponentialMovingAverage(model, 0.999)
            # ... training the model
            model.apply_model_update(statistics)
            ema.update()
            # backup model parameters in ``state''
            state = model.get_parameters()
            # set model parameters to their EMA values
            ema.assign()
            # ... evaluating the model
            # restore the original model state
            model.set_parameters(state)

    :param decay:
        EMA decaying rate, a floating point number between 0 and 1
    :param dynamic_decay_rate:
        Whether to tweak the decay rate dynamically using the
        count of training steps. If `True`, the actual decay rate used is:
        `min(decay, (1 + num_updates) / (10 + num_updates))`
        in which case the decay rate is lower at the start of training.
        This makes moving averages move faster in the early iterations.
    """

    def __init__(self,
                 model: StatefulModel,
                 decay: float,
                 dynamic_decay_rate: bool = False):
        assert 0. < decay < 1.0, "EMA decaying rate should be between 0 and 1"
        self._model = model
        self._decay = decay
        self._dynamic_decay_rate = dynamic_decay_rate
        self._num_updates = 0
        self._initialize()

    @property
    def shadow_variable_map(self) -> Dict[str, Any]:
        """
        :return:
            A dictionary that maps variable name to the shadow EMA
            variables of ``self._model``.
        """
        return self._shadow_variable_map

    @property
    def decay(self) -> float:
        """
        :return:
            A ``float`` for the EMA decaying rate. Dynamically set if
            ``self._dynamic_decay_rate = True``
        """
        if self._dynamic_decay_rate:
            dynamic_decay = (1 + self._num_updates) / (10 + self._num_updates)
            return min(self._decay, dynamic_decay)
        return self._decay

    def _initialize(self) -> None:
        """
        Initialize shadow EMA variables with model's current variables
        """
        variable_map = self._get_model_variable_map()
        # According to TensorFlow implementation, EMA variables
        # are not initialized to zeros but to the original variables
        self._shadow_variable_map = {
            variable_name:
            get_framework_module().clone_variable(variable, name="ema")
            for variable_name, variable in variable_map.items()
        }

    def update(self) -> None:
        """
        Perform one step of EMA update on shadow variables using framework's
        specific operation after each central optimization step.
        """
        variable_map = self._get_model_variable_map()
        variable_names = variable_map.keys()
        get_framework_module().exponential_moving_average_update(
            variables=[
                variable_map[variable_name] for variable_name in variable_names
            ],
            ema_variables=[
                self._shadow_variable_map[variable_name]
                for variable_name in variable_names
            ],
            decay=self.decay)
        self._num_updates += 1

    def assign(self) -> None:
        """
        Assign the EMA shadow variables to model variables
        """
        variable_map = self._get_model_variable_map()
        for variable_name in variable_map:
            get_framework_module().assign_variable(
                reference=variable_map[variable_name],
                value=self._shadow_variable_map[variable_name])

    def save(self, dir_path: str) -> None:
        """
        Save the EMA shadow variables. First model need to backup the current
        variables and then assign the EMA variables for saving. Then model need
        to restore the variables
        """
        model_state = self._model.get_parameters()
        self.assign()
        self._model.save(dir_path)
        self._model.set_parameters(model_state)

    def _get_model_variable_map(self) -> Dict[str, Any]:
        assert hasattr(self._model, "variable_map"), (
            "EMA only support `Model` classes with `variable_map` property "
            "(e.g. `TFModel`, and `PyTorchModel`).")
        return self._model.variable_map
