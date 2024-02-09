import inspect
import logging
import types
from typing import Optional

from peft import PeftConfig, get_peft_model
from transformers import PreTrainedModel

from pfl.metrics import Weighted

logger = logging.getLogger(__name__)


def wrap_hugging_face_model(
        model: PreTrainedModel,
        peft_config: Optional[PeftConfig] = None) -> PreTrainedModel:
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(f"trainable params: {trainable_params:,d} || "
                    f"all params: {all_param:,d} || "
                    f"trainable%: {100 * trainable_params / all_param}")

    forward_signature = inspect.signature(model.forward)

    def compute_loss(self: PreTrainedModel, **kwargs):
        inputs = {}
        for p in forward_signature.parameters:
            if p in kwargs:
                inputs[p] = kwargs[p]
        outputs = self.forward(**inputs)
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, "
                f"only the following keys: {','.join(outputs.keys())}. "
                "For reference, the inputs it received are "
                f"{','.join(inputs.keys())}.")
        # We don't use .loss here since the model may return tuples instead of
        # ModelOutput.
        return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    def compute_metrics(self: PreTrainedModel, **kwargs):
        loss = compute_loss(self, **kwargs).item()
        return {"loss": Weighted.from_unweighted(loss)}

    model.loss = types.MethodType(compute_loss, model)  # type: ignore
    model.metrics = types.MethodType(compute_metrics, model)  # type: ignore
    return model
