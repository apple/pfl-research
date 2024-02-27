# Copyright © 2023-2024 Apple Inc.

import inspect
import logging
import types
from typing import Any, Callable, Dict, Optional

import torch
from dataset.hugging_face import IGNORE_INDEX
from peft import PeftConfig, get_peft_model
from transformers import PreTrainedModel, PreTrainedTokenizer

from pfl.metrics import MetricValue, Summed, Weighted

logger = logging.getLogger(__name__)


def add_special_tokens(tokenizer: PreTrainedTokenizer):
    """ Add special tokens if they don't exist in the tokenzier. """
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "[PAD]"  # noqa: S105
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "</s>"  # noqa: S105
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "<s>"  # noqa: S105
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = "<unk>"  # noqa: S105
    return tokenizer.add_special_tokens(special_tokens_dict)


def smart_embedding_resize(
    num_new_tokens: int,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    logger.info(f"Resizing model's token embedding to {len(tokenizer)} "
                f"with {num_new_tokens} new tokens.")
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _get_forward_inputs(forward_signature: inspect.Signature,
                        **kwargs) -> Dict[str, Any]:
    inputs = {}
    for p in forward_signature.parameters:
        if p in kwargs:
            inputs[p] = kwargs[p]
    return inputs


def causal_lm_metrics_fn(model: PreTrainedModel,
                         **kwargs) -> Dict[str, MetricValue]:
    """ Add common metrics used in Causal LM model. """
    forward_signature = inspect.signature(model.forward)
    inputs = _get_forward_inputs(forward_signature, **kwargs)
    inputs["return_dict"] = True
    outputs = model.forward(**inputs)
    # In Hugging Face Causal LM, labels are shifted by 1 to calculate loss
    shifted_labels = inputs["labels"][..., 1:]
    num_tokens = torch.sum((shifted_labels != IGNORE_INDEX).float()).item()
    loss = outputs["loss"].item() * num_tokens
    metrics: Dict[str, MetricValue] = {
        "loss": Weighted(loss, num_tokens),
        "number of tokens": Summed(num_tokens)
    }
    if "logits" in outputs:
        # Add LM next token prediction accuracy
        predicted_labels = torch.argmax(outputs["logits"][..., :-1, :], dim=-1)
        num_correct = torch.sum(
            (shifted_labels == predicted_labels).float()).item()
        metrics["accuracy"] = Weighted(num_correct, num_tokens)
    return metrics


def wrap_hugging_face_model(
    model: PreTrainedModel,
    peft_config: Optional[PeftConfig] = None,
    metrics_fn: Optional[Callable] = None,
) -> PreTrainedModel:
    """
    Wrap a Hugging Face model, so it has `loss` and `metrics` functions and
    can be used in `pfl`.
    """
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.info(f"trainable params: {trainable_params:,d} || "
                    f"all params: {all_param:,d} || "
                    f"trainable%: {100 * trainable_params / all_param}")

    if hasattr(model, "loss") and hasattr(model, "metrics"):
        return model

    # PFL requires PyTorch module to implement loss and metrics functions.
    # Add these functions manually
    forward_signature = inspect.signature(model.forward)

    def compute_loss(self: PreTrainedModel, **kwargs):
        inputs = _get_forward_inputs(forward_signature, **kwargs)
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
        if metrics_fn is not None:
            return metrics_fn(self, **kwargs)
        # If no metrics_fn provided, return the per-batch loss as metrics
        loss = compute_loss(self, **kwargs).item()
        return {"per-batch loss": Weighted.from_unweighted(loss)}

    if not hasattr(model, "loss"):
        model.loss = types.MethodType(compute_loss, model)
    if not hasattr(model, "metrics"):
        model.metrics = types.MethodType(compute_metrics, model)
    return model
