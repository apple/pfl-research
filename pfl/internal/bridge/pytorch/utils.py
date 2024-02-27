# Copyright © 2023-2024 Apple Inc.
import torch


def clip_norm_and_update(pytorch_model, local_optimizer, train_step_args):
    """ Clipping the gradients followed by a local optimizer step. """
    grad_accumulation_state = train_step_args.grad_accumulation_state
    if train_step_args.grad_scaler is None:
        if grad_accumulation_state.optimizer_should_update:
            if train_step_args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(),
                                               train_step_args.max_grad_norm)
            local_optimizer.step()
    else:
        if grad_accumulation_state.optimizer_should_update:
            if train_step_args.max_grad_norm is not None:
                train_step_args.grad_scaler.unscale_(local_optimizer)
                torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(),
                                               train_step_args.max_grad_norm)
            train_step_args.grad_scaler.step(local_optimizer)
            train_step_args.grad_scaler.update()

    if grad_accumulation_state.optimizer_should_update:
        local_optimizer.zero_grad()
