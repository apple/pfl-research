# Copyright © 2023-2024 Apple Inc.
from torch.optim.lr_scheduler import LambdaLR


def get_polynomial_decay_schedule_with_warmup(optimizer,
                                              num_warmup_steps,
                                              num_training_steps,
                                              lr_end=1e-7,
                                              power=1.0,
                                              last_epoch=-1):
    """ polynomial LR decay schedule, implementation followed:
    https://huggingface.co/transformers/v4.6.0/_modules/transformers/optimization.html#get_polynomial_decay_schedule_with_warmup """

    lr_init = optimizer.defaults["lr"]
    assert lr_init >= lr_end, (f"lr_end ({lr_end}) must be be smaller than or "
                               f"equal to initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)
