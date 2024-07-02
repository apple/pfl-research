
# Copyright © 2023-2024 Apple Inc.
import mlx.core as mx


def get_polynomial_decay_schedule_with_warmup(lr_init,
                                            num_warmup_steps,
                                            num_training_steps,
                                            lr_end=1e-7,
                                                power=1.0):
    """ polynomial LR decay schedule, implementation followed:
    https://huggingface.co/transformers/v4.6.0/_modules/transformers/optimization.html#get_polynomial_decay_schedule_with_warmup """

    assert lr_init >= lr_end, (f"lr_end ({lr_end}) must be be smaller than or "
                            f"equal to initial lr ({lr_init})")

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return current_step.astype(mx.float32) / float(max(1, num_warmup_steps)) * lr_init
        elif current_step > num_training_steps:
            return lr_end
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay

    return lr_lambda
