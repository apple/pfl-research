# Copyright © 2023-2024 Apple Inc.
from typing import Optional

from lm.argument_parsing import add_lm_arguments
from peft import LoraConfig, PeftConfig, TaskType
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from utils.argument_parsing import store_bool


def parse_peft_config(args) -> Optional[PeftConfig]:
    if args.peft_type is None:
        return None
    assert args.peft_type == 'lora', "Currently only supports PEFT with LoRA."
    return LoraConfig(task_type=args.peft_task_type,
                      r=args.lora_r,
                      lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout)


def add_peft_arguments(argument_parser):
    argument_parser.add_argument(
        '--peft_type',
        type=str,
        default=None,
        choices=['lora'],
        help='Hugging Face PEFT type. Currently only supports LoRA.')

    argument_parser.add_argument('--peft_task_type',
                                 type=str,
                                 default="CAUSAL_LM",
                                 choices=[t.value for t in TaskType],
                                 help='Hugging Face PEFT task type.')

    known_args, _ = argument_parser.parse_known_args()
    if known_args.peft_type == 'lora':
        argument_parser.add_argument('--lora_r',
                                     type=int,
                                     default=8,
                                     help='LoRA attention dimension.')

        argument_parser.add_argument('--lora_alpha',
                                     type=int,
                                     default=8,
                                     help='LoRA alpha.')

        argument_parser.add_argument('--lora_dropout',
                                     type=float,
                                     default=0.0,
                                     help='LoRA dropout.')

    return argument_parser


def add_central_lr_scheduler_arguments(argument_parser):
    argument_parser.add_argument(
        '--central_lr_scheduler',
        type=str,
        default=None,
        choices=['constant', 'linear', 'cosine', 'polynomial'],
        help='Central learning scheduler.')

    known_args, _ = argument_parser.parse_known_args()
    if known_args.central_lr_scheduler == 'cosine':
        argument_parser.add_argument(
            '--cosine_num_cycles',
            type=float,
            default=0.5,
            help='Number of waves in cosine learning rate scheduler.')
    if known_args.central_lr_scheduler == 'polynomial':
        argument_parser.add_argument(
            '--polynomial_power',
            type=float,
            default=1.0,
            help='Power in polynomial learning rate scheduler.')
    return argument_parser


def parse_central_lr_scheduler(arguments, optimizer):
    if arguments.central_lr_scheduler == 'constant':
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=arguments.central_lr_num_warmup_iterations)
    if arguments.central_lr_scheduler == 'linear':
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=arguments.central_lr_num_warmup_iterations,
            num_training_steps=arguments.central_num_iterations)
    if arguments.central_lr_scheduler == 'cosine':
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=arguments.central_lr_num_warmup_iterations,
            num_training_steps=arguments.central_num_iterations,
            num_cycles=arguments.cosine_num_cycles)
    if arguments.central_lr_scheduler == 'polynomial':
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=arguments.central_lr_num_warmup_iterations,
            num_training_steps=arguments.central_num_iterations,
            power=arguments.polynomial_power)
    return None


def add_llm_arguments(argument_parser):
    argument_parser = add_lm_arguments(argument_parser)
    argument_parser = add_peft_arguments(argument_parser)
    argument_parser = add_central_lr_scheduler_arguments(argument_parser)

    # Hugging Face model args
    argument_parser.add_argument('--hugging_face_model_name_or_path',
                                 type=str,
                                 help='Hugging Face model name or path.')

    argument_parser.add_argument('--model_max_length',
                                 type=int,
                                 default=512,
                                 help='Hugging Face model max length.')

    # Hugging Face tokenizer args
    argument_parser.add_argument(
        '--use_fast_tokenizer',
        action=store_bool,
        default=False,
        help='Set use_fast argument when loading Hugging Face tokenizer.')

    argument_parser.add_argument(
        '--padding_side',
        type=str,
        default=None,
        help='Set padding_side argument when loading Hugging Face tokenizer.')

    # LLM training args
    argument_parser.add_argument(
        '--amp_dtype',
        type=str,
        default='float32',
        choices=['float32', 'float16', 'bfloat16'],
        help='Float format in mixed precision training.')

    argument_parser.add_argument(
        '--grad_scaling',
        action=store_bool,
        default=False,
        help='Whether enable gradient scaling when using'
        ' mixed precision training.')

    argument_parser.add_argument(
        '--model_dtype_same_as_amp',
        action=store_bool,
        default=False,
        help='Cast the model weights precision to the same as used in '
        'autocast. This saves memory but may cause divergence due to '
        'lower precision.')

    argument_parser.add_argument(
        '--grad_accumulation_steps',
        type=int,
        default=1,
        help='Effective local batch size is local batch size '
        'multiplied by this number.')

    argument_parser.add_argument(
        '--use_torch_compile',
        action=store_bool,
        default=False,
        help='Whether to use `torch.compile` on the PyTorch module.')

    return argument_parser
