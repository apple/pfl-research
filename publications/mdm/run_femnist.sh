#!/bin/bash

set -eoux

export PYTHONPATH=.

num_components=3
max_num_samples_mixture_component=450

python mdm_paper/training/train_femnist.py --num_mixture_components "$num_components" --cohort_size_init_algorithm 3400 --max_num_samples_mixture_component_init_algorithm "$max_num_samples_mixture_component" --central_num_iterations_init_algorithm 1 --cohort_size_algorithm 3400 --max_num_samples_mixture_component_algorithm "$max_num_samples_mixture_component" --central_num_iterations_algorithm 3 --data_dir data/femnist 
