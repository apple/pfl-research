#!/bin/bash

set -oeux

export PYTHONPATH=.

for num_components in 1 2 3; do
    max_num_samples_mixture_component=450
    filter_method='sample' # 'sample' 'index'
    sample_fraction=0.5
    include_sampled='True'
    start_idx=0
    end_idx=1302

    if [ "$filter_method" == "sample" ]; then
      cohort_size=$(echo "scale=0; ($sample_fraction * 3400)" | bc | cut -d'.' -f1)
    else
      cohort_size=$(echo "$end_idx - $start_idx" | bc)
    fi

    cohort_size_init_algorithm=$cohort_size
    central_num_iterations_init_algorithm=1

    cohort_size_algorithm=$cohort_size
    central_num_iterations_algorithm=50

    if [ "$filter_method" == "index" ]; then
      python3 publications/mdm/mdm_paper/training/train_femnist_rebuttal.py --num_mixture_components "$num_components" --cohort_size_init_algorithm "$cohort_size_init_algorithm" --max_num_samples_mixture_component_init_algorithm "$max_num_samples_mixture_component" --central_num_iterations_init_algorithm "$central_num_iterations_init_algorithm" --cohort_size_algorithm "$cohort_size_algorithm" --max_num_samples_mixture_component_algorithm "$max_num_samples_mixture_component" --central_num_iterations_algorithm "$central_num_iterations_algorithm" --data_dir data/femnist --filter_method "$filter_method" --start_idx "$start_idx" --end_idx "$end_idx"

    elif [ "$filter_method" == "sample" ]; then
      python3 publications/mdm/mdm_paper/training/train_femnist_rebuttal.py --num_mixture_components "$num_components" --cohort_size_init_algorithm "$cohort_size_init_algorithm" --max_num_samples_mixture_component_init_algorithm "$max_num_samples_mixture_component" --central_num_iterations_init_algorithm "$central_num_iterations_init_algorithm" --cohort_size_algorithm "$cohort_size_algorithm" --max_num_samples_mixture_component_algorithm "$max_num_samples_mixture_component" --central_num_iterations_algorithm "$central_num_iterations_algorithm" --data_dir data/femnist --filter_method "$filter_method" --sample_fraction "$sample_fraction" --include_sampled "$include_sampled"

    fi
done
