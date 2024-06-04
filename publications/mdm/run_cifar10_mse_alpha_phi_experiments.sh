#!/bin/bash

set -oeux

export PYTHONPATH=.

num_components='1'
type='easy'

if [ "$num_components" -eq 1 ] && [ "$type" == "easy" ]; then

    alphas='1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0'
    phi='1.0'
    user_dataset_length='20'

elif [ "$num_components" -eq 2 ] && [ "$type" == "easy" ]; then

    alphas='1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8'
    phi='0.5 0.5'
    user_dataset_length='20 20'

elif [ "$num_components" -eq 3 ] && [ "$type" == "easy" ]; then

    alphas='1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1'
    phi='0.334 0.333 0.333'
    user_dataset_length='20 20 20'

elif [ "$num_components" -eq 1 ] && [ "$type" == "medium" ]; then

    alphas='0.1 0.2 0.6 1.0 2.0 0.1 1.0 2.0 0.5 0.5'
    phi='1.0'
    user_dataset_length='20'

elif [ "$num_components" -eq 2 ] && [ "$type" == "medium" ]; then

    alphas='0.1 0.2 0.6 1.0 2.0 0.1 1.0 2.0 0.5 0.5 1.1 0.1 0.7 0.9 1.0 0.2 0.5 1.0 0.8 1.5'
    phi='0.4 0.6'
    user_dataset_length='20 20'

elif [ "$num_components" -eq 3 ] && [ "$type" == "medium" ]; then

    alphas='0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.1 0.2 0.6 1.0 2.0 0.1 1.0 2.0 0.5 0.5 1.1 0.1 0.7 0.9 1.0 0.2 0.5 1.0 0.8 1.5'
    phi='0.2 0.3 0.5'
    user_dataset_length='20 20 20'

elif [ "$num_components" -eq 1 ] && [ "$type" == "hard" ]; then

    alphas=0.1
    phi='1.0'
    user_dataset_length='20'

elif [ "$num_components" -eq 2 ] && [ "$type" == "hard" ]; then

    alphas='0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3'
    phi='0.1 0.9'
    user_dataset_length='20 20'

elif [ "$num_components" -eq 3 ] && [ "$type" == "hard" ]; then

    alphas='0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 0.3 1.1 0.1 0.7 0.9 1.0 0.2 0.5 1.0 0.8 1.5'
    phi='0.05 0.15 0.8'
    user_dataset_length='20 20 20'

else 

    echo 'fail'

fi

python3 publications/mdm/mdm_paper/training/train.py --num_mixture_components "$num_components" --component_mean_user_dataset_length "$user_dataset_length" --component_alphas "$alphas" --cohort_size_init_algorithm 1000 --central_num_iterations_init_algorithm 1 --max_num_samples_mixture_component_init_algorithm 40 --cohort_size_algorithm 1000 --central_num_iterations_algorithm 60 --component_phi "$phi" --data_dir data/cifar10 --dirname "$type"
