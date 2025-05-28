# CIFAR10 benchmark for TensorFlow Federated

## Installation

```
git clone git@github.com:google-research/federated.git google-research-federated 
pip install tensorflow==2.11 tensorflow_federated==0.48.0 \
    tensorflow_datasets \
    tensorflow_addons \
    pandas -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

## Run

Make sure Google research repo is in `PYTHONPATH` and run main:

```
PYTHONPATH=./google-research-federated:.; python -m main --server_learning_rate 1.0 --cohort_size 50 --client_optimizer sgd --local_batch_size 10 --server_optimizer sgd --central_num_iterations 1500 --local_num_epochs 1 --client_learning_rate 0.1 --evaluation_frequency 10
```
