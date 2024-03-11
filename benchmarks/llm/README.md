# Federated Large Language Models (LLMs) with Hugging Face

## Setup environment

1. Same as the [default setup](../README.md).
2. Install Hugging Face modules via `pip install transformers peft datasets`

## Run benchmarks

Benchmarks are implemented for Hugging Face with PyTorch only. 
We included three example datasets available on Hugging Face Hub:
1. [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca): 52,000 generated instructions and demonstrations. Alpaca does not have user ID and thus the dataset is IID partitioned for FL benchmarks.
2. [Aya](https://huggingface.co/datasets/CohereForAI/aya_dataset): 204,000 multilingual instruction and completions with user partition.
3. [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst2): 129,000 conversation messages with user patition.

The commands to run the benchmarks on the three datasets are the same:
```
# alpaca, aya or oasst
dataset=alpaca
```

LLM benchmark no DP:
```
python -m llm.train --args_config llm/configs/{dataset}.yaml
```

LLM benchmark Central DP:
```
python -m llm.train --args_config llm/configs/{dataset}.yaml --central_privacy_mechanism gaussian_moments_accountant
```
