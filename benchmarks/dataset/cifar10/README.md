# CIFAR10 dataset

Original source from https://www.cs.toronto.edu/~kriz/cifar.html

## Downloading data

To download and preprocess dataset, move to root example directory and run:
```
python -m dataset.cifar10.download_preprocess --output_dir data
```

You can inspect dataset by doing:
```python
with open('data/cifar10_train.p', 'rb') as f:
    images, labels = pickle.load(f)
```
