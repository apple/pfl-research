"""Util functions for CIFAR10/100."""


from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from flwr.common.parameter import ndarrays_to_parameters
from flwr.common.typing import NDArrays, Parameters, Scalar
from flwr.server.history import History
from PIL import Image
from torch import Tensor, load
from torch.nn import GroupNorm, Module
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import ResNet, resnet18
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

from flwr_baselines.dataset.utils.common import (
    XY,
    create_lda_partitions,
)

from compare_utils.pytorch import simple_cnn


# transforms
def get_transforms(num_classes: int = 10) -> Dict[str, Compose]:
    """Returns the right Transform Compose for both train and evaluation.

    Args:
        num_classes (int, optional): Defines whether CIFAR10 or CIFAR100. Defaults to 10.

    Returns:
        Dict[str, Compose]: Dictionary with 'train' and 'test' keywords and Transforms
        for each
    """
    normalize_cifar = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    """
    train_transform = Compose(
        [RandomCrop(24), RandomHorizontalFlip(), ToTensor(), normalize_cifar]
    )
    test_transform = Compose([CenterCrop(24), ToTensor(), normalize_cifar])
    """
    train_transform = Compose([ToTensor(), normalize_cifar])
    test_transform = Compose([ToTensor(), normalize_cifar])
    return {"train": train_transform, "test": test_transform}


def get_cifar_model(num_classes: int = 10) -> Module:
    """Generates ResNet18 model using GroupNormalization rather than
    BatchNormalization. Two groups are used.

    Args:
        num_classes (int, optional): Number of classes {10,100}. Defaults to 10.

    Returns:
        Module: ResNet18 network.
    """
    return simple_cnn((32, 32, 3), num_classes, False)


class ClientDataset(Dataset):
    """Client Dataset."""

    def __init__(self, path_to_data: Path, transform: Compose = None):
        """Implements local dataset.

        Args:
            path_to_data (Path): Path to local '.pt' file is located.
            transform (Compose, optional): Transforms to be used when sampling.
            Defaults to None.
        """
        super().__init__()
        self.transform = transform
        self.inputs, self.labels = load(path_to_data)

    def __len__(self) -> int:
        """Size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        """Fetches item in dataset.

        Args:
            idx (int): Position of item being fetched.

        Returns:
            Tuple[Tensor, int]: Tensor image and respective label
        """
        this_input = Image.fromarray(self.inputs[idx])
        this_label = self.labels[idx]
        if self.transform:
            this_input = self.transform(this_input)

        return this_input, this_label


def save_partitions(
    list_partitions: List[XY], fed_dir: Path, partition_type: str = "train"
):
    """Saves partitions to individual files.

    Args:
        list_partitions (List[XY]): List of partitions to be saves
        fed_dir (Path): Root directory where to save partitions.
        partition_type (str, optional): Partition type ("train" or "test"). Defaults to "train".
    """
    for idx, partition in enumerate(list_partitions):
        path_dir = fed_dir / f"{idx}"
        path_dir.mkdir(exist_ok=True, parents=True)
        torch.save(partition, path_dir / f"{partition_type}.pt")


def partition_cifar10_and_save(
    dataset: XY,
    fed_dir: Path,
    dirichlet_dist: Optional[npt.NDArray[np.float32]] = None,
    num_partitions: int = 500,
    concentration: float = 0.1,
) -> np.ndarray:
    """Creates and saves partitions for CIFAR10.

    Args:
        dataset (XY): Original complete dataset.
        fed_dir (Path): Root directory where to save partitions.
        dirichlet_dist (Optional[npt.NDArray[np.float32]], optional):
            Pre-defined distributions to be used for sampling if exist. Defaults to None.
        num_partitions (int, optional): Number of partitions. Defaults to 500.
        concentration (float, optional): Alpha value for Dirichlet. Defaults to 0.1.

    Returns:
        np.ndarray: Generated dirichlet distributions.
    """
    # Create partitions
    # concentration=0.1, partitions=1000, will create datasets of 50 images each.
    clients_partitions, dist = create_lda_partitions(
        dataset=dataset,
        dirichlet_dist=dirichlet_dist,
        num_partitions=num_partitions,
        concentration=float("inf"),
    )
    # Save partions
    save_partitions(list_partitions=clients_partitions, fed_dir=fed_dir)

    return dist


def gen_cifar10_partitions(
    path_original_dataset: Path,
    dataset_name: str,
    num_total_clients: int,
) -> Path:
    """Defines root path for partitions and calls functions to create them.

    Args:
        path_original_dataset (Path): Path to original (unpartitioned) dataset.
        dataset_name (str): Friendly name to dataset.
        num_total_clients (int): Number of clients.
        distributions.

    Returns:
        Path: [description]
    """
    fed_dir = (
        path_original_dataset
        / f"{dataset_name}"
        / "partitions"
        / f"{num_total_clients}"
    )

    trainset = CIFAR10(root=path_original_dataset, train=True, download=True)
    flwr_trainset = (trainset.data, np.array(trainset.targets, dtype=np.int64))
    partition_cifar10_and_save(
        dataset=flwr_trainset,
        fed_dir=fed_dir,
        dirichlet_dist=None,
        num_partitions=num_total_clients,
    )

    return fed_dir


def train(
    net: Module,
    trainloader: DataLoader,
    epochs: int,
    device: str,
    learning_rate: float = 0.01,
) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net: Module, testloader: DataLoader, device: str) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def gen_on_fit_config_fn(
    epochs_per_round: int, batch_size: int, client_learning_rate: float
) -> Callable[[int], Dict[str, Scalar]]:
    """Generates ` On_fit_config`

    Args:
        epochs_per_round (int):  number of local epochs.
        batch_size (int): Batch size
        client_learning_rate (float): Learning rate of clinet

    Returns:
        Callable[[int], Dict[str, Scalar]]: Function to be called at the beginnig of each rounds.
    """

    def on_fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with specific client learning rate."""
        local_config: Dict[str, Scalar] = {
            "epoch_global": server_round,
            "epochs": epochs_per_round,
            "batch_size": batch_size,
            "client_learning_rate": client_learning_rate,
        }
        return local_config

    return on_fit_config


def get_cifar_eval_fn(
    path_original_dataset: Path, evaluation_frequency: int, num_classes: int = 10
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Returns an evaluation function for centralized evaluation."""
    CIFAR = CIFAR10 if num_classes == 10 else CIFAR100
    transforms = get_transforms(num_classes=num_classes)

    testset = CIFAR(
        root=path_original_dataset,
        train=False,
        download=True,
        transform=transforms["test"],
    )

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if server_round % evaluation_frequency != 0:
            return None
        # pylint: disable=unused-argument
        """Use the entire CIFAR-10 test set for evaluation."""
        # determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = get_cifar_model(num_classes=num_classes)
        state_dict = OrderedDict(
            {
                k: torch.tensor(np.atleast_1d(v))
                for k, v in zip(net.state_dict().keys(), parameters_ndarrays)
            }
        )
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=1000)
        loss, accuracy = test(net, testloader, device=device)
        # return statistics
        return loss, {"central accuracy": accuracy, "central loss": loss}

    return evaluate


def get_initial_parameters(num_classes: int = 10) -> Parameters:
    """Returns initial parameters from a model.

    Args:
        num_classes (int, optional): Defines if using CIFAR10 or 100. Defaults to 10.

    Returns:
        Parameters: Parameters to be sent back to the server.
    """
    model = get_cifar_model(num_classes)
    weights = [val.cpu().numpy() for _, val in model.state_dict().items()]
    parameters = ndarrays_to_parameters(weights)

    return parameters


def plot_metric_from_history(
    hist: History,
    dataset_name: str,
    strategy_name: str,
    expected_maximum: float,
    save_plot_path: Path,
) -> None:
    """Simple plotting method for Classification Task.

    Args:
        hist (History): Object containing evaluation for all rounds.
        dataset_name (str): Name of the dataset.
        strategy_name (str): Strategy being used
        expected_maximum (float): Expected final accuracy.
        save_plot_path (Path): Where to save the plot.
    """
    rounds, values = zip(*hist.metrics_centralized["accuracy"])
    plt.figure()
    plt.plot(rounds, np.asarray(values) * 100, label=strategy_name)  # Accuracy 0-100%
    # Set expected graph
    plt.axhline(y=expected_maximum, color="r", linestyle="--")
    plt.title(f"Centralized Validation - {dataset_name}")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(save_plot_path)
    plt.close()
