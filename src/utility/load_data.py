import torch
import numpy as np

from neuralop.data.datasets.tensor_dataset import TensorDataset


def load_data(
    n_samples=1000,
    batch_size=32,
    path="assets/datasets/experiment",
    type="train",
    res: int = 16,
    use_lattice: bool = False,
    seed: int = 0,
    shuffle=True,
):
    """
    Loads the specified data.

    ---
    n_samples : int
       the amount of loaded train data pairs.
    batch_size : int
       the batch size of the loaded data.
    path : pathLike
       the base folder of the datasets (structure to use: 'path/').
    type of dataset : string
       one of the three types of datasets: {'train', 'test', 'generalization'}.
    res : int
       resolution of the train and test data.
    use_lattice: bool
       wether the dataset contains lattice data.
    seed : int
       seed used for random number generation of training set.
    shuffle : bool
       wether the resulting samples need to be shuffled or not.
    """
    # Extend path:
    if use_lattice:
        path = f"{path}/{type}/data-lattice-res-{res}-n-{n_samples}-seed-{seed}.pt"
    else:
        path = f"{path}/{type}/data-res-{res}-n-{n_samples}-seed-{seed}.pt"

    data = torch.load(path, weights_only=False)

    # Select & Clone data
    if data["x"].dtype == torch.complex64:
       x_data = data["x"][0:n_samples].unsqueeze(1).type(torch.complex64).clone()
       y_data = data["y"][0:n_samples].unsqueeze(1).type(torch.complex64).clone()
    else:
       x_data = data["x"][0:n_samples].unsqueeze(1).type(torch.float32).clone()
       y_data = data["y"][0:n_samples].unsqueeze(1).type(torch.float32).clone()

    dataset = TensorDataset(
        x_data,
        y_data,
    )

    del data
    del x_data
    del y_data

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    return loader
