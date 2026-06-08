"""
Functions to easily train FNO's with specific characteristics.
"""

import time
import os

from neuralop.models.fno import FNO
from neuralop.training.trainer import Trainer
from neuralop.utils import count_model_params
from neuralop.data.transforms.data_processors import DefaultDataProcessor
from neuralop.layers.index_sets import HyperbolicCrossIndexSet, HyperRectangleIndexSet
from neuralop.layers.spectral_transforms import RegularGridFFT, Rank1LatticeFFT

from utility.loss_function import ComplexLpLoss

from .load_data import load_data
import torch
import numpy as np
import torch.nn.functional as F


def train_model(
    model_id="(-1)",
    device="cpu",
    save_dir="assets/models",
    data_dir="assets/datasets",
    n_train=1000,
    n_test=200,
    batch_size=32,
    res=512,
    z=None,
    M=5,
    omega=0,
    train_seed=0,
    test_seed=1,
    n_layers=4,
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    lifting_channels_ratio=2,
    projecting_channels_ratio=2,
    epochs=500,
    interval=100,
    init_lr=1e-3,
    weight_decay=1e-8,
    positional_embedding=None,
    scheduler_parameters=[0.8, 50],
    seed=0,
    n_dim=2,
    verbose=False,
    overwrite=False,
    scheduler_type="reduce",
    complex_data=True,
):
    """
    Wrapper function to used to train a FNO

    Parameters
    ----------
    model_id : int
        ID number of the models to differentiate between multiple iterations of the same model
    device : {"cpu", "cuda"}
        indicates which computational resource should be used for training
    save_dir : pathlike,
        save directory
    data_dir : pathlike,
        directory of the data
    n_train : int
        number of training samples
    n_test : int
        number of test samples
    batch_size : int
        batchsize used for training
    res : int
        resolution of the data
    z : torch.Tensor
        generating vector
    M : int
        M value used for index set
    alpha : float
        alpha value used for the index set
    train_seed : int
        random seed of the training seed
    test_seed : int
        random seed of the test seed
    n_layers : int
        number of layers of the model
    in_channels : int
        number of input channels
    out_channels : int
        number of output channels
    hidden_channels : int
        number of hidden channels
    lifting_channels_ratio : float
        number of lifting channels = hidden_channels * lifting_channels_ratio
    projecting_channels_ratio=2
        number of projecting channels = hidden_channels * projecting_channels_ratio
    epochs : int
        number of epochs to train
    interval : int
        interval at which the model is saved intermediate
    init_lr : float
        initial learning rate
    weight_decay : float
        regularization value
    positional_embedding : {LatticeEmbedding, GridEmbedding2D, None}
        which positional embedding to use (if None than no embedding is used)
    scheduler_parameters :(float, float),
        schedular parameters, i.e. step and factor value
    seed : int
        random seed
    n_dim : int
        number of dimensions
    verbose : bool
        indicates wether logs should be printed
    overwrite : bool
        indicates wether files should be overwritten if a certain model already exists
    scheduler_type : {"reduce", "linear"}
        indicates the type of scheduler used during training
    complex_data : bool
        indicates wether the input and output functions are complex valued
    """

    # Set-up paths
    save_folder = save_dir + "/models"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    models = f"{save_folder}/res-{res}-n_train-{n_train}-n_test-{n_test}-train_seed-{train_seed}-test_seed-{test_seed}"

    if z is not None:
        models += "-lattice"

    if not os.path.exists(models):
        os.mkdir(models)

    models += "/model_id-" + str(model_id)
    if not os.path.exists(models):
        os.mkdir(models)
    elif not overwrite:
        print("already trained.")
        return 0

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data and dataprocessor
    train_loader = load_data(
        n_samples=n_train,
        batch_size=batch_size,
        path=data_dir,
        type="train",
        res=res,
        use_lattice=(z is not None),
        seed=train_seed,
    )
    test_loader = load_data(
        n_samples=n_test,
        batch_size=batch_size,
        path=data_dir,
        type="test",
        res=res,
        use_lattice=(z is not None),
        seed=test_seed,
    )
    data_processor = data_processor = DefaultDataProcessor().to(device)
    # Define model.

    if omega == 0:
        index_set = HyperRectangleIndexSet(M + ((M % 2) / 2), n_dim)
        n_modes = [M * 2 + 1] * n_dim
    else:
        index_set = HyperbolicCrossIndexSet(M, n_dim, beta=float(2 * omega))
        n_modes = [M**(1/(2 * omega)) * 2 + 1] * 2

    if z is not None:

        model = FNO(
            n_modes=n_modes,
             in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            fno_skip="linear",
            use_channel_mlp=False,
            projection_channel_ratio=projecting_channels_ratio,
            lifting_channel_ratio=lifting_channels_ratio,
            non_linearity=F.gelu,
            positional_embedding=positional_embedding,
            complex_data=complex_data,
            index_set=index_set,
            spectral_transform=Rank1LatticeFFT(n=res, z=z, complex_data=complex_data)
        )

    else:
        model = FNO(
            n_modes=n_modes,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            fno_skip="linear",
            use_channel_mlp=False,
            projection_channel_ratio=projecting_channels_ratio,
            lifting_channel_ratio=lifting_channels_ratio,
            non_linearity=F.gelu,
            positional_embedding=positional_embedding,
            complex_data=complex_data,
            index_set=index_set,
            spectral_transform=RegularGridFFT(order=n_dim, complex_data=complex_data)
        )

    model = model.to(device)

    torch.save(model, models + "/model_trained-epochs-0.pt")

    n_params = count_model_params(model)
    if verbose:
        print(f"\nOur model has {n_params} parameters.")

    # Initialise trainer.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=init_lr, weight_decay=weight_decay
    )

    if scheduler_type == "linear":
        scheduler = torch.optim.lr_scheduler.StepLR(
                        optimizer=optimizer,
                        gamma=scheduler_parameters[0],
                        step_size=scheduler_parameters[1],
                )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=scheduler_parameters[0],
                patience=scheduler_parameters[1],
        )

    model_path = models + f"/model_trained-id-{model_id}"
    time_taken = 0

    learning_rates = np.zeros(epochs//interval)

    if complex_data:
        training_loss = ComplexLpLoss(d=2)
    else:
        training_loss = None

    for idx in range(epochs // interval):
        if idx == 0:
            trainer = Trainer(
                model=model,
                n_epochs=interval,
                device=device,
                data_processor=data_processor,
                eval_interval=interval,
                verbose=verbose,
            )
            # Train initial set of epochs
            start_time = time.time()
            trainer.train(
                train_loader=train_loader,
                test_loaders={res: test_loader},
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=model_path,
                training_loss=training_loss,
                save_every=interval,
            )
            time_taken += time.time() - start_time
            torch.save(model, models + f"/model_trained-epochs-{interval}.pt")

        else:
            # Train other epoch intervals
            trainer = Trainer(
                model=model,
                n_epochs=interval*(idx+1),
                device=device,
                data_processor=data_processor,
                eval_interval=interval,
                verbose=verbose,
            )
            start_time = time.time()
            trainer.train(
                train_loader=train_loader,
                test_loaders={res: test_loader},
                optimizer=optimizer,
                scheduler=scheduler,
                save_dir=model_path,
                save_every=interval,
                training_loss=training_loss,
                resume_from_dir=model_path,
            )
            time_taken += time.time() - start_time
            torch.save(model, models + f"/model_trained-epochs-{interval*(idx+1)}.pt")
        learning_rates[idx] = scheduler.get_last_lr()[-1]
    return time_taken, n_params, learning_rates
