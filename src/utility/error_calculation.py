"""
Script to calculate the errors of a trained model.
"""

import os
import numpy as np
import torch
import gc

from .load_data import load_data

from neuralop.data.transforms.data_processors import DefaultDataProcessor


def errorCalculationScript(
    res,
    input_data_res,
    eval_data_res=None,
    use_lattice=False,
    data_folder="",
    save_path="assets",
    model_folder="assets",
    start_id=0,
    stop_id=1,
    save_epochs=(0, 500),
    train_seed=0,
    test_seed=1,
    generalization_seed=2,
    n_train=1000,
    n_test=200,
    n_generalization=4000,
    error_types = ["train", "test", "generalization"],
    batch_size=None,
    device=None,
):
    """
    Calculates the errors for certain epochs-iterations of the specified model for the given datasets.


    Parameters:
    -----------
    res : int
        training resolution of the model
    input_res : int
            resolution of the input dataset (i.e. 'x' samples)
    eval_res : int
            resolution of the eval dataset (i.e. 'y' samples)
    use_lattice : bool
        wether the model is a lattice based or not
    data_folder : pathlike
        location of the database
    save_path : pathlike
        save location of error log
    model_folder : pathlike
        folder of the model
    start_id : int
        ID of the first model when averaging over multiple models
    stop_id : int
        ID of the last model when averaging over multiple models
    save_epochs : arraylike
        list of epochs of which the error should be calculated
    train_seed : int
        seed of train database
    test_seed : int
        seed of test database
    generalization_seed : int
        seed of generalization database
    n_train : int
        number of training samples
    n_test : int
        number of test samples
    n_generalization : int
        number of generalization samples
    error_types : list[str]
        the errors that are calculated i.e. train, test, generalization
    batch_size : int
        batchsize used during error calculation
    device : {"cpu", "cuda"}
        indicates which computational resource should be used for the error calculation
    """
    if device == None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    if type(save_epochs) == int:
        save_epochs = [save_epochs]

    if eval_data_res is None:
        eval_data_res = input_data_res

    # Set-up for training error
    if "train" in error_types:
        MAT = np.zeros(len(save_epochs))
        MRT = np.zeros(len(save_epochs))

        train_path = data_folder

    # Set-up for testing error
    if "test" in error_types:
        MAA = np.zeros(len(save_epochs))
        MRA = np.zeros(len(save_epochs))

        test_path = data_folder

    # Set-up for generalization error
    if "generalization" in error_types:
        MAG = np.zeros(len(save_epochs))
        MRG = np.zeros(len(save_epochs))

        generalization_path = data_folder

    if batch_size is None:
        batch_size = 512

    # Calculate norm
    if "generalization" in error_types:
        norm = calculateNorm(
            n_samples=n_generalization,
            path=data_folder,
            batch=batch_size,
            type="generalization",
            res=eval_data_res,
            use_lattice=use_lattice,
            seed=generalization_seed,
            device=device,
        )
    elif "train" in error_types:
        norm = calculateNorm(
            n_samples=n_train,
            path=data_folder,
            batch=batch_size,
            type="train",
            res=eval_data_res,
            use_lattice=use_lattice,
            seed=train_seed,
            device=device,
        )
    elif "test" in error_types:
        norm = calculateNorm(
            n_samples=n_test,
            path=data_folder,
            batch=batch_size,
            type="test",
            res=eval_data_res,
            use_lattice=use_lattice,
            seed=test_seed,
            device=device,
        )
    else:
        raise Exception("No valid error type given")

    for idx, epoch in enumerate(save_epochs):
        print(f"Analysing epoch {epoch} ({idx+1}/{len(save_epochs)})")
        mean_relative_approximation_error = 0
        mean_relative_generalization_error = 0
        mean_relative_training_error = 0
        mean_absolute_approximation_error = 0
        mean_absolute_generalization_error = 0
        mean_absolute_training_error = 0

        for i in range(start_id, stop_id):
            model = f"{model_folder}/models/res-{res}-n_train-{n_train}-n_test-{n_test}-train_seed-{train_seed}-test_seed-{test_seed}"

            if use_lattice:
                model += "-lattice"

            model += f"/model_id-{i}/model_trained-epochs-{epoch}.pt"

            # Calculate training errors
            if "train" in error_types:
                (relative_training_error, absolute_training_error) = calculateError(
                    n_samples=n_train,
                    model_path=model,
                    path=train_path,
                    batch=batch_size,
                    type="train",
                    input_res=input_data_res,
                    eval_res=eval_data_res,
                    use_lattice=use_lattice,
                    seed=train_seed,
                    numerator=norm,
                    device=device,
                )
                mean_relative_training_error += relative_training_error
                mean_absolute_training_error += absolute_training_error

            # Calculate testing errors
            if "test" in error_types:
                relative_approximation_error, absolute_approximation_error = (
                    calculateError(
                        n_samples=n_test,
                        model_path=model,
                        path=test_path,
                        batch=batch_size,
                        type="test",
                        input_res=input_data_res,
                        eval_res=eval_data_res,
                        use_lattice=use_lattice,
                        seed=test_seed,
                        numerator=norm,
                        device=device,
                    )
                )
                mean_relative_approximation_error += relative_approximation_error
                mean_absolute_approximation_error += absolute_approximation_error

            # Calculate generalization errors
            if "generalization" in error_types:
                (
                    relative_generalization_error,
                    absolute_generalization_error,
                ) = calculateError(
                    n_samples=n_generalization,
                    model_path=model,
                    path=generalization_path,
                    batch=batch_size,
                    type="generalization",
                    input_res=input_data_res,
                    eval_res=eval_data_res,
                    use_lattice=use_lattice,
                    seed=generalization_seed,
                    numerator=norm,
                    device=device,
                )
                mean_relative_generalization_error += relative_generalization_error
                mean_absolute_generalization_error += absolute_generalization_error

        # Rescale training error means
        if "train" in error_types:
            mean_relative_training_error = mean_relative_training_error / (
                stop_id - start_id
            )
            mean_absolute_training_error = mean_absolute_training_error / (
                stop_id - start_id
            )
            print(f"MRT: {mean_relative_training_error}")
            print(f"MAT: {mean_absolute_training_error}")
            MAT[idx] = mean_absolute_training_error
            MRT[idx] = mean_relative_training_error

        # Rescale testing error means
        if "test" in error_types:
            mean_relative_approximation_error = mean_relative_approximation_error / (
                stop_id - start_id
            )
            mean_absolute_approximation_error = mean_absolute_approximation_error / (
                stop_id - start_id
            )
            print(f"MRA: {mean_relative_approximation_error}")
            print(f"MAA: {mean_absolute_approximation_error}")
            MAA[idx] = mean_absolute_approximation_error
            MRA[idx] = mean_relative_approximation_error

        # Rescale generalization error means
        if "generalization" in error_types:
            mean_relative_generalization_error = mean_relative_generalization_error / (
                stop_id - start_id
            )
            mean_absolute_generalization_error = mean_absolute_generalization_error / (
                stop_id - start_id
            )
            print(f"MRG: {mean_relative_generalization_error}")
            print(f"MAG: {mean_absolute_generalization_error}")
            MAG[idx] = mean_absolute_generalization_error
            MRG[idx] = mean_relative_generalization_error

    # Create necessary folders
    basename_folder = f"{save_path}/errorAnalysis"
    if not os.path.exists(basename_folder):
        os.makedirs(basename_folder)

    basename_folder += "/errorLog"
    if not os.path.exists(basename_folder):
        os.makedirs(basename_folder)
    for i in ["relative", "absolute"]:
        if "test" in error_types:
            basename = f"{basename_folder}/{i}_approximation_error-res-{res}-train_seed-{train_seed}-test_seed-{test_seed}-generalization_seed-{generalization_seed}"
            if use_lattice:
                basename += "-lattice"

            filename = f"{basename}-start_id-{start_id}-stop_id-{stop_id}.txt"

            if i == "relative":
                np.savetxt(filename, MRA)
            else:
                np.savetxt(filename, MAA)

        if "train" in error_types:
            basename = f"{basename_folder}/{i}_training_error-res-{res}-train_seed-{train_seed}-test_seed-{test_seed}-generalization_seed-{generalization_seed}"

            if use_lattice:
                basename += "-lattice"

            filename = f"{basename}-start_id-{start_id}-stop_id-{stop_id}.txt"

            if i == "relative":
                np.savetxt(filename, MRT)
            else:
                np.savetxt(filename, MAT)

        if "generalization" in error_types:
            basename = f"{basename_folder}/{i}_generalization_error-res-{res}-train_seed-{train_seed}-test_seed-{test_seed}-generalization_seed-{generalization_seed}"

            if use_lattice:
                basename += "-lattice"

            filename = f"{basename}-start_id-{start_id}-stop_id-{stop_id}.txt"

            if i == "relative":
                np.savetxt(filename, MRG)
            else:
                np.savetxt(filename, MAG)


def calculateError(
    n_samples: int,
    model_path: str,
    path: str,
    batch: int = 1,
    type: str = "generalization",
    input_res: int = 16,
    eval_res: int | None = None,
    use_lattice: bool = False,
    seed=0,
    numerator=1,
    device=None,
    ):
    """
    Calculates the errors for certain epochs-iterations of the specified model for the given datasets.


    Parameters:
    -----------
    n_samples : int
            number of samples
    model_path : pathlike
            location of the model
    path : pathlike
            location of the dataset
    batch_size : int
            batchsize used during error calculation
    type : {"train" ,"test", "generalization"}
            type of database
    input_res : int
            resolution of the input dataset (i.e. 'x' samples)
    eval_res : int
            resolution of the eval dataset (i.e. 'y' samples)
    use_lattice : bool
            wether the model is a lattice based or not
    seed : int
            random seed
    numerator : float
            numerator of the error term
    device : {"cpu", "cuda"}
            indicates which computational resource should be used for the error calculation
    """
    input_dataloader = load_data(
        n_samples=n_samples,
        batch_size=batch,
        path=path,
        type=type,
        res=input_res,
        use_lattice=use_lattice,
        seed=seed,
        shuffle=False
    )

    if eval_res is None:
        eval_dataloader = input_dataloader
    else:
        eval_dataloader = load_data(
            n_samples=n_samples,
            batch_size=batch,
            path=path,
            type=type,
            res=eval_res,
            use_lattice=use_lattice,
            seed=seed,
            shuffle=False
        )


    data_processor = DefaultDataProcessor()

    if device == None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model = torch.load(model_path, weights_only=False, map_location=device)
    model.eval()
    model.to(device)

    absolute_error_total = 0

    input_dataloader_iterator = iter(input_dataloader)
    eval_dataloader_iterator = iter(eval_dataloader)

    for idx in range(len(input_dataloader_iterator)):
        input_data = data_processor.preprocess(next(input_dataloader_iterator))
        eval_data = data_processor.preprocess(next(eval_dataloader_iterator))

        x = input_data["x"].to(device)
        y = eval_data["y"].to(device)

        output_shape = [None] * (model.n_layers-1)
        output_shape.append([*y.shape[2:]])

        with torch.no_grad():
            y_h = model(x, output_shape=output_shape)

        # Calculate error
        y_h = y_h.reshape((y.shape[0], torch.prod(torch.tensor(y_h.shape[2:]))))
        y = y.reshape(y_h.shape)
        batched_error = torch.norm(y_h - y, p=2, dim=-1)

        # Update numerator if necessary
        del x
        del y
        del y_h

        absolute_error_total += torch.sum(batched_error)  # Sum error over batchsize

        del batched_error
        del input_data
        del eval_data
        gc.collect()

    relative_error = absolute_error_total / n_samples / numerator
    absolute_error = absolute_error_total / n_samples

    return (
        relative_error.item(),
        absolute_error.item(),
    )


def calculateNorm(
    n_samples: int,
    path: str,
    batch: int = 1,
    type: str = "generalization",
    res: int = 16,
    use_lattice: bool = False,
    seed=0,
    device=None,
):
    """
    Calculates the norm of a given database.


    Parameters:
    -----------
    n_samples : int
            number of samples
    path : pathlike
            location of the dataset
    batch_size : int
            batchsize used during error calculation
    type : {"train" ,"test", "generalization"}
            type of database
    res : int
            resolution of the dataset
    use_lattice : bool
            wether the model is a lattice based or not
    seed : int
            random seed
    device : {"cpu", "cuda"}
            indicates which computational resource should be used for the error calculation
    """
    dataloader = load_data(
        n_samples=n_samples,
        batch_size=batch,
        path=path,
        type=type,
        res=res,
        use_lattice=use_lattice,
        seed=seed,
    )

    data_processor = DefaultDataProcessor()

    if device == None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    data_processor = DefaultDataProcessor()

    data_set_norm = 0

    for data in iter(dataloader):
        data = data_processor.preprocess(data)

        y = data["y"].to(device)

        y = y.reshape((y.shape[0], torch.prod(torch.tensor(y.shape[2:]))))
        batched_norm_y = torch.norm(y, p=2, dim=-1)

        data_set_norm += torch.sum(batched_norm_y)

        del y
        del batched_norm_y
        del data
        gc.collect()

    return data_set_norm / n_samples
