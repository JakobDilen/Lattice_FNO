"""
This file runs the experiment comparing the original FNO to the hyperbolic cross lattice variations,
with a similar number of parameter.
"""

import os
import sys
import numpy as np
import torch

from utility.train_model import train_model
from utility.error_calculation import errorCalculationScript

from neuralop.layers.embeddings import GridEmbedding2D, LatticeEmbedding


# Dataset parameters
TRAIN_SEED = [503580726]
TEST_SEED = [869268807]
GENERALIZATION_SEED = [819144454]
RES = [32]
Z = torch.tensor([1, 721])
N = 1024

RES_GENERALIZATION_ERROR = 64
N_GENERALIZATION_ERROR = 4096

N_SAMPLES_TRAIN = 1024
N_SAMPLES_TEST = 256
N_SAMPLES_GENERALIZATION = 4096

# Model parameters
M_NORMAL = [3]
M_HYPERBOLIC_CROSS = [9]
N_LAYERS = [2]
N_CHANNELS = [16]

# Train parameters
BATCH_SIZE = 16
EPOCHS = 2000
INTERVAL = 2000
SCHEDULER_FACTOR = 0.8
SCHEDULER_STEP_SIZE = 50
INIT_LR = 0.01
VERBOSE = True
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def trainModels(
    res,
    iteration,
    n,
    z,
    M_normal,
    M_hyperbolic_cross,
    model_types,
    base_folder,
    data_folder,
    scheduler_step,
    scheduler_factor,
    init_lr,
    model_seeds,
):
    base_experiment_folder = f"{base_folder}/equal_parameter_analysis"
    data_experiment_folder = data_folder

    specific_experiment_folder = f"{base_experiment_folder}/res-{res}-n-{n}-z-{z[1:]}-batch-{BATCH_SIZE}-epochs-{EPOCHS}"

    normal_time_results = np.zeros((len(M_normal), 4))
    hyperbolic_cross_normal_time_results = np.zeros((len(M_hyperbolic_cross), 4))
    hyperbolic_cross_lattice_time_results = np.zeros((len(M_hyperbolic_cross), 4))

    print("Started training models.")

    for i in range(len(M_normal)):
        for n_layers in N_LAYERS:
            for n_channels in N_CHANNELS:

                if "normal" in model_types:
                    print("Training regular model: ")

                    experiment_folder = f"{specific_experiment_folder}-nodes-{M_normal[i]}-n_layers-{n_layers}-n_channels-{n_channels}"

                    model_stats = train_model(
                        model_id=iteration,
                        device=DEVICE,
                        save_dir=f"{experiment_folder}/normal-model_seed-{model_seeds[0]}",
                        data_dir=data_experiment_folder,
                        n_train=N_SAMPLES_TRAIN,
                        n_test=N_SAMPLES_TEST,
                        batch_size=BATCH_SIZE,
                        res=res,
                        M=M_normal[i],
                        train_seed=TRAIN_SEED[iteration],
                        test_seed=TEST_SEED[iteration],
                        n_layers=n_layers,
                        hidden_channels=n_channels,
                        lifting_channels_ratio=0,
                        projecting_channels_ratio=0,
                        epochs=EPOCHS,
                        interval=INTERVAL,
                        init_lr=init_lr,
                        weight_decay=1e-8,
                        positional_embedding=GridEmbedding2D(in_channels=1),
                        scheduler_parameters=[scheduler_factor, scheduler_step],
                        seed=model_seeds[0],
                        overwrite=True,
                        verbose=VERBOSE,
                    )

                    normal_time_results[i] = [
                        M_normal[i],
                        model_seeds[0],
                        model_stats[0],
                        model_stats[1],
                    ]

                    filename = (
                        specific_experiment_folder
                        + "-normal-logs-seed-"
                        + str(model_seeds[0])
                        + ".txt"
                    )
                    np.savetxt(filename, normal_time_results, fmt="%s")

                    filename = (
                        specific_experiment_folder
                        + "-normal-lr-seed-"
                        + str(model_seeds[0])
                        + ".txt"
                    )
                    np.savetxt(filename, model_stats[2], fmt="%s")

    for i in range(len(M_hyperbolic_cross)):
        for n_layers in N_LAYERS:
            for n_channels in N_CHANNELS:

                if "hyperbolic_cross_normal" in model_types:
                    print("Training hyperbolic_cross_normal model: ")

                    experiment_folder = f"{specific_experiment_folder}-nodes-{M_hyperbolic_cross[i]}-n_layers-{n_layers}-n_channels-{n_channels}"

                    model_stats = train_model(
                        model_id=iteration,
                        device=DEVICE,
                        save_dir=f"{experiment_folder}/hyperbolic_cross_normal-model_seed-{model_seeds[0]}",
                        data_dir=data_experiment_folder,
                        n_train=N_SAMPLES_TRAIN,
                        n_test=N_SAMPLES_TEST,
                        batch_size=BATCH_SIZE,
                        res=res,
                        M=M_normal[i],
                        train_seed=TRAIN_SEED[iteration],
                        test_seed=TEST_SEED[iteration],
                        n_layers=n_layers,
                        hidden_channels=n_channels,
                        lifting_channels_ratio=0,
                        projecting_channels_ratio=0,
                        epochs=EPOCHS,
                        interval=INTERVAL,
                        init_lr=init_lr,
                        weight_decay=1e-8,
                        positional_embedding=GridEmbedding2D(in_channels=1),
                        scheduler_parameters=[scheduler_factor, scheduler_step],
                        seed=model_seeds[0],
                        overwrite=True,
                        verbose=VERBOSE,
                        omega=1,
                    )
                    torch.cuda.empty_cache()

                    hyperbolic_cross_normal_time_results[i] = [
                        M_hyperbolic_cross[i],
                        model_seeds[1],
                        model_stats[0],
                        model_stats[1],
                    ]

                    filename = (
                        specific_experiment_folder
                        + "-hyperbolic_cross_normal-logs-seed-"
                        + str(model_seeds[1])
                        + ".txt"
                    )
                    np.savetxt(filename, hyperbolic_cross_normal_time_results, fmt="%s")

                    filename = (
                        specific_experiment_folder
                        + "-hyperbolic_cross_normal-lr-seed-"
                        + str(model_seeds[1])
                        + ".txt"
                    )
                    np.savetxt(filename, model_stats[2], fmt="%s")

                if "hyperbolic_cross_lattice" in model_types:
                    print("Training hyperbolic_cross_lattice model: ")

                    experiment_folder = f"{specific_experiment_folder}-nodes-{M_hyperbolic_cross[i]}-n_layers-{n_layers}-n_channels-{n_channels}"

                    model_stats = train_model(
                        model_id=iteration,
                        device=DEVICE,
                        save_dir=f"{experiment_folder}/hyperbolic_cross_lattice-model_seed-{model_seeds[1]}",
                        data_dir=data_experiment_folder,
                        n_train=N_SAMPLES_TRAIN,
                        n_test=N_SAMPLES_TEST,
                        batch_size=BATCH_SIZE,
                        res=n,
                        z=Z,
                        omega=1,
                        M=M_hyperbolic_cross[i],
                        train_seed=TRAIN_SEED[iteration],
                        test_seed=TEST_SEED[iteration],
                        n_layers=n_layers,
                        hidden_channels=n_channels,
                        lifting_channels_ratio=0,
                        projecting_channels_ratio=0,
                        epochs=EPOCHS,
                        interval=INTERVAL,
                        init_lr=init_lr,
                        weight_decay=1e-8,
                        positional_embedding=LatticeEmbedding(in_channels=1, z=z),
                        scheduler_parameters=[scheduler_factor, scheduler_step],
                        seed=model_seeds[0],
                        overwrite=True,
                        verbose=VERBOSE,
                    )
                    torch.cuda.empty_cache()

                    hyperbolic_cross_lattice_time_results[i] = [
                        M_hyperbolic_cross[i],
                        model_seeds[1],
                        model_stats[0],
                        model_stats[1],
                    ]

                    filename = (
                        specific_experiment_folder
                        + "-hyperbolic_cross_lattice-logs-seed-"
                        + str(model_seeds[1])
                        + ".txt"
                    )
                    np.savetxt(filename, hyperbolic_cross_lattice_time_results, fmt="%s")

                    filename = (
                        specific_experiment_folder
                        + "-hyperbolic_cross_lattice-lr-seed-"
                        + str(model_seeds[1])
                        + ".txt"
                    )
                    np.savetxt(filename, model_stats[2], fmt="%s")

def errorAnalysis(
    res,
    iteration,
    n,
    z,
    M_normal,
    M_hyperbolic_cross,
    model_types,
    base_folder,
    data_folder,
    model_seeds,
):
    base_experiment_folder = f"{base_folder}/equal_parameter_analysis"
    data_experiment_folder = data_folder

    specific_experiment_folder = f"{base_experiment_folder}/res-{res}-n-{n}-z-{z[1:]}-batch-{BATCH_SIZE}-epochs-{EPOCHS}"

    print("Started analysis.")
    for n_layers in N_LAYERS:
        for n_channels in N_CHANNELS:
            if "normal" in model_types:
                print("Calculate normal error: ")
                for M in M_normal:
                    print("#Modes: ", M)
                    experiment_folder = f"{specific_experiment_folder}-nodes-{M}-n_layers-{n_layers}-n_channels-{n_channels}"

                    errorCalculationScript(
                        res=res,
                        input_data_res=res,
                        eval_data_res=RES_GENERALIZATION_ERROR,
                        data_folder=data_experiment_folder,
                        save_path=f"{experiment_folder}/normal-model_seed-{model_seeds[0]}",
                        model_folder=f"{experiment_folder}/normal-model_seed-{model_seeds[0]}",
                        start_id=iteration,
                        stop_id=iteration + 1,
                        save_epochs=np.arange(0, EPOCHS + 1, INTERVAL),
                        train_seed=TRAIN_SEED[iteration],
                        test_seed=TEST_SEED[iteration],
                        generalization_seed=GENERALIZATION_SEED[iteration],
                        n_train=N_SAMPLES_TRAIN,
                        n_test=N_SAMPLES_TEST,
                        n_generalization=N_SAMPLES_GENERALIZATION,
                        device=DEVICE,
                        error_types=["generalization"],
                    )
                    torch.cuda.empty_cache()
            if "hyperbolic_cross_normal" in model_types:
                for M in M_hyperbolic_cross:
                    print("#Modes: ", M)
                    experiment_folder = f"{specific_experiment_folder}-nodes-{M}-n_layers-{n_layers}-n_channels-{n_channels}"

                    print("Calculate hyperbolic_cross error: ")
                    errorCalculationScript(
                        res=n,
                        input_data_res=n,
                        eval_data_res=N_GENERALIZATION_ERROR,
                        use_lattice=True,
                        data_folder=data_experiment_folder,
                        save_path=f"{experiment_folder}/hyperbolic_cross_lattice-model_seed-{model_seeds[0]}",
                        model_folder=f"{experiment_folder}/hyperbolic_cross_lattice-model_seed-{model_seeds[0]}",
                        start_id=iteration,
                        stop_id=iteration + 1,
                        save_epochs=np.arange(0, EPOCHS + 1, INTERVAL),
                        train_seed=TRAIN_SEED[iteration],
                        test_seed=TEST_SEED[iteration],
                        generalization_seed=GENERALIZATION_SEED[iteration],
                        n_train=N_SAMPLES_TRAIN,
                        n_test=N_SAMPLES_TEST,
                        n_generalization=N_SAMPLES_GENERALIZATION,
                        device=DEVICE,
                        error_types=["generalization"],
                    )
                    torch.cuda.empty_cache()

            if "hyperbolic_cross_lattice" in model_types:
                for M in M_hyperbolic_cross:
                    print("#Modes: ", M)
                    experiment_folder = f"{specific_experiment_folder}-nodes-{M}-n_layers-{n_layers}-n_channels-{n_channels}"

                    print("Calculate hyperbolic_cross error: ")
                    errorCalculationScript(
                        res=n,
                        input_data_res=n,
                        eval_data_res=N_GENERALIZATION_ERROR,
                        use_lattice=True,
                        data_folder=data_experiment_folder,
                        save_path=f"{experiment_folder}/hyperbolic_cross_normal-model_seed-{model_seeds[1]}",
                        model_folder=f"{experiment_folder}/hyperbolic_cross_normal-model_seed-{model_seeds[1]}",
                        start_id=iteration,
                        stop_id=iteration + 1,
                        save_epochs=np.arange(0, EPOCHS + 1, INTERVAL),
                        train_seed=TRAIN_SEED[iteration],
                        test_seed=TEST_SEED[iteration],
                        generalization_seed=GENERALIZATION_SEED[iteration],
                        n_train=N_SAMPLES_TRAIN,
                        n_test=N_SAMPLES_TEST,
                        n_generalization=N_SAMPLES_GENERALIZATION,
                        device=DEVICE,
                        error_types=["generalization"],
                    )
                    torch.cuda.empty_cache()


def main(
    train: str = "True",
    analyse: str = "True",
    iteration=0,
    model_types=None,
    base_folder=None,
    data_folder=None,
    scheduler_step=SCHEDULER_STEP_SIZE,
    scheduler_factor=SCHEDULER_FACTOR,
    init_lr=INIT_LR,
    model_seeds=[None, None, None],
):
    res = RES[0]
    n = N
    z = Z
    M_normal = M_NORMAL
    M_hyperbolic_cross = M_HYPERBOLIC_CROSS

    if model_types is None:
        model_types = ["normal", "hyperbolic_cross_normal", "hyperbolic_cross_lattice"]

    if base_folder is None:
        base_folder = "results"

    if not os.path.exists(base_folder):
        os.mkdir(base_folder)

    if data_folder is None:
        data_folder = base_folder

    elif not os.path.exists(data_folder):
        os.mkdir(data_folder)

    if train == "True":
        trainModels(
            res,
            iteration,
            z=z,
            n=n,
            M_normal=M_normal,
            M_hyperbolic_cross=M_hyperbolic_cross,
            model_types=model_types,
            base_folder=base_folder,
            data_folder=data_folder,
            scheduler_step=scheduler_step,
            scheduler_factor=scheduler_factor,
            init_lr=init_lr,
            model_seeds=model_seeds,
        )
    if analyse == "True":
        errorAnalysis(
            res,
            iteration,
            z=z,
            n=n,
            M_normal=M_normal,
            M_hyperbolic_cross=M_hyperbolic_cross,
            model_types=model_types,
            base_folder=base_folder,
            data_folder=data_folder,
            model_seeds=model_seeds,
        )


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if len(arguments) == 4:
        train, analyse, randomise, iteration = arguments[:]
    elif len(arguments) == 3:
        train, analyse, randomise = arguments[:]
        iteration = "0"
    else:
        train, analyse, randomise = (
            "True",
            "True",
            "False",
        )
        iteration = "0"
    if randomise == "True":
        TRAIN_SEED, TEST_SEED, GENERALISATION_SEED = np.random.randint(0, 1000000000, 3)
        if TRAIN_SEED == TEST_SEED:
            TEST_SEED += 1
        if TRAIN_SEED == GENERALISATION_SEED:
            GENERALISATION_SEED += 1
        if TEST_SEED == GENERALISATION_SEED:
            GENERALISATION_SEED += 1
        TRAIN_SEED = [TRAIN_SEED]
        TEST_SEED = [TEST_SEED]
        GENERALISATION_SEED = [GENERALISATION_SEED]
    main(
        train=train,
        analyse=analyse,
        iteration=int(iteration),
    )
