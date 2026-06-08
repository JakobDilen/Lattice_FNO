import time_analysis
import os
import sys

# General settings
BASE_FOLDER = "assets"
MODELS = ["normal", "lattice", "hyperbolic_cross", "hyperbolic_cross_normal"]
MODEL_SEEDS = [
    [836749, 330296, 390034],
    [211961, 573564, 880315],
    [996457, 101187, 331431],
    [754507, 153491, 341192],
    [222821, 469231, 989424],
]

# Training parameters
SCHEDULER_STEP = 50
SCHEDULER_FACTOR = 0.8
INIT_LR = 0.01

# Dataset parameters
M = 9
ALPHA = 1
XI = 0.2


def main(base_folder=BASE_FOLDER, base_data_folder=BASE_FOLDER):
    for model_seeds in MODEL_SEEDS:
        data_folder = f"{base_data_folder}/datasets/data-m-{M}-alpha-{ALPHA}-xi-{XI}"

        if not os.path.exists(f"{base_folder}/results"):
            os.mkdir(f"{base_folder}/results")
        experiment_folder = f"{base_folder}/results/experiment_time_analysis-step-{SCHEDULER_STEP}-factor-{SCHEDULER_FACTOR}-lr-{INIT_LR}-m-{M}-alpha-{ALPHA}-xi-{XI}"

        print("TRAIN MODELS \n ")
        for model_type in MODELS:
            time_analysis.main(
                train="True",
                analyse="False",
                model_types=[model_type],
                base_folder=experiment_folder,
                data_folder=data_folder,
                scheduler_step=SCHEDULER_STEP,
                scheduler_factor=SCHEDULER_FACTOR,
                init_lr=INIT_LR,
                model_seeds=model_seeds,
            )
            print("\n")

        print("CALCULATE ERRORS \n")
        for model_type in MODELS:
            time_analysis.main(
                train="False",
                analyse="True",
                model_types=[model_type],
                base_folder=experiment_folder,
                data_folder=data_folder,
                model_seeds=model_seeds,
            )
            print("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
        main(base_folder=base_folder)
    else:
        main()
