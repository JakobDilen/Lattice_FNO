"""
This file can be run to generate all the necessary datasets.
"""
import run_single_dataset_ensemble
import numpy as np
from generate_data import f_checker_board
import os
import torch
import datetime
import sys

def log_with_time(message: str):
    print(f"{datetime.datetime.now().strftime('%H:%M:%S')}: {message}")



SCRATCH = "/scratch/leuven/360/vsc36022/Lattice_FNO_Cluster"


N_list = [64, 128, 256, 512, 1024, 4096]
N_training_samples_list = [64, 128, 256, 512] # 1024 already calculated
Z = torch.as_tensor([1, 721], dtype=torch.int64)
M = 9
ALPHA = 1
XI = 0.2

base_data_folder = os.path.join(SCRATCH, "assets")
if not os.path.exists(base_data_folder):
    os.makedirs(base_data_folder, exist_ok=True)

N_max  = max(N_list)
N_list.pop(N_list.index(N_max))

for a_type in ["affine", "periodic"]:
    # Generate high resolution datasets
    print(f"Calculating {a_type} datasets with (approx) {N_max} points.\n")
    run_single_dataset_ensemble.main(
        res=int(np.round(np.sqrt(N_max))),
        n=N_max,
        z=Z,
        f=f_checker_board,
        M=M,
        alpha=ALPHA,
        xi=XI,
        a_type=a_type,
        base_folder=base_data_folder,
        n_generalization=4096,
        n_test=256,
        n_train=1024,
    )

    # Generate different resolution training and test datasets through subsampling
    for n in N_list:
        print(f"Calculating {a_type} datasets with (approx) {n} points.\n")
        run_single_dataset_ensemble.main(
            res=int(np.round(np.sqrt(n))),
            n=n,
            z=Z,
            f=f_checker_board,
            M=M,
            alpha=ALPHA,
            xi=XI,
            a_type=a_type,
            base_folder=base_data_folder,
            n_generalization=4096, 
            n_test=256,
            n_train=1024,
            subsample_n=4096,
            subsample_res=64,
        )

    # Generate different number of training samples through subsampling
    for n_samples in N_training_samples_list:
        print(f"Calculating {a_type} datasets with (approx) {n_samples} samples.\n")
        run_single_dataset_ensemble.main(
            res=int(np.round(np.sqrt(1024))),
            n=1024,
            z=Z,
            f=f_checker_board,
            M=M,
            alpha=ALPHA,
            xi=XI,
            a_type=a_type,
            base_folder=base_data_folder,
            n_generalization=4096, # No generalization set needed
            n_test=256,
            n_train=n_samples,
            subsample_n_samples=1024,
        )

print(f"Finished creating datasets.\n")
