from generate_data import dataGen
import sys
import os

# Standard dataset information
N_TRAIN = 1024
N_TEST = 256
N_GENERALIZATION = 4096
COMPLEX = True

# Seeds
TRAIN_SEED = 503580726
TEST_SEED = 869268807
GENERALIZATION_SEED = 819144454

BASE_FOLDER = "assets"


def main(res, n, z, f, M, alpha, xi, M_calc=25, alpha_calc=0, n_train=0, n_test=0, n_generalization=0, subsample_n=0, subsample_res=0, subsample_n_samples=0, complex=COMPLEX, train_seed=TRAIN_SEED, test_seed=TEST_SEED, generalization_seed=GENERALIZATION_SEED, base_folder=BASE_FOLDER):
    # Path set-up
    data_folder = base_folder
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    data_folder += f"/datasets"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    data_folder += f"/data-m-{M}-alpha-{alpha}-xi-{xi}"
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    if n_train != 0:
        print("Generating training dataset.")
        dataGen(
            res=res,
            n_samples=n_train,
            M=M,
            alpha=alpha,
            xi=xi,
            path=f"{data_folder}/train",
            n=n,
            z=z,
            seed=train_seed,
            complex=complex,
            f=f,
            M_calc=M_calc,
            alpha_calc=alpha_calc,
            subsample_n=subsample_n,
            subsample_res=subsample_res,
            subsample_n_samples=subsample_n_samples,
        )
        print("Done \n")

    if n_test != 0:
        print("Generating test dataset")
        dataGen(
            res=res,
            n_samples=n_test,
            M=M,
            alpha=alpha,
            xi=xi,
            path=f"{data_folder}/test",
            n=n,
            z=z,
            seed=test_seed,
            complex=complex,
            f=f,
            M_calc=M_calc,
            alpha_calc=alpha_calc,
            subsample_n=subsample_n,
            subsample_res=subsample_res,
            subsample_n_samples=subsample_n_samples,
        )
        print("Done \n")

    if n_generalization != 0:
        print("Generating generalization dataset")
        dataGen(
            res=res,
            n_samples=n_generalization,
            M=M,
            alpha=alpha,
            xi=xi,
            path=f"{data_folder}/generalization",
            n=n,
            z=z,
            seed=generalization_seed,
            complex=complex,
            f=f,
            M_calc=M_calc,
            alpha_calc=alpha_calc,
            subsample_n=subsample_n,
            subsample_res=subsample_res,
            subsample_n_samples=subsample_n_samples,
        )
        print("Done \n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_folder = sys.arv[1]
        main(base_folder=base_folder)
    else:
        main()
