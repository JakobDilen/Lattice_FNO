"""
Functions used for creating the data sets for the PDE case using the Fourier coefficients.
"""

import gc

from embedded_lattice_files.utility import ex_invj2_base2_m20_d3600, load_generating_vector, rand_shift
from matrixGeneration import createMatrixVectorSystem_grid, createMatrixVectorSystem_lattice, reconstructU_grid, reconstructU_lattice

import os
import torch
import numpy as np

def sigmoid(x, k=1, s=1, complex=False):
    if complex:
        return s * (1 / (1 + torch.exp(-k * x.real)) -0.5) + 1j * s * (1 / (1 + torch.exp(-k * x.imag)) -0.5)
    return s / (1 + torch.exp(-k * x)) -0.5

# Checker-board forcing term
def f_checker_board(x, y, k=3, omega=2, s=500, complex=False):
    if complex:
        result = torch.zeros(x.shape, dtype=torch.complex64)
    else:
        result = torch.zeros(x.shape, dtype=torch.float32)
    result -= 1 * (sigmoid(torch.sin(omega * torch.pi * (x)), k=k) + sigmoid(torch.sin(omega * torch.pi * (y)), k=k))
    if complex:
        result += 1j * (sigmoid(torch.sin(omega * torch.pi * (x)), k=k) + sigmoid(torch.sin(omega * torch.pi * (y)), k=k))
    return result * s

def dataGen(
    n_samples: int,
    res: int,
    M: int,
    alpha: float,
    n: int,
    z: torch.Tensor,
    xi: float,
    f = f_checker_board,
    M_calc: int = None,
    alpha_calc: float = 0, 
    batchsize=64,
    path: str = 'assets/datasets',
    seed: int = 0,
    complex: bool = False,
    midpoint=1,
    types = ["regular", "lattice"],
    subsample_res = 64,
    subsample_n = None,
    subsample_n_samples = None,
):
    """
    Actually calculation of the samples and generation of the specified dataset.

    Parameters:
    -----------
    n_samples : int
        number o samples
    res : int
        resolution of the dataset
    M : int
        M parameter of index set
    alpha : float
        alpha parameter of index set
    n : int
        number of lattice points
    z : torch.Tensor
        generating vector
    xi : float
        scaling constant of the random fields
    f : function
        forcing term
    M_calc : int 
        M parameter of index set used in matrix vector system
    alpha_calc : 
        alpha parameter of index set used in matrix vector system
    batchsize : int
        number of samples calculated at the same time
    path : pathlike
        save path
    seed : int
        random seed
    complex : bool
        indicates if a is complex
    types : list["regular", "lattice"]
        indicates which datasets to generate
    subsample_res : int
        the resolution to subsample from if possible
    subsample_n : int | None
        the number of lattice points to subsample from if possible
    subsample_n_samples : int | None
        the number of samples to subsample from if possible
    """
    # Path set-up
    if not os.path.exists(path):
        os.mkdir(path)

    if "regular" in types:
        generate_regular = True
    if "lattice" in types:
        generate_lattice = True

    filename = f"{path}/data-res-{res}-n-{n_samples}-seed-{seed}.pt"
    lattice_filename = f"{path}/data-lattice-res-{n}-n-{n_samples}-seed-{seed}.pt"
  
    if os.path.exists(filename):
        print("Regular databse already generated.")
        generate_regular = False

    if os.path.exists(lattice_filename):
        print("Lattice database already generated.")
        generate_lattice = False

    # Check if subsampling is possible
    if subsample_res is not None:
        subsample_in_x_filename = f"{path}/data-res-{subsample_res}-n-{n_samples}-seed-{seed}.pt"
        
        if generate_regular and ((os.path.exists(subsample_in_x_filename) and (subsample_res % res == 0))):
            print("Subsampling regular database in 'x'.")
            original_dataset = torch.load(subsample_in_x_filename)
            data = subsample_regular_dataset_in_x(original_dataset, res)
            torch.save(data, filename)
            generate_regular = False

        subsample_lattice_in_x_filename = f"{path}/data-lattice-res-{subsample_n}-n-{n_samples}-seed-{seed}.pt"

        if generate_lattice and (os.path.exists(subsample_lattice_in_x_filename)):
            print("Subsampling lattice database in 'x'.")
            original_dataset = torch.load(subsample_lattice_in_x_filename)
            lattice_data = subsample_lattice_dataset_in_x(original_dataset, n, z)
            torch.save(lattice_data, lattice_filename)
            generate_lattice = False

    if subsample_n is not None:
        subsample_in_y_filename = f"{path}/data-res-{res}-n-{subsample_n_samples}-seed-{seed}.pt"

        if  generate_regular and (os.path.exists(subsample_in_y_filename)):
            print("Subsampling regular database in 'y'.")
            original_dataset = torch.load(subsample_in_y_filename)
            data = subsample_dataset_in_y(original_dataset, n_samples, M, alpha, complex)
            torch.save(data, filename)
            generate_regular = False

        subsample_lattice_in_y_filename = f"{path}/data-lattice-res-{n}-n-{subsample_n_samples}-seed-{seed}.pt"
    
        if generate_lattice and (os.path.exists(subsample_lattice_in_y_filename)):
            print("Subsampling lattice database in 'y'.")
            original_dataset = torch.load(subsample_lattice_in_y_filename)
            lattice_data = subsample_dataset_in_y(original_dataset, n_samples, M, alpha, complex)
            torch.save(lattice_data, lattice_filename)
            generate_lattice = False

    if generate_regular or generate_lattice:
        if M_calc is None:
            M_calc = (res - 1) // 2 
        elif M_calc > (res - 1) // 2:
            M_calc = (res - 1) // 2 

        if complex:
            data_type = torch.complex64
            grid_data_type = torch.float32
        else:
            data_type = torch.float32
            grid_data_type = data_type

        # Set seed
        np.random.seed(seed)

        if generate_regular:
            # Generate meshgrid
            X1_reg, X2_reg = torch.meshgrid(
                torch.as_tensor(np.linspace(0, 1, res, endpoint=False), dtype=grid_data_type), torch.as_tensor(np.linspace(0, 1, res, endpoint=False), dtype=grid_data_type), indexing="xy"
            )

            # Allocate datasets
            data = dict()
            x = torch.zeros([n_samples, res, res], dtype=data_type)
            y = torch.zeros([n_samples, res, res], dtype=data_type)

            # Define f
            f_eval_reg = f(X1_reg, X2_reg, complex)
            f_coeff_reg = torch.fft.fft2(f_eval_reg, norm="forward")

        if generate_lattice:
            # Generate lattice point set
            X1_lattice = ((torch.arange(0, n, dtype=grid_data_type) * z[0]) % n) / n
            X2_lattice = ((torch.arange(0, n, dtype=grid_data_type) * z[1]) % n) / n

            # Allocate datasets
            lattice_data = dict()
            x_lattice = torch.zeros([n_samples, n], dtype=data_type)
            y_lattice = torch.zeros([n_samples, n], dtype=data_type)

            # Define f
            f_eval_lattice = f(X1_lattice, X2_lattice, complex)
            f_coeff_lattice = torch.fft.fft(f_eval_lattice, norm="forward")

        # Index set
        h_r_pairs = createIndexSet(M=M, alpha=alpha, complex=complex)
        s = len(h_r_pairs)
        if complex:
            s *= 2

        # Random coefficients using embedded lattice
        random_coefficients = torch.Tensor(np.reshape(
            ex_invj2_base2_m20_d3600(N=n_samples, d=s, shift=rand_shift(d=s))[0:n_samples],
            (n_samples, s)
        ))

        # Apply correct transformation on random coefficients
        random_coefficients -= 0.5

        # Calculation modes
        i = torch.arange(-M_calc, M_calc + 1, dtype=torch.int32)
        grids = torch.meshgrid([i] * 2, indexing="ij")
        modes = torch.stack(grids, dim=-1).reshape(-1, 2)

        # Remove [0, 0]
        indices = list(range(0, (2 * M_calc + 1)**2))
        indices.pop(M_calc * (2 * M_calc+1) + M_calc)
        modes = modes[indices]

        h_modes = torch.as_tensor(list(map(lambda x: x[0], h_r_pairs)))
        if generate_lattice:
            h_modes_lattice = torch.inner(h_modes, z.flip(dims=[0])) % n
        
        r_values = torch.as_tensor(list(map(lambda x: x[1], h_r_pairs)))

        # Actual data generation
        iteration = 0
        while n_samples > 0:
            if n_samples < batchsize:
                batchsize = n_samples
            n_samples -= batchsize

            # Define a
            if generate_regular:
                a_coeff_grid = torch.zeros(batchsize, res, res, dtype=data_type)
                a_coeff_grid[:, 0, 0] = 1 * midpoint
                
            
            if generate_lattice:
                a_coeff_lattice = torch.zeros(batchsize, n, dtype=data_type)
                a_coeff_lattice[:, 0] = 1 * midpoint
                
            if complex:
                coeff_vector_real= (random_coefficients[iteration * batchsize:(iteration + 1) * batchsize, ::2])
                coeff_vector_complex = (random_coefficients[iteration * batchsize:(iteration + 1) * batchsize, 1::2])
                coeff_vector = coeff_vector_real + 1j * coeff_vector_complex
            else:
                coeff_vector = (random_coefficients[iteration * batchsize:(iteration + 1) * batchsize, :])

            scaled_coeff_vector = xi * coeff_vector / r_values

            if generate_regular:
                # Grid calculation
                a_coeff_grid[:, h_modes[:, 1], h_modes[:, 0]] += scaled_coeff_vector
                x[(batchsize * iteration):(batchsize * (iteration + 1))] = torch.fft.ifftn(a_coeff_grid, dim=[1, 2], norm="forward")
                
                A_reg, f_reg = createMatrixVectorSystem_grid(a_coeff_grid, f_coeff_reg, modes, M_calc, alpha_calc)
                u_coeff_reg = torch.linalg.solve(A_reg, f_reg)

                del A_reg, f_reg
                gc.collect()

                if complex:
                    y[(batchsize * iteration):(batchsize * (iteration + 1))] = reconstructU_grid(u_coeff_reg, modes, res)
                else:
                    y[(batchsize * iteration):(batchsize * (iteration + 1))] = reconstructU_grid(u_coeff_reg, modes, res).real

            if generate_lattice:
                # Lattice calculation
                a_coeff_lattice[:, h_modes_lattice] += scaled_coeff_vector
                x_lattice[(batchsize * iteration):(batchsize * (iteration + 1))] = torch.fft.ifftn(a_coeff_lattice, dim=[-1], norm="forward")
            
                A_lattice, f_lattice = createMatrixVectorSystem_lattice(a_coeff_lattice, f_coeff_lattice, modes, n, z, M_calc, alpha_calc)
                u_coeff_lattice = torch.linalg.solve(A_lattice, f_lattice)

                del A_lattice, f_lattice
                gc.collect()

                if complex:
                    y_lattice[(batchsize * iteration):(batchsize * (iteration + 1))] = reconstructU_lattice(u_coeff_lattice, modes, n, z)
                else:
                    y_lattice[(batchsize * iteration):(batchsize * (iteration + 1))] = reconstructU_lattice(u_coeff_lattice, modes, n, z).real
            
            iteration += 1

        if generate_regular:
            data["x"] = x
            data["y"] = y

            torch.save(data, filename)
            
        if generate_lattice:
            lattice_data["x"] = x_lattice
            lattice_data["y"] = y_lattice

            torch.save(lattice_data, lattice_filename)

def createIndexSet(M: int, alpha: float, complex: bool = False):
    """
    Creates the index set for a given index bound M and a smoothness parameter alpha.

    Parameters:
    -----------
    M : int
            M value of index set
    alpha : float
            alpha value of index set
    complex : bool
            indicates if all or only half coefficients have to be kept,
            due to the complex conjugate nature of Fourier coefficients of real functions
    """
    h_r_pairs = []
    h_i_max = int(np.floor(M ** (1 / (2 * alpha))))
    for i in range(-h_i_max, h_i_max + 1):
        start = 0
        if complex:
            start = -h_i_max
        for j in range(start, h_i_max + 1):
            if complex or (not (i < 0 and j == 0)):
                r_i = (max(abs(i), 1) * max(abs(j), 1)) ** (2 * alpha)
                if r_i <= M:
                    h_i = [i, j]
                    h_r_pairs.append((h_i, r_i))
    h_r_pairs.sort(key=lambda pair: pair[1])
    return h_r_pairs


def subsample_lattice_dataset_in_x(original_dataset, n_new, z):
    """
    Subsamples the given dataset to a lower number of lattice points. 
    Note that 'z' is assumed to be an embedded generating vector for both the 'n_original' and 'n_new'.
    """
    _, n_original = original_dataset["x"].shape
    lattice_original = ((torch.outer(torch.arange(0, n_original), z)) % n_original) / n_original
    lattice_new = ((torch.outer(torch.arange(0, n_new), z)) % n_new) / n_new

    indices = torch.where(torch.isin(lattice_original, lattice_new)[:, 0])[0]
    new_dataset = dict()
    new_dataset["x"] = original_dataset["x"][:, indices]
    new_dataset["y"] = original_dataset["y"][:, indices]
    
    return new_dataset

def subsample_regular_dataset_in_x(original_dataset, res_new):
    """
    Subsamples the given dataset to a lower number of lattice points. 
    Note that 'res_new' is assumed to be able to divide 'res_original'.
    """
    _, res_original, _ = original_dataset["x"].shape

    factor = int((res_original // res_new))

    new_dataset = dict()
    new_dataset["x"] = original_dataset["x"][:, ::factor, ::factor]
    new_dataset["y"] = original_dataset["y"][:, ::factor, ::factor]
    
    return new_dataset


def subsample_dataset_in_y(original_dataset, n_samples_new, M, alpha, complex):
    """
    Subsamples the given dataset to a lower number of sample points. 
    Note that it is assumed that the original dataset is created using the same seed as the subsampled.
    """
    n_samples_original, *dims = original_dataset["x"].shape

    h_r_pairs = createIndexSet(M=M, alpha=alpha, complex=complex)
    s = len(h_r_pairs)
    if complex:
        s *= 2
    z = torch.as_tensor(load_generating_vector("lattice-39102-1024-1048576.3600", d=s, skipdims=1, col=1).astype(np.int64), dtype=torch.int64)

    lattice_large = ((torch.outer(torch.arange(0, n_samples_original), z)) % n_samples_original) / n_samples_original
    lattice_small = ((torch.outer(torch.arange(0, n_samples_new), z)) % n_samples_new) / n_samples_new

    indices = torch.where(torch.isin(lattice_large, lattice_small)[:, 0])[0]
    
    new_dataset = dict()
    new_dataset["x"] = original_dataset["x"][indices]
    new_dataset["y"] = original_dataset["y"][indices]
    
    return new_dataset


