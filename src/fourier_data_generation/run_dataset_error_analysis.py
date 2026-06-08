from embedded_lattice_files.utility import ex_invj2_base2_m20_d3600, rand_shift
from generate_data import createIndexSet
from matrixGeneration import createMatrixVectorSystem_grid, reconstructU_grid
import torch
import numpy as np

def f(x, y):
    return 5 * torch.sin(2*torch.pi * x) * torch.sin(2*torch.pi * y)

def main():
    # Calculation information
    res_1 = 32
    res_2 = 128
    M_calc_1 = (res_1 - 1) // 2
    M_calc_2 = (res_2 - 1) // 2
    alpha_calc = 0
    batchsize = 4

    # Data information
    M = 10
    alpha = 0.5
    xi = 0.1

    
    h_r_pairs = createIndexSet(M=M, alpha=alpha)
    s = len(h_r_pairs)

    random_coefficients = torch.Tensor(np.reshape(
        ex_invj2_base2_m20_d3600(N=batchsize, d=s, shift=rand_shift(d=s)),
        (batchsize, s)
    ))

    # Calculation modes
    i = torch.arange(-M_calc_1, M_calc_1 + 1, dtype=torch.int32)
    grids = torch.meshgrid([i] * 2, indexing="ij")
    modes = torch.stack(grids, dim=-1).reshape(-1, 2)

    # Remove [0, 0]
    indices = list(range(0, (2 * M_calc_1 + 1)**2))
    indices.pop(M_calc_1 * (2 * M_calc_1+1) + M_calc_1)
    modes = modes[indices]

    # Calculate solution on original grid
    X1_1, X2_1 = torch.meshgrid(
        torch.Tensor(np.linspace(0, 1, res_1, endpoint=False)), torch.Tensor(np.linspace(0, 1, res_1, endpoint=False)), indexing="xy"
    )

    f_eval_reg = f(X1_1, X2_1)
    f_coeff_reg = torch.fft.fft2(f_eval_reg, norm="forward")

    a_eval_grid = torch.ones(batchsize, res_1, res_1)

    for i in range(len(h_r_pairs)):
        h = h_r_pairs[i][0]
        r = h_r_pairs[i][1]

        coeff_vector = (random_coefficients[:, i])

        temp = xi * 2 / r * torch.cos(2 * torch.pi * (X1_1 * h[0] + X2_1 * h[1]))
        a_eval_grid[:, :, :] += (
            torch.einsum("nm,b->bnm", temp, coeff_vector)
        )
    a_coeff_grid = torch.fft.fftn(a_eval_grid, dim=[1, 2], norm="forward")
    A_reg, f_reg = createMatrixVectorSystem_grid(a_coeff_grid, f_coeff_reg, modes, M_calc_1, alpha_calc)

    u_coeff_reg = torch.linalg.solve(A_reg, f_reg)
    u_1 = reconstructU_grid(u_coeff_reg, modes, res_1).real

    # Calculate solution on increased grid
    X1_2, X2_2 = torch.meshgrid(
        torch.Tensor(np.linspace(0, 1, res_2, endpoint=False)), torch.Tensor(np.linspace(0, 1, res_2, endpoint=False)), indexing="xy"
    )

    f_eval_reg = f(X1_2, X2_2)
    f_coeff_reg = torch.fft.fft2(f_eval_reg, norm="forward")

    a_eval_grid = torch.ones(batchsize, res_2, res_2)

    for i in range(len(h_r_pairs)):
        h = h_r_pairs[i][0]
        r = h_r_pairs[i][1]

        coeff_vector = (random_coefficients[:, i])

        temp = xi * 2 / r * torch.cos(2 * torch.pi * (X1_2 * h[0] + X2_2 * h[1]))
        a_eval_grid[:, :, :] += (
            torch.einsum("nm,b->bnm", temp, coeff_vector)
        )
    a_coeff_grid = torch.fft.fftn(a_eval_grid, dim=[1, 2], norm="forward")
    A_reg, f_reg = createMatrixVectorSystem_grid(a_coeff_grid, f_coeff_reg, modes, M_calc_2, alpha_calc)

    u_coeff_reg = torch.linalg.solve(A_reg, f_reg)
    u_2 = reconstructU_grid(u_coeff_reg, modes, res_2).real

    # subsample u_2
    u_2_res_1 = u_2[:, ::res_2//res_1, ::res_2//res_1]

    # calculate error
    error = torch.sum((u_2_res_1 - u_1)**2) / batchsize
    print(f"Attained error is {error.item()}.")







if __name__=="__main__":
    main()