"""
Credits to Nuyens D., Kuo F., Keller A.
"""

import numpy as np

DATA_PATH = "src/fourier_data_generation/embedded_lattice_files/"


def lattice_points_shift(fn, N, d, shift=None, skipdims=1, col=0):
    # Adaption of original code for easier integration.
    if (N - 1) & N != 0:
        N = 2 ** int(np.ceil((np.log2(N))))

    z = load_generating_vector(fn, d, skipdims, col)
    assert len(z) == d
    if shift is None:
        return np.mod(np.outer(np.arange(N, dtype=np.uint64), z), N) / N
    else:
        return np.mod(
            np.mod(np.outer(np.arange(N, dtype=np.uint64), z), N) / N + shift, 1
        )


def load_generating_vector(fn, d, skipdims, col):
    z = np.loadtxt(fn, max_rows=d, dtype=np.uint64, skiprows=skipdims, usecols=col)
    assert len(z) == d
    return z


def ex_invj2_base2_m20_d3600(N, d, shift=None):
    return lattice_points_shift(
        DATA_PATH + "lattice-39102-1024-1048576.3600", N, d, shift, col=1
    )


def rand_shift(d):
    return np.random.rand(1, d)
