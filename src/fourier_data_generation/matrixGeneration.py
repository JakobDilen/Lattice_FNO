import torch


def createMatrixVectorSystem_grid(a_coeff, f_coeff, modes, M, alpha=0):
    """
    Creates the matrix 'A' and vector 'f' needed to solve the PDE using the Fourier coefficients
    in a batched manner, assuming the coefficients are given in a regular grid format.

    Parameters:
    -----------
    -a_coeff : torch.Tensor
        Fourier coefficients of a
    -f_coeff : torch.Tensor
        Fourier coefficients of f
    -modes : torch.Tensor
        corresponding modes
    -M : int
        the M of the index set
    -alpha : float
        the alpha of the hyperbolic cross (alpha=0 gives hyperrectangle)
    """
    batchsize, *dims = a_coeff.shape
    n_dims = len(dims)
    assert(len(modes.shape) == 2)
    assert(n_dims == modes.shape[1])
    assert(n_dims == len(f_coeff.shape))

    if modes.dtype != torch.int:
        modes = modes.to(dtype=torch.int)
        
    device = a_coeff.device
    if modes.device != device:
        modes = modes.to(device=device)
    if f_coeff.device != device:
        f_coeff = f_coeff.to(device=device)

    n_modes = len(modes)

    L = torch.repeat_interleave(modes.unsqueeze(0), n_modes, axis=0)
    H = torch.repeat_interleave(modes.unsqueeze(0), n_modes, axis=1).reshape(n_modes, n_modes, n_dims)

    # Generate A
    a_indices = (L - H)
    if alpha != 0:
        mask = abs(torch.prod(torch.max(abs(a_indices), torch.ones_like(a_indices))[:], dim=-1))**(2 * alpha) <= M
    else:
        mask = (torch.prod(a_indices[:] >= -M, dim=-1)) * (torch.prod(a_indices[:] <= M, dim=-1)) 
    a_indices_masked = torch.repeat_interleave(mask.unsqueeze(-1), n_dims, dim=-1) * a_indices

    A_shape = [n_modes] * n_dims
    A_shape.insert(0, batchsize)
    dtype = a_coeff.dtype
    if (dtype == torch.complex64 or dtype == torch.float32):
        A = torch.zeros(A_shape, device=device).to(torch.complex64)
    elif (dtype == torch.complex128 or dtype == torch.float64):
        A = torch.zeros(A_shape, device=device).to(torch.complex128)
    else:
        raise Exception("Wrong datatype for 'a_coeff'.")

    A += torch.repeat_interleave((torch.sum(L * H, axis=-1)).unsqueeze(0), batchsize, axis=0)
    A *= a_coeff[(slice(None), *(a_indices_masked[:].unbind(dim=-1)))]
    A -= a_coeff[(slice(None), *torch.Tensor([0] * n_dims).to(dtype=torch.int))].unsqueeze(-1).unsqueeze(-1) * torch.repeat_interleave(((~mask) * torch.sum(L * H, axis=-1)).unsqueeze(0), batchsize, axis=0)
    A *= 4 * torch.pi**2

    # Generate f
    f = f_coeff[L[0, :, 0], L[0, :, 1]]

    if batchsize == 1:
        f = f.unsqueeze(0)
    else:
        f = torch.repeat_interleave(f.unsqueeze(0), batchsize, axis=0)

    return A, f

def createMatrixVectorSystem_lattice(a_coeff, f_coeff, modes, n, z, M, alpha=0):
    """
    Creates the matrix 'A' and vector 'f' needed to solve the PDE using the Fourier coefficients
    in a batched manner, assuming the coefficients gained through applying the 1-dimensional FFT

    Parameters:
    -----------
    -a_coeff : torch.Tensor
        Fourier coefficients of a
    -f_coeff : torch.Tensor
        Fourier coefficients of f
    -modes : torch.Tensor
        corresponding modes
    -n : int
        number of lattice points
    -z : torch.Tensor
        generating vector
    -M : int
        the M of the index set
    -alpha : float
        the alpha of the hyperbolic cross (alpha=0 gives hyperrectangle)
    """
    assert(len(f_coeff.shape) == 1)
    assert(len(a_coeff.shape) == 2)
    assert(modes.shape[-1] == len(z))

    batchsize = a_coeff.shape[0]
    
    if modes.dtype != torch.int:
        modes = modes.to(dtype=torch.int)
        
    device = a_coeff.device
    if modes.device != device:
        modes = modes.to(device=device)
    if f_coeff.device != device:
        f_coeff = f_coeff.to(device=device)

    n_modes, n_dim = modes.shape

    L = torch.repeat_interleave(modes.unsqueeze(0), n_modes, axis=0)
    H = torch.repeat_interleave(modes.unsqueeze(0), n_modes, axis=1).reshape(n_modes, n_modes, 2)

    a_indices = (L - H)

    if alpha != 0:
        mask = abs(torch.prod(torch.max(abs(a_indices), torch.ones_like(a_indices))[:], dim=-1))**(2 * alpha) <= M
    else:
        mask = (torch.prod(a_indices[:] >= -M, dim=-1)) * (torch.prod(a_indices[:] <= M, dim=-1)) 
    a_indices_masked = torch.repeat_interleave(mask.unsqueeze(-1), n_dim, dim=-1) * a_indices
    a_indices_masked = (torch.linalg.matmul(a_indices_masked.to(dtype=torch.int32), z.to(dtype=torch.int32)) % n).to(dtype=torch.int)
    
    if (a_coeff.dtype == torch.complex64) or (a_coeff.dtype == torch.float32):
        A = torch.zeros((batchsize, n_modes, n_modes)).to(torch.complex64)
    elif (a_coeff.dtype == torch.complex128) or (a_coeff.dtype == torch.float64):
        A = torch.zeros((batchsize, n_modes, n_modes)).to(torch.complex128)   

    A += torch.repeat_interleave((torch.sum(L * H, axis=-1)).unsqueeze(0), batchsize, axis=0)
    A *= a_coeff[:, a_indices_masked[:, :]]
    A -= a_coeff[:, 0].unsqueeze(-1).unsqueeze(-1) * torch.repeat_interleave(((~mask) * torch.sum(L * H, axis=-1)).unsqueeze(0), batchsize, axis=0)
    A *= 4 * torch.pi**2

    f = f_coeff[(torch.linalg.matmul(L[0, :], z.to(dtype=torch.int32))%n).to(dtype=torch.int)]
    return A, f

def reconstructU_grid(u_coeff: torch.Tensor, modes: torch.Tensor, res: int):
    """
    Reconstructs 'u' on a regular grid with given resolution 
    using the provided Fourier coefficients and there position.

    Parameters:
    -----------
    -u_coeff : torch.Tensor
        Fourier coefficients of u
    -modes : torch.Tensor
        corresponding modes
    -res : int
        resolution of the grid
    """

    assert(len(modes.shape) == 2)
    assert((len(u_coeff.shape) == 1) or (len(u_coeff.shape) == 2))
    assert(u_coeff.shape[-1] == len(modes))

    # Set-up
    if modes.dtype != torch.int:
        modes = modes.to(dtype=int)
    device = u_coeff.device
    if modes.device != device:
        modes = modes.to(device=device)
    n_modes, n_dim = modes.shape
    fft_dim = tuple(torch.arange(-n_dim, 0, dtype=torch.int).numpy())

    # Decide batchsize
    if len(u_coeff.shape) == 1:
        u_coeff = u_coeff.unsqueeze(0)
        batchsize = 1
    else:
        batchsize = u_coeff.shape[0]

    dtype = u_coeff.dtype
    if (dtype == torch.complex64 or dtype == torch.float32):
        u_fft = torch.zeros((batchsize, res, res), device=device).to(dtype=torch.complex64)
    elif (dtype == torch.complex128 or dtype == torch.float64):
        u_fft = torch.zeros((batchsize, res, res), device=device).to(dtype=torch.complex128)
    else:
        raise Exception("Wrong datatype for 'u_coeff'.")

    # Asign coefficients
    u_fft[(slice(None), *modes.unbind(dim=-1))] = u_coeff

    return torch.fft.ifftn(u_fft, dim=fft_dim, norm="forward")

def reconstructU_lattice(u_coeff, modes, n, z):
    """
    Reconstructs 'u' on the given rank-1 lattice
    using the provided Fourier coefficients and there position.

    Parameters:
    -----------
    -u_coeff : torch.Tensor
        Fourier coefficients of u
    -modes : torch.Tensor
        corresponding modes
    -n : int
        number of lattice points
    -z : torch.Tensor
        generating vector
    """
    assert((len(u_coeff.shape) == 1) or (len(u_coeff.shape) == 2))
    assert(u_coeff.shape[-1] == len(modes))

    # Set-up
    if modes.dtype != torch.int:
        modes = modes.to(dtype=torch.int)
    device = u_coeff.device
    if modes.device != device:
        modes = modes.to(device=device)

    # Decide batchsize
    if len(u_coeff.shape) == 1:
        u_coef = u_coeff.unsqueeze(0)
        batchsize = 1
    else:
        batchsize = u_coeff.shape[0]

    dtype = u_coeff.dtype
    if (dtype == torch.complex64 or dtype == torch.float32):
        u_fft = torch.zeros((batchsize, n), device=device).to(dtype=torch.complex64)
    elif (dtype == torch.complex128 or dtype == torch.float64):
        u_fft = torch.zeros((batchsize, n), device=device).to(dtype=torch.complex128)
    else:
        raise Exception("Wrong datatype for 'u_coeff'.")

    lattice_coefficients = (torch.linalg.matmul(modes, z.to(dtype=torch.int)) % n).to(dtype=torch.int64)
    u_out_fft = torch.scatter(
        u_fft, 1, lattice_coefficients.repeat(batchsize, 1), u_coeff
    )

    u = torch.fft.ifftn(u_out_fft, dim=[-1], norm="forward")
    return u
