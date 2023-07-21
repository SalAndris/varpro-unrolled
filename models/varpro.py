import torch

from .functions import FnSys


def var_pro_feval(function_system: FnSys, alpha: torch.Tensor, y: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    '''Compute the coefficients and the residual of the VP model at a given point.'''
    F = function_system(m, alpha)[0]
    coeff = torch.linalg.lstsq(F, y).solution
    y_hat = F @ coeff
    residual = y - y_hat
    return residual, coeff


def var_pro_jac(function_system: FnSys, alpha: torch.Tensor, y: torch.Tensor, m: int, reg: float = -1.0) -> tuple[torch.Tensor, torch.Tensor]:
    '''Compute the Jacobian and the residual of the VP model at a given point.'''
    assert alpha.ndim == y.ndim, f'The data and the nonlinear parameters have different number of dimensions: {y.ndim}, {alpha.ndim}'
    orig_dim = alpha.ndim
    alpha = alpha.unsqueeze(0) if orig_dim == 1 else alpha
    y = y.unsqueeze(0) if orig_dim == 1 else y
    F, dF, indices = function_system(m, alpha)
    batch_size = F.shape[0]
    k = alpha.shape[1]  # number of nonlinear variables
    n = F.shape[2]  # number of linear variables

    # Solve the linear least squares
    if reg > 0:
        F = F + torch.rand(F.shape, device=F.device) * reg
    coeff = torch.linalg.lstsq(F, y).solution  # shape: (batch_size, n)

    # Compute the residual
    y_hat = (F @ coeff.unsqueeze(-1)).squeeze(-1)
    residual = y - y_hat  # shape: (batch_size, m)

    # Compute the Jacobian
    dF_residual = (residual.unsqueeze(1) @ dF).squeeze(1)  # shape: (batch_size, dF.shape[2])
    A = torch.zeros((batch_size, m, k), device=F.device)
    D = torch.zeros((batch_size, n, k), device=F.device)
    for j in range(k):
        cols = indices[1] == j
        A[:, :, j] = (dF[:, :, cols] @ coeff[:, indices[0, cols]].unsqueeze(-1)).squeeze(-1)
        D[:, indices[0, cols], j] = dF_residual[:, cols]
    U, s, V = torch.linalg.svd(F)  # shapes: (batch_size, m, m), (batch_size, n), (batch_size, n, n)
    A = U[:, :, n:] @ (U[:, :, n:].permute(0, 2, 1) @ A)
    D = torch.diag_embed(1/s) @ V @ D  # shape: (batch_size, n, n_nonlin)
    B = U[:, :, :n] @ D
    J = -(A + B)  # shape: (batch_size, m, n_nonlin)
    if orig_dim == 1:
        J, residual = J[0], residual[0]
    return J, residual


def var_pro_grad(function_system: FnSys, alpha: torch.Tensor, y: torch.Tensor, m: int) -> torch.Tensor:
    '''Compute the gradient of the objective function for general purpose minimization.'''
    J, res = var_pro_jac(function_system, alpha, y, m)
    return res @ J
