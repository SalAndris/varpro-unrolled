from typing import Callable
import torch


SysMats = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
FnSys = Callable[[int, torch.Tensor], SysMats]


def create_function_example(name: str) -> tuple[FnSys, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Create a function system with appropriate parameters.'''
    if name == 'sin':
        alpha_true = torch.tensor([2.5, torch.pi/2, -1.0, torch.pi/3], dtype=torch.float32)
        alpha_init = torch.ones(len(alpha_true), dtype=torch.float32) * 2
        coeff_true = torch.tensor([2, -1], dtype=torch.float32)
        function_system = sin_system
    elif name == 'exp':
        alpha_true = torch.tensor([0.5, 3, 6], dtype=torch.float32)
        alpha_init = torch.ones(len(alpha_true), dtype=torch.float32) * 4
        coeff_true = torch.tensor([-2, 1], dtype=torch.float32)
        function_system = exp_system
    elif name == 'hermite':
        alpha_true = torch.tensor([0.1, 0.15], dtype=torch.float32)
        alpha_init = torch.ones(len(alpha_true), dtype=torch.float32) * 0.3
        coeff_true = torch.tensor([1, -1, 0.5, -1, 2], dtype=torch.float32)
        function_system = lambda m, alpha: hermite_system(m, alpha, n_functions=len(coeff_true))
    else:
        raise ValueError(f'Unknown function system: "{name}".')
    return function_system, alpha_true, alpha_init, coeff_true


def sin_system(m: int, alpha: torch.Tensor) -> SysMats:
    '''Sin example function system.'''
    if alpha.ndim > 1 and alpha.shape[0] > 1:
        raise ValueError('Batch dimension greater than 1 not supported')
    orig_dim = alpha.ndim
    alpha = alpha.squeeze()
    t = torch.linspace(0, torch.pi, m)
    F = torch.zeros((len(t), 2))
    F[:, 0] = torch.sin(alpha[0] * t + alpha[1])
    F[:, 1] = torch.sin(alpha[2] * t + alpha[3])

    indices = torch.tensor([
        [0, 0, 1, 1],
        [0, 1, 2, 3]
    ], dtype=torch.int32)

    dF = torch.zeros((len(t), 4))
    dF[:, 0] = torch.cos(alpha[0] * t + alpha[1]) * t
    dF[:, 1] = torch.cos(alpha[0] * t + alpha[1])
    dF[:, 2] = torch.cos(alpha[2] * t + alpha[3]) * t
    dF[:, 3] = torch.cos(alpha[2] * t + alpha[3])
    if orig_dim > 1:
        F, dF = F.unsqueeze(0), dF.unsqueeze(0)
    return F, dF, indices


def exp_system(m: int, alpha: torch.Tensor) -> SysMats:
    '''Exponentional (and trig.) example function system.'''
    if alpha.ndim > 1 and alpha.shape[0] > 1:
        raise ValueError('Batch dimension greater than 1 not supported')
    orig_dim = alpha.ndim
    alpha = alpha.squeeze()
    t = torch.linspace(0, 1.5, m)
    F = torch.zeros((len(t), 2))
    F[:, 0] = torch.exp(-alpha[1] * t) * torch.cos(alpha[2] * t)
    F[:, 1] = torch.exp(-alpha[0] * t) * torch.cos(alpha[1] * t)

    indices = torch.tensor([
        [0, 0, 1, 1],
        [1, 2, 0, 1]
    ], dtype=torch.int32)

    dF = torch.zeros((len(t), 4))
    dF[:, 0] = -t * F[:, 0]
    dF[:, 1] = -t * torch.exp(-alpha[1] * t) * torch.sin(alpha[2] * t)
    dF[:, 2] = -t * F[:, 1]
    dF[:, 3] = -t * torch.exp(-alpha[0] * t) * torch.sin(alpha[1] * t)
    if orig_dim > 1:
        F, dF = F.unsqueeze(0), dF.unsqueeze(0)
    return F, dF, indices
    

def hermite_system(m: int, alpha: torch.Tensor, n_functions: int = 7) -> SysMats:
    '''Hermite function system.'''
    orig_dim = alpha.ndim
    alpha = alpha.unsqueeze(0) if orig_dim == 1 else alpha
    dilation, translation = alpha.unbind(dim=-1)
    dilation, translation = dilation.unsqueeze(-1), translation.unsqueeze(-1)
    device = alpha.device
    batch_size = dilation.shape[0]
    n = n_functions
    t = torch.arange(-(m // 2), m // 2 + 1, device=device).unsqueeze(0).expand(batch_size, m) if m % 2 else torch.arange(-(m / 2), m / 2, device=device).unsqueeze(0).expand(batch_size, m)
    x = dilation * (t - translation * m / 2)
    w = torch.exp(-0.5 * x ** 2)
    dw = -x * w
    pi_sqrt = torch.sqrt(torch.sqrt(torch.tensor(torch.pi)))
    
    # Phi, dPhi
    Phi = torch.zeros((batch_size, m, n), device=device)
    Phi[:, :, 0] = 1
    Phi[:, :, 1] = 2 * x
    for j in torch.arange(1, n - 1, device=device):
        Phi[:, :, j + 1] = 2 * (x * Phi[:, :, j].clone() - j * Phi[:, :, j - 1].clone())
    Phi[:, :, 0] = w * Phi[:, :, 0].clone() / pi_sqrt
    dPhi = torch.zeros((batch_size, m, 2 * n), device=device)
    dPhi[:, :, 0] = dw / pi_sqrt
    dPhi[:, :, 1] = dPhi[:, :, 0]
    f = torch.tensor(1, requires_grad=False, device=device).unsqueeze(0)
    for j in torch.arange(1, n):
        f *= j
        Phi[:, :, j] = w * Phi[:, :, j].clone() / torch.sqrt(2 ** j * f) / pi_sqrt
        dPhi[:, :, 2 * j] = torch.sqrt(2 * j) * Phi[:, :, j - 1].clone() - x * Phi[:, :, j].clone()
        dPhi[:, :, 2 * j + 1] = dPhi[:, :, 2 * j]
    dPhi[:, :, 0::2] = dPhi[:, :, 0::2].clone() * (t - translation * m / 2).unsqueeze(-1)
    dPhi[:, :, 1::2] = -dPhi[:, :, 1::2].clone() * dilation.unsqueeze(-1) * m / 2
    
    # ind
    ind = torch.zeros((2, 2 * n), dtype=torch.int64, device=device)
    ind[0, 0::2] = torch.arange(n, dtype=torch.int64, device=device)
    ind[0, 1::2] = torch.arange(n, dtype=torch.int64, device=device)
    ind[1, 0::2] = torch.zeros(n, dtype=torch.int64, device=device)
    ind[1, 1::2] = torch.ones(n, dtype=torch.int64, device=device)
    if orig_dim == 1:
        Phi, dPhi = Phi[0], dPhi[0]
    return Phi, dPhi, ind
