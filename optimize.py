from typing import Optional
from matplotlib import pyplot as plt
from scipy.optimize import least_squares, minimize
import torch
import tqdm

from data import Config
from models import varpro as vp
from models import create_function_example, UnrolledVP, FnSys, UVPRegressor


def ex_residual(function_system: FnSys, k: int, y: torch.Tensor, m: int, x: torch.Tensor) -> torch.Tensor:
    '''Compute the residual of the objective function at a given point.'''
    alpha, coeff = x[:k], x[k:]
    f = function_system(m, alpha)[0]
    f = f @ coeff - y
    return f


def indices_to_jac(function_system: FnSys, k: int, n: int, m: int, x: torch.Tensor) -> torch.Tensor:
    '''Convert our concise jacobian representation to a full jacobian matrix.'''
    alpha, coeff = x[:k], x[k:]
    f, dF, indices = function_system(m, alpha)
    J = torch.zeros((m, k+n))
    for i in range(k):
        param_mask = indices[1] == i
        param_indices = torch.arange(param_mask.shape[0])[param_mask]
        fun_indices = indices[0, param_mask]
        assert fun_indices.shape == param_indices.shape, 'The `indices` array does not represent the function properly'
        for j in range(fun_indices.shape[0]):
            J[:, i] += coeff[fun_indices[j]] * dF[:, param_indices[j]]
    for i in range(n):
        J[:, k+i] = f[:, i]
    return J


def scipy_solve(function_system: FnSys, k: int, n: int, y: torch.Tensor, m: int, x0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''Fit the function using a nonlinear least squares solver.'''
    fun = lambda x: ex_residual(function_system, k, y, m, torch.tensor(x, dtype=torch.float32)).numpy()
    jac = lambda x: indices_to_jac(function_system, k, n, m, torch.tensor(x, dtype=torch.float32)).numpy()
    opt_res = least_squares(fun, x0, jac=jac, method='lm')
    xk = torch.tensor(opt_res.x, dtype=torch.float32)
    alpha, coeff = xk[:k], xk[k:]
    return alpha, coeff


def varpro_solve(function_system: FnSys, y: torch.Tensor, m: int, alpha_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''Fit the function using the VP derivative and a nonlinear least squares solver.'''
    fun = lambda alpha: vp.var_pro_feval(function_system, torch.from_numpy(alpha), y, m)[0].numpy()
    jac = lambda alpha: vp.var_pro_jac(function_system, torch.from_numpy(alpha), y, m)[0].numpy()
    opt_res = least_squares(fun, alpha_init, jac=jac, method='lm')
    alpha = torch.from_numpy(opt_res.x)
    _, coeff = vp.var_pro_feval(function_system, alpha, y, m)
    return alpha, coeff


def varpro_min_solve(function_system: FnSys, y: torch.Tensor, m: int, alpha_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''Fit the function using the VP derivative but with a general purpose minimizer.'''
    obj_fun = lambda alpha: torch.square(vp.var_pro_feval(function_system, torch.from_numpy(alpha), y, m)[0]).sum().numpy()
    jac = lambda alpha: vp.var_pro_grad(function_system, torch.from_numpy(alpha), y, m).numpy()
    opt_res = minimize(obj_fun, alpha_init, method='BFGS', jac=jac)
    alpha = torch.from_numpy(opt_res.x)
    _, coeff = vp.var_pro_feval(function_system, alpha, y, m)
    return alpha, coeff


def unrolled_solve(conf: Config, function_system: FnSys, alpha_true: torch.Tensor, y: torch.Tensor, m: int,
                   alpha_init: torch.Tensor, num_epochs: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    '''Train an Unrolled VP model to fit the data.'''
    y = y.unsqueeze(0)
    model = UVPRegressor(UnrolledVP(conf, m, 7, function_system, alpha_init))
    print('No. params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    min_alpha: Optional[torch.Tensor] = None
    min_loss = 100
    pbar = tqdm.tqdm(range(num_epochs))
    for _ in pbar:
        optimizer.zero_grad()
        y_pred, alpha = model(y)
        loss = loss_fn(y_pred, y)
        #loss = loss_fn(alpha, alpha_true.unsqueeze(0))
        loss.backward()
        optimizer.step()
        pbar.set_description_str(f'loss: {loss.cpu().item():.3f}')
        if loss < min_loss:
            min_alpha, min_loss = alpha.detach(), loss
    
    if min_alpha is None or len(min_alpha) == 0:
        raise ValueError('No reasonable levels of loss were reached')
    _, coeff = vp.var_pro_feval(function_system, min_alpha[0], y[0], m)
    return min_alpha[0], coeff


def plot_results(ax: plt.Axes, label: str, function_system: FnSys, t: torch.Tensor, y: torch.Tensor, M: int, alpha: torch.Tensor, coeff: torch.Tensor) -> None:
    '''Plot the results.'''
    y_pred = function_system(M, alpha)[0] @ coeff
    print(f'Linear variables: {coeff}, Nonlinear variables: {alpha}')
    ax.plot(t.numpy(), y_pred.numpy())
    ax.scatter(t.numpy(), y.numpy(), s=12)
    ax.set_title(f'{label}, MSE: {torch.nn.functional.mse_loss(y_pred, y):.3f}')


def main() -> None:
    '''Run various algorithms on a function fitting problem.'''
    conf = Config.parse_args()
    torch.manual_seed(0)
    function_system, alpha_true, alpha_init, coeff_true = create_function_example('hermite')
    k = len(alpha_true)  # number of nonlinear parameters
    n = len(coeff_true)  # number of linear parameters
    print(f'True nonlinear variables: {alpha_true}')
    
    M = 100
    a_init = torch.ones(len(coeff_true))
    y_true = function_system(M, alpha_true)[0] @ coeff_true
    y = y_true + torch.randn(M) * 0.15
    t = torch.arange(y_true.shape[0])
    print('Data shape:', y.shape)

    axs: list[plt.Axes] = plt.subplots(1, 4, figsize=(15, 3))[1].tolist()

    # Scipy
    alpha, coeff = scipy_solve(function_system, k, n, y, M, torch.cat([alpha_init, a_init]))
    plot_results(axs[0], 'LM', function_system, t, y, M, alpha, coeff)

    # VarPro
    alpha, coeff = varpro_solve(function_system, y, M, alpha_init)
    plot_results(axs[1], 'VP+LM', function_system, t, y, M, alpha, coeff)

    # VarPro general minimization
    alpha, coeff = varpro_min_solve(function_system, y, M, alpha_init)
    plot_results(axs[2], 'VP+BFGS', function_system, t, y, M, alpha, coeff)

    # Unrolled VarPro
    alpha, coeff = unrolled_solve(conf, function_system, alpha_true, y, M, alpha_init)
    plot_results(axs[3], 'Unrolled VP', function_system, t, y, M, alpha, coeff)

    plt.show()


if __name__ == '__main__':
    main()
