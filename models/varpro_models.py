from typing import Callable, Optional
import torch
import torch.nn.functional as F

from .varpro import var_pro_jac
from .functions import FnSys, hermite_system
from data import Config


def get_vp_model(conf: Config, n_classes: int, n_samples: int) -> torch.nn.Module:
    '''Instantiate an Unrolled VarPro classifier.'''
    assert n_samples > conf.vp_n_coeffs, 'Some operations cannot operate with more coefficients than data points'
    function_system = lambda m, alpha: hermite_system(m, alpha, n_functions=conf.vp_n_coeffs)
    alpha_init = torch.tensor(conf.vp_init, device=conf.device)
    bound_fn = lambda x: bound(x, torch.tensor([0.1, -0.4]), torch.tensor([0.7, 0.4]))
    unrolled_vp = UnrolledVP(conf, n_samples, conf.vp_depth, function_system, alpha_init, bound_fn)
    model = UVPClassifier(conf, n_classes, conf.vp_n_coeffs, unrolled_vp)
    model = model.to(conf.device)
    return model


def get_vpnet_model(conf: Config, n_classes: int, n_samples: int) -> torch.nn.Module:
    '''Instantiate a VPNet classifier.'''
    assert n_samples > conf.vp_n_coeffs, 'Some operations cannot operate with more coefficients than data points'
    function_system = lambda m, alpha: hermite_system(m, alpha, n_functions=conf.vp_n_coeffs)
    alpha_init = torch.tensor(conf.vp_init, device=conf.device)
    bound_fn = lambda x: bound(x, torch.tensor([0.1, -0.4]), torch.tensor([0.7, 0.4]))
    model = VPNet(conf, n_classes, conf.vp_n_coeffs, n_samples, alpha_init, function_system, bound_fn)
    model = model.to(conf.device)
    return model


def clip_norm(x: torch.Tensor) -> torch.Tensor:
    '''Clip norm to 1 if it is larger than 1. Input is assumed to have shape (batch_size, n).'''
    norm = torch.norm(x, dim=1).unsqueeze(1)
    normed = x / norm
    x = torch.where(norm < 1.0, x, normed)
    return x


def create_eye_lin(in_f: int, out_f: int, add_noise: bool = True) -> torch.nn.Linear:
    '''Create Linear layer initialized with identity weight matrix and zero bias.'''
    lin = torch.nn.Linear(in_f, out_f)
    lin.bias.data.copy_(torch.zeros(out_f, requires_grad=True))
    W = torch.eye(out_f, in_f, requires_grad=True)
    if add_noise:
        W = W + torch.eye(out_f, in_f) * torch.randn(in_f) * 0.01
    lin.weight.data.copy_(W)
    return lin


def bound(alpha: torch.Tensor, lower: Optional[torch.Tensor] = None, upper: Optional[torch.Tensor] = None) -> torch.Tensor:
    '''Bound the parameters to the desired range. Input is assumed to have shape
    (batch_size, n_params), lower and upper are assumed to have shape (n_params,).'''
    alpha = torch.max(alpha, lower.unsqueeze(0)) if lower is not None else alpha
    alpha = torch.min(alpha, upper.unsqueeze(0)) if upper is not None else alpha
    return alpha


BoundFn = Callable[[torch.Tensor], torch.Tensor]


class VPLayer(torch.nn.Module):
    '''Varible Projection unfolded iteration layer.'''

    def __init__(self, conf: Config, m: int, function_system: FnSys, bound_fn: Optional[BoundFn] = None) -> None:
        super().__init__()
        self.conf = conf
        self.m = m
        self.function_system = function_system
        self.bound_fn = bound_fn
        self.delta = 0.001 if conf.vp_delta_fix else torch.nn.Parameter(torch.tensor(0.001, dtype=torch.float32), requires_grad=True)
        self.update = self._update if conf.vp_compute_grad else self._update_no_grad

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        alpha, grad = self.update(x, alpha)
        alpha_next = alpha + self.delta * grad
        return alpha_next
    
    def _update(self, x: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = self.bound_fn(alpha) if self.bound_fn is not None else alpha
        J, residual = var_pro_jac(self.function_system, alpha, x, self.m, reg=self.conf.vp_reg_factor)
        grad = (residual.unsqueeze(1) @ J).squeeze(dim=1)  # shape: (b, len(alpha))
        grad = clip_norm(grad)
        return alpha, grad
    
    def _update_no_grad(self, x: torch.Tensor, alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            alpha, grad = self._update(x, alpha)
        return alpha, grad


class UnrolledVP(torch.nn.Module):
    '''Variable Projection model with n iterations.'''

    def __init__(self, conf: Config, n_samples: int, vp_depth: int, function_system: FnSys,
                 alpha_init: torch.Tensor, bound_fn: Optional[BoundFn] = None) -> None:
        super().__init__()
        self.conf = conf
        self.n_samples = n_samples
        self.vp_depth = vp_depth
        self.function_system = function_system
        self.alpha_init = alpha_init
        self.bound_fn = bound_fn
        self.layers = torch.nn.ModuleList()
        for _ in range(self.vp_depth):
            self.layers.append(VPLayer(self.conf, self.n_samples, function_system, bound_fn=bound_fn))
            if self.conf.vp_linear_init:
                self.layers.append(create_eye_lin(len(alpha_init), len(alpha_init), add_noise=conf.vp_linear_init_noise))
            else:
                self.layers.append(torch.nn.Linear(len(alpha_init), len(alpha_init)))
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.tile(self.alpha_init, (x.shape[0], 1))
        for i in range(self.vp_depth):
            vp_layer, lin = self.layers[i*2 : (i+1)*2]
            alpha = vp_layer(x, alpha)
            if self.conf.vp_res:
                alpha = F.relu((lin(alpha) + alpha) / 2)
            else:
                alpha = F.relu(lin(alpha))

        alpha = self.bound_fn(alpha) if self.bound_fn is not None else alpha
        f = self.function_system(self.n_samples, alpha)[0]
        coeff = torch.linalg.lstsq(f, x).solution
        return alpha, coeff


class VPNet(torch.nn.Module):
    '''VPNet model without exact derivative. See for more details: http://128.84.21.203/abs/2006.15590.'''
    
    def __init__(self, conf: Config, n_classes: int, n_coeffs: int, n_samples: int, alpha_init: torch.Tensor,
                 function_system: FnSys, bound_fn: Optional[BoundFn] = None) -> None:
        super().__init__()
        self.conf = conf
        self.n_samples = n_samples
        self.function_system = function_system
        self.bound_fn = bound_fn
        self.alpha = torch.nn.Parameter(alpha_init, requires_grad=True)
        self.lin1 = torch.nn.Linear(n_coeffs, conf.vp_lin_weights)
        self.lin2 = torch.nn.Linear(conf.vp_lin_weights, n_classes)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.tile(self.alpha, (x.shape[0], 1))
        alpha = self.bound_fn(alpha) if self.bound_fn is not None else alpha
        f, _, _ = self.function_system(self.n_samples, alpha)
        coeff = torch.linalg.lstsq(f, x).solution
        x_fit = (f @ coeff.unsqueeze(-1)).squeeze(-1)
        y_pred = self.lin2(F.relu(self.lin1(coeff)))
        return y_pred, x_fit


class UVPClassifier(torch.nn.Module):
    '''Classifier model using the UnrolledVP model.'''

    def __init__(self, conf: Config, n_classes: int, n_coeffs: int, unrolled_vp: UnrolledVP) -> None:
        super().__init__()
        self.conf = conf
        self.unrolled_vp = unrolled_vp
        self.lin1 = torch.nn.Linear(n_coeffs, conf.vp_lin_weights)
        self.lin2 = torch.nn.Linear(conf.vp_lin_weights, n_classes)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha, coeff = self.unrolled_vp(x)
        f, _, _ = self.unrolled_vp.function_system(self.unrolled_vp.n_samples, alpha)
        x_fit = (f @ coeff.unsqueeze(-1)).squeeze(-1)
        y_pred = self.lin2(F.relu(self.lin1(coeff)))
        return y_pred, x_fit


class UVPRegressor(torch.nn.Module):
    '''Regression model using the UnrolledVP model.'''

    def __init__(self, unrolled_vp: UnrolledVP) -> None:
        super().__init__()
        self.unrolled_vp = unrolled_vp
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha, coeff = self.unrolled_vp(x)
        f, _, _ = self.unrolled_vp.function_system(self.unrolled_vp.n_samples, alpha)
        y_pred = (f @ coeff.unsqueeze(-1)).squeeze(-1)
        return y_pred, alpha
