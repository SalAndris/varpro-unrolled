import argparse


class Config(argparse.Namespace):
    '''Configuration class for easy experimentation.'''
    data_type: str = 'syn'
    model_name: str = 'vp'
    device: str = 'cpu'
    epochs: int = 100
    optimizer: str = 'adam'
    lr: float = 0.01
    momentum: float = 0.95
    weight_decay: float = 1e-4
    vp_init: tuple[float, float] = (0.2, 0.0)
    vp_depth: int = 5
    vp_n_coeffs: int = 7
    vp_linear_init: bool = True
    vp_linear_init_noise: bool = True
    vp_delta_fix: bool = False
    vp_compute_grad: bool = True
    vp_reg_factor: float = 0.01
    vp_res: bool = False
    vp_lin_weights: int = 5
    vp_loss_factor: float = 1.0
    ff_weights: list[int] = []
    cnn_kernel: int = 5
    cnn_pool_stride: int = 2

    @classmethod
    def parse_args(cls) -> 'Config':
        '''Parse command line arguments.'''
        parser = argparse.ArgumentParser(description='Training hyperparameters')
        parser.add_argument('--data-type', default=cls.data_type, choices=['syn', 'ecg'], help='Dataset type.')
        parser.add_argument('--model-name', default=cls.model_name, choices=['vp', 'vpnet', 'ff', 'cnn'], help='Model name.')
        parser.add_argument('--device', default=cls.device, help='Device.')
        parser.add_argument('--epochs', default=cls.epochs, type=int, help='Number of training epochs.')
        parser.add_argument('--optimizer', default=cls.optimizer, choices=['adam', 'sgd'], help='Optimizer algorithm.')
        parser.add_argument('--lr', default=cls.lr, type=float, help='Learning rate.')
        parser.add_argument('--momentum', default=cls.momentum, type=float, help='SGD momentum.')
        parser.add_argument('--weight-decay', default=cls.weight_decay, type=float, help='Optimizer weight decay.')
        parser.add_argument('--vp-init', default=cls.vp_init, type=float, nargs=2, help='VPLayer initial nonlinear parameters.')
        parser.add_argument('--vp-depth', default=cls.vp_depth, type=int, help='Number of unfolded VP layers.')
        parser.add_argument('--vp-n-coeffs', default=cls.vp_n_coeffs, type=int, help='Number of basis functions to use in VP.')
        parser.add_argument('--vp-linear-init', default=cls.vp_linear_init, action='store_false', help='VPLayer linear initialization with identity.')
        parser.add_argument('--vp-linear-init-noise', default=cls.vp_linear_init_noise, action='store_false',
                            help='If the VPLayer is initialized with identity, optionally some noise can be added to it.')
        parser.add_argument('--vp-delta-fix', default=cls.vp_delta_fix, action='store_true',
                            help='Set the internal VP learning rate parameter fixed instead of learnable.')
        parser.add_argument('--vp-compute-grad', default=cls.vp_compute_grad, action='store_false',
                            help='Allow gradient computation inside the VPLayer.')
        parser.add_argument('--vp-reg-factor', default=cls.vp_reg_factor, type=float,
                            help='VPLayer function system regularization factor. Negative value means no regularization.')
        parser.add_argument('--vp-res', default=cls.vp_res, action='store_true', help='VPLayer apply residual connection.')
        parser.add_argument('--vp-lin-weights', default=cls.vp_lin_weights, type=int, help='Number of weigths in the hidden layer after the VP.')
        parser.add_argument('--vp-loss-factor', default=cls.vp_loss_factor, type=float, help='VP loss weight.')
        parser.add_argument('--ff-weights', default=cls.ff_weights, type=int, nargs='*', help='List of sizes for the FF model hidden layers.')
        parser.add_argument('--cnn-kernel', default=cls.cnn_kernel, type=int, help='Kernel size for CNN model conv layer.')
        parser.add_argument('--cnn-pool-stride', default=cls.cnn_pool_stride, type=int, help='Stride size for CNN model pooling layer.')
        return parser.parse_args(namespace=cls())
    

    def __repr__(self) -> str:
        attrs = sorted([attr for attr in dir(self) if not attr.startswith('_') and attr != 'parse_args'])
        names = []
        for attr in attrs:
            value = getattr(self, attr)
            value = f'"{value}"' if isinstance(value, str) else value
            names.append(f'{attr}={value}')
        return f'{self.__class__.__name__}({", ".join(names)})'
