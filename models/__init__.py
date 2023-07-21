import torch

from .classic_models import *
from .functions import *
from .varpro_models import *
from .varpro import *


def get_model(conf: Config, n_classes: int, n_samples: int) -> torch.nn.Module:
    '''Create model instance from configuration.'''
    if conf.model_name == 'vp':
        return get_vp_model(conf, n_classes, n_samples)
    if conf.model_name == 'vpnet':
        return get_vpnet_model(conf, n_classes, n_samples)
    if conf.model_name == 'ff':
        return FFClassifier(n_classes, n_samples, conf.ff_weights)
    if conf.model_name == 'cnn':
        return CNNClassifier(n_classes, n_samples, conf.cnn_kernel, conf.cnn_pool_stride, conf.ff_weights)
    raise ValueError(f'Unknown model name: "{conf.model_name}". Valid values: "vp", "vpnet", "ff", "cnn".')
