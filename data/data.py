import scipy.io
import torch


def load_data(data_type: str, shuffle: bool = True, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Load the specified dataset, shuffle and split.'''
    if data_type == 'syn':
        return load_hermite(shuffle=shuffle, device=device)
    if data_type == 'ecg':
        return load_ecg(shuffle=shuffle, device=device)
    else:
        raise ValueError(f'Unknown dataset type: "{data_type}". Valid values: "syn", "ecg".')


def load_hermite(shuffle: bool = True, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Load the synthetic hermite dataset, shuffle and split.'''
    x, y = _load_mat('./data/synhermite.mat', device=device)
    x, y = _shuffle(x, y) if shuffle else (x, y)
    n = x.shape[0] // 2
    train_x, test_x = x[:n], x[n:]
    train_y, test_y = y[:n], y[n:]
    return train_x, train_y, test_x, test_y


def load_ecg(shuffle: bool = True, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Load the ecg dataset, shuffle and split.'''
    train_x, train_y = _load_mat('./data/ecg_train.mat', device=device)
    test_x, test_y = _load_mat('./data/ecg_test.mat', device=device)
    train_x, train_y = _shuffle(train_x, train_y) if shuffle else (train_x, train_y)
    return train_x, train_y, test_x, test_y


def _load_mat(path: str, device: str = 'cpu') -> tuple[torch.Tensor, torch.Tensor]:
    '''Load matlab matrix into torch tensors.'''
    mat = scipy.io.loadmat(path)
    x = torch.tensor(mat['samples'], dtype=torch.float32, device=device)
    y = torch.tensor(mat['labels'], dtype=torch.float32, device=device)
    return x, y


def _shuffle(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''Shuffle the input tensors.'''
    order = torch.randperm(x.shape[0])
    x, y = x[order], y[order]
    return x, y
