import torch
import torch.nn.functional as F


class FFClassifier(torch.nn.Module):
    '''Simple feed forward classifier.'''

    def __init__(self, n_classes: int, n_samples: int, n_weights: list[int]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for n_in, n_out in list(zip([n_samples] + n_weights, n_weights)):
            self.layers.append(torch.nn.Linear(n_in, n_out))
        self.lin_out = torch.nn.Linear(([n_samples] + n_weights)[-1], n_classes)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.lin_out(x)
        return x, x_in


class CNNClassifier(torch.nn.Module):
    '''Simple CNN classifier.'''

    def __init__(self, n_classes: int, n_samples: int, kernel: int, pool_stride: int, n_weights: list[int]) -> None:
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=kernel)
        self.pool = torch.nn.MaxPool1d(kernel_size=1, stride=pool_stride)
        dummy = self.pool(self.conv(torch.zeros(1, 1, n_samples)))
        self.ff = FFClassifier(n_classes, dummy.shape[-1], n_weights)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x
        x = self.conv(x.unsqueeze(1)).squeeze(1)
        x = self.pool(x)
        x, _ = self.ff(x)
        return x, x_in
