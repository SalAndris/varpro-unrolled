import math
import tqdm
import torch

from models import get_model
from data import Config, load_data


def acc_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    '''Accuracy metric.'''
    pred = torch.argmax(y_pred, dim=1)
    true = torch.argmax(y_true, dim=1)
    return torch.sum(pred == true) / true.shape[0]


def get_optimizer(conf: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    '''Create optimizer instance from config.'''
    if conf.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
    if conf.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.momentum, weight_decay=conf.weight_decay)
    raise ValueError(f'Unknown optimizer "{conf.optimizer}"')


def train_cls(conf: Config, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 256) -> None:
    '''Train the classifier model.'''
    num_epochs = conf.epochs
    model.train()
    optimizer = get_optimizer(conf, model)
    loss_fn = torch.nn.CrossEntropyLoss()

    n_iter = math.ceil(len(x) / batch_size)
    for epoch in range(num_epochs):
        epoch_loss, epoch_vp_loss, epoch_acc = 0.0, 0.0, 0.0
        pbar = tqdm.tqdm(range(n_iter))
        for i in pbar:
            sample_x = x[i*batch_size:(i+1)*batch_size]
            sample_y = y[i*batch_size:(i+1)*batch_size]
            optimizer.zero_grad()
            y_pred, x_fit = model(sample_x)
            loss = loss_fn(y_pred, sample_y)
            vp_loss = torch.nn.functional.mse_loss(x_fit, sample_x) * conf.vp_loss_factor
            loss += vp_loss
            loss.backward()
            optimizer.step()
            acc = acc_fn(y_pred, sample_y)
            epoch_loss += loss.cpu().item()
            epoch_vp_loss += vp_loss.cpu().item()
            epoch_acc += acc.cpu().item()
            pbar.set_description_str(f'loss: {loss.cpu().item():.3f}, vploss: {vp_loss.cpu().item():.3f}, acc: {acc.cpu().item()*100:.2f}%')
        print(f'Epoch {epoch+1:2d} - Loss: {epoch_loss/n_iter:.3f}, VP Loss: {epoch_vp_loss/n_iter:.3f}, Accuracy: {epoch_acc/n_iter*100:.2f}%')


def eval_cls(conf: Config, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1024) -> None:
    '''Evaluate the trained model.'''
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss, vp_loss, acc = 0.0, 0.0, 0.0
    n_iter = math.ceil(len(x) / batch_size)
    for i in tqdm.tqdm(range(n_iter), desc='Eval'):
        sample_x = x[i*batch_size:(i+1)*batch_size]
        sample_y = y[i*batch_size:(i+1)*batch_size]
        y_pred, x_fit = model(sample_x)
        loss += loss_fn(y_pred, sample_y).cpu().item()
        vp_loss += torch.nn.functional.mse_loss(x_fit, sample_x).cpu().item() * conf.vp_loss_factor
        acc += acc_fn(y_pred, sample_y).cpu().item()
    loss += vp_loss
    print(f'Loss: {loss/n_iter:.2f}, VP Loss: {vp_loss/n_iter:.2f}, Accuracy: {acc/n_iter*100:.2f}%')


def main() -> None:
    '''Start the training process of the Unrolled VP network.'''
    torch.manual_seed(0)
    conf = Config.parse_args()
    train_x, train_y, test_x, test_y = load_data(conf.data_type, device=conf.device)
    model = get_model(conf, train_y.shape[1], train_x.shape[1])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print('No. params:', total_params)
    train_cls(conf, model, train_x, train_y)
    eval_cls(conf, model, test_x, test_y)
    #torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
