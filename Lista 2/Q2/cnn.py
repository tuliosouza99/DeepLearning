import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from utils import update_pbar


class Net(torch.nn.Module):
    def __init__(self, input_size: int = 28, input_channels: int = 1):
        super().__init__()

        n_maxpool = 2
        last_conv_dim = input_size // (2**n_maxpool)

        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=3, padding=1
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc3 = torch.nn.Linear(64 * last_conv_dim * last_conv_dim, 128)
        self.fc4 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)

        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, 0.5)
        x = self.fc4(x)

        return x


def train(
    net: Net,
    device: torch.device,
    n_epochs: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 0.001,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(n_epochs):
        train_loss = 0.0
        train_acc = 0.0
        net.train(mode=True)

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{n_epochs}') as pbar:
            for i, data in enumerate(train_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_acc += (predicted == labels).cpu().float().mean()

                update_pbar(
                    pbar,
                    n_epochs,
                    epoch,
                    train_loss / (i + 1),
                    train_acc / (i + 1),
                    'train',
                )

            update_pbar(
                pbar,
                n_epochs,
                epoch,
                train_loss / len(train_loader),
                train_acc / len(train_loader),
                'train',
            )

        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc / len(train_loader))

        net.train(mode=False)
        val_loss = 0.0
        val_acc = 0.0

        with tqdm(total=len(val_loader), desc=f'Epoch {epoch+1}/{n_epochs}') as pbar:
            for i, data in enumerate(val_loader):
                inputs, labels = data[0].to(device), data[1].to(device)

                with torch.no_grad():
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_acc += (predicted == labels).cpu().float().mean()

                update_pbar(
                    pbar, n_epochs, epoch, val_loss / (i + 1), val_acc / (i + 1), 'val'
                )

            update_pbar(
                pbar,
                n_epochs,
                epoch,
                val_loss / len(val_loader),
                val_acc / len(val_loader),
                'val',
            )
            print()

        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc / len(val_loader))

    return train_losses, train_accs, val_losses, val_accs


def test(net: Net, device: torch.device, test_loader: DataLoader):
    net.train(mode=False)
    y_true = []
    y_pred = []

    for data in tqdm(test_loader):
        images, labels = data[0].to(device), data[1].to(device)

        with torch.no_grad():
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predicted.cpu().tolist())

    return y_true, y_pred
