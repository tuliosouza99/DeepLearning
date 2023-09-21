import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size: int, n_classes: int):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 12)
        self.layer2 = nn.Linear(12, n_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.softmax(x)

        return x
