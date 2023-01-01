from torch import nn
import torch.nn.functional as F

class VariableNet(nn.Module):
    def __init__(self, n_units, n_layers, a):
        super().__init__()

        if a == "ReLU":
            self.activation = F.relu

        self.n_layers = n_layers

        self.layer = nn.ModuleList()

        self.layer.append(nn.Linear(6, n_units))

        for i in range(1, self.n_layers + 1):
            self.layer.append(nn.Linear(n_units, n_units))

        self.layer.append(nn.Linear(n_units, 7))

    def forward(self, x):
        x = self.layer[0](x)

        for i in range(1, self.n_layers + 1):
            x = self.activation(self.layer[i](x))

        x = self.layer[self.n_layers + 1](x)

        return x