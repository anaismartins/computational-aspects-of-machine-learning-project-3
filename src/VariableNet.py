from torch import nn

class VariableNet(nn.Module):
    """
    model with variable number of hidden layers and variable number of units

    :param num_classes: number of classes
    :param n_units: number of units in the hidden layer
    :param n_layers: number of hidden layers
    :param a: activation function
    """
    def __init__(self, num_classes, n_units, n_layers, a):
        super().__init__()

        if a == "ReLU":
            self.activation = nn.ReLU

        self.n_layers = n_layers

        self.layer = nn.ModuleList()

        self.layer.append(nn.Linear(6, n_units))

        for i in range(1, self.n_layers + 1):
            self.layer.append(nn.Linear(n_units, n_units))

        self.layer.append(nn.Linear(n_units, num_classes))

    def forward(self, x):
        x = self.layer[0](x)

        for i in range(1, self.n_layers + 1):
            x = self.activation(self.layer[i](x))

        x = self.layer[self.n_layers + 1](x)

        return x