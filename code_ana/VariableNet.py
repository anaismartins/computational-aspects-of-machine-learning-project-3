from torch import nn

class VariableNet(nn.Module):
    def __init__(self, n_units, n_layers):
        super().__init__()
        self.n_layers = n_layers

        self.layers = nn.ModuleDict()
        self.layers["input"] = nn.Linear(6, n_units)

        for i in range(self.n_layers):
            self.layers[str(i)] = nn.Linear(n_units, n_units)

        self.layers["output"] = nn.Linear(n_units, 2)

    def forward(self, x):
        x = self.layers["input"](x)

        for i in range(self.n_layers):
            x = nn.functional.relu(self.layers[str(i)](x))

        return self.layers["output"](x)