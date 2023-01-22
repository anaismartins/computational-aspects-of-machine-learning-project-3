from torch import nn

class OneLayer(nn.Module):
    """
    model with one hidden layer
    :param num_classes: number of classes
    :param n_units: number of units in the hidden layer
    :param a: activation function
    """
    def __init__(self, num_classes, n_units, a):
        super().__init__()

        if a == "ReLU":
            self.activation = nn.ReLU

        self.output = nn.Sequential(
            nn.Linear(6, n_units),
            self.activation(),
            nn.Linear(n_units, num_classes)
        )
    
    def forward(self, x):
        x = self.output(x)
        return x