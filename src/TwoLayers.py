from torch import nn

class TwoLayers(nn.Module):
    """
    model with two hidden layers
    :param num_classes: number of classes
    :param n_units1: number of units in the first hidden layer
    :param n_units2: number of units in the second hidden layer
    :param a: activation function (string of the name)
    """
    def __init__(self, num_classes, n_units1, n_units2, a):
        super().__init__()

        if a == "ReLU":
            self.activation = nn.ReLU

        self.output = nn.Sequential(
            nn.Linear(6, n_units1),
            self.activation(),
            nn.Linear(n_units1, n_units2),
            self.activation(),
            nn.Linear(n_units2, num_classes)
        )
    
    def forward(self, x):
        x = self.output(x)
        return x