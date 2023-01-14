from torch import nn

class TwoLayers(nn.Module):
    """
    model with one hidden layer
    """
    def __init__(self, num_classes, n_units1, n_units2, a):
        """
        runs when the object is created
        """
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