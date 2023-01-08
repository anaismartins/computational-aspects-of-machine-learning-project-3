from torch import nn

class OneLayer(nn.Module):
    """
    model with one hidden layer
    """
    def __init__(self, num_classes, n_units, a):
        """
        runs when the object is created
        """
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