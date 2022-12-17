from torch import nn

class OneLayer(nn.Module):
    """
    model with one hidden layer
    """
    def __init__(self, n_units):
        """
        runs when the object is created
        """
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(6, n_units),
            nn.ReLU(),
            nn.Linear(n_units, 3)
        )
    
    def forward(self, x):
        x = self.output(x)
        return x