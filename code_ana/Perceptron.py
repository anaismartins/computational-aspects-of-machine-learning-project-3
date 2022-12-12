from torch import nn

class Perceptron(nn.Module):
    """
    A simple perceptron model
    """
    def __init__(self, activation):
        """
        runs when the object is created
        """
        super().__init__()
        self.output = nn.Sequential(
            activation(),
            nn.Linear(6, 2)
            )

    def forward(self, x):
        x = self.output(x)
        return x