from torch import nn

class Perceptron(nn.Module):
    """
    A simple perceptron model
    """
    def __init__(self):
        """
        runs when the object is created
        """
        super(Perceptron, self).__init__()

        # self.activation = activation()

        self.output = nn.Sequential(
            nn.Linear(6, 1),
            nn.Linear(1, 7)
            )

    def forward(self, x):
        x = self.output(x)
        return x