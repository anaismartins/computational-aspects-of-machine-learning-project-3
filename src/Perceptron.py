from torch import nn

class Perceptron(nn.Module):
    """
    A simple perceptron model
    """
    def __init__(self, num_classes):
        """
        runs when the object is created
        """
        super(Perceptron, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(6, num_classes)
            )

    def forward(self, x):
        x = self.output(x)
        return x