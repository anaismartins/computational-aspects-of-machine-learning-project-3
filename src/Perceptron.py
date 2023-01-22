from torch import nn

class Perceptron(nn.Module):
    """
    A simple perceptron model
    :param num_classes: number of classes
    """
    def __init__(self, num_classes):
        super(Perceptron, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(6, num_classes)
            )

    def forward(self, x):
        x = self.output(x)
        return x