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
<<<<<<< HEAD
            nn.Linear(6, 1),
            nn.Linear(1, 7)
=======
            nn.ReLU(),
            nn.Linear(6, 3)
>>>>>>> 7c6a57c998369e9c6228d933b9bd6aab4800935a
            )

    def forward(self, x):
        x = self.output(x)
        return x