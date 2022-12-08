from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu = nn.Sequential(
            nn.Linear(994, 2),
        )

    def forward(self, x):
        logits = self.linear_relu(x)
        return logits