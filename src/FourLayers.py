from torch import nn

class FourLayers(nn.Module):
    def __init__(self, num_classes, n_units, n_units2, n_units3, n_units4, a):
        super().__init__()

        if a == "ReLU":
            self.activation = nn.ReLU

        self.output = nn.Sequential(
            nn.Linear(6, n_units),
            self.activation(),
            nn.Linear(n_units, n_units2),
            self.activation(),
            nn.Linear(n_units2, n_units3),
            self.activation(),
            nn.Linear(n_units3, n_units4),
            self.activation(),
            nn.Linear(n_units4, num_classes)
        )
    
    def forward(self, x):
        x = self.output(x)
        return x