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
<<<<<<< HEAD
            nn.Linear(n_units, 7)
=======
<<<<<<< HEAD
            nn.Linear(n_units, 3)
=======
            nn.Linear(n_units, 2) 
>>>>>>> ced683fc06b7163308b8c19c61db8fc4cac0301f
>>>>>>> 7c6a57c998369e9c6228d933b9bd6aab4800935a
        )
    
    def forward(self, x):
        x = self.output(x)
        return x