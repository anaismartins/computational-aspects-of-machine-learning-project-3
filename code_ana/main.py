from torch import nn

# my modules
import globals as g
from LoadCSV import LoadCSV

filenames = os.listdir(g.path_to_data)
print(filenames)

# dataloader = LoadCSV

model = nn.Sequential(
    nn.Linear(n_datapoints, 2),
    nn.Softmax(dim=1)
)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
