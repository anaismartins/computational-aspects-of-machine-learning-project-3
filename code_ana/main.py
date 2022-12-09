# general modules
import os
import numpy as np
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix

# torch
from torch import nn
import torch

# my modules
import globals as g
from LoadCSV import LoadCSV

blip = LoadCSV("Blip_H1_O3a.csv", g.path_to_data, "Blip")
injection = LoadCSV("Injections_H1_O3a.csv", g.path_to_data, "Injections")

blip_data = blip.dataset
injection_data = injection.dataset

all_data = blip.dataset
y = blip.y

for i in range(len(blip)):
    all_data = np.append(all_data, injection_data[i])
    y = np.append(y, injection.y[i])

all_data_sparse = coo_matrix(all_data)

all_data, all_data_sparse, y = shuffle(all_data, all_data_sparse, y)

n_datapoints = len(all_data)

model = nn.Sequential(
    nn.Linear(n_datapoints, 2),
    nn.Softmax(dim=1)
)

print(model)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#training_data = all_data

#train_dataloader = DataLoader(training_data, batch_size=64)
