# general modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# sklearn
from sklearn.model_selection import train_test_split

# torch
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# my modules
import globals as g
from LoadCSV import LoadCSV
from Perceptron import Perceptron
from VariableNet import VariableNet
from OneLayer import OneLayer

from model_training import train_model
from results_plotting import plot_results

# loading the data
blip = LoadCSV("Blip_H1_O3a.csv", g.path_to_data, "Blip")
injection = LoadCSV("Injections_H1_O3a.csv", g.path_to_data, "Injections")

# joining the data and preping it for the model
X = blip.dataset
y = blip.y

for i in range(len(blip)):
    X.append(injection.dataset[i])
    y.append(injection.y[i])

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=142) # 12 batches for the data size we have 
test_dataloader = DataLoader(test_data, batch_size=len(test_data.tensors[0])) # loading the whole test data at once

# SPECIFY THE MODEL HERE ---------------------------------------------
n_units = 6
n_layers = 2

activation = nn.ReLU()
a = "ReLU"

model = Perceptron()
m = "Perceptron"
print(model)

# specifications for compiling the model
epochs = 200
loss_fn = nn.CrossEntropyLoss()
l = "CrossEntropyLoss"
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
o = "Adam"

train_accuracies, test_accuracies = train_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, lr = 0.01, epochs = 200)

plot_results(train_accuracies, test_accuracies, m, a, l, o, n_units)