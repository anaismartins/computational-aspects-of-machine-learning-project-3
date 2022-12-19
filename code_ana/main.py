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

<<<<<<< HEAD
data = np.load("../dataset_all_h1.npy")
=======
data = np.load("../dataset_inj_blip_koyfish.npy")
>>>>>>> 7c6a57c998369e9c6228d933b9bd6aab4800935a
X = data[:,:-1]
y = data[:,-1]

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

print(y)

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 429

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size) # 12 batches for the data size we have 
test_dataloader = DataLoader(test_data, batch_size=len(test_data.tensors[0])) # loading the whole test data at once

# SPECIFY THE MODEL HERE ---------------------------------------------
<<<<<<< HEAD
n_units = 10 # number of units in the hidden layer
=======
n_units = 10 # generally 10 to 512
>>>>>>> ced683fc06b7163308b8c19c61db8fc4cac0301f
n_layers = 100
lr = 0.0001

activation = nn.ReLU()
a = "ReLU"

model = OneLayer(n_units)
m = "OneLayer"
print(model)

# specifications for compiling the model
epochs = 200
loss_fn = nn.CrossEntropyLoss()
l = "CrossEntropyLoss"
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
o = "Adam"

train_accuracies, test_accuracies = train_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, lr = lr, epochs = epochs)

plot_results(train_accuracies, test_accuracies, m, a, l, o, n_units, n_layers)