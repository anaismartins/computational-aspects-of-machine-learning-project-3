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
import torch.optim.lr_scheduler as lr_scheduler

# my modules
import globals as g
from Perceptron import Perceptron
from VariableNet import VariableNet
from OneLayer import OneLayer

from model_training import train_model
from results_plotting import plot_results

data = np.load("../datasets/dataset_all_h1_bootstrap.npy") # uncomment this line to use the bootstrap dataset
#data = np.load("../datasets/dataset_all_h1.npy") # uncomment this line to use the original dataset
X = data[:,:-1]
y = data[:,-1]

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = round(len(train_data.tensors[0])/20)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size) 
test_dataloader = DataLoader(test_data, batch_size=len(test_data.tensors[0])) # loading the whole test data at once

# SPECIFY THE MODEL HERE ---------------------------------------------
n_units = 20 # generally 10 to 512
n_layers = 3
lr = 1e-3
lr_decay_factor = 0.9 # factor by which the learning rate will be multiplied
lr_decay_patience = 50 # number of epochs with no improvement after which learning rate will be reduced

a = "ReLU"

#model = OneLayer(n_units, a)
#m = "OneLayer"

model = VariableNet(n_units, n_layers, a)
m = "VariableNet"
print(model)

# specifications for the loss function
class_weights = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1]) #[injection, blips, fast scattering, koyfish, lowfreq, tomte, whistle]
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
l = "CrossEntropyLoss"

# specifications for compiling the model
epochs = 2000
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = lr_decay_factor, patience = lr_decay_patience)
o = "Adam"


train_accuracies, test_accuracies = train_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, lr_scheduler, epochs = epochs)

plot_results(train_accuracies, test_accuracies, m, a, l, o, lr, epochs, n_units, n_layers)