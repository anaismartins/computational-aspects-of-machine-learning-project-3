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

# DEFINE DETECTOR ----------------------------------------------------
detector = "H1"

if detector != "V1":
    num_classes = 7
else:
    num_classes = 6


# LOAD DATA AND SPLIT INTO TRAIN AND TEST -----------------------------
data = np.load("../datasets/dataset_all_" + detector + "_bootstrap.npy") # uncomment this line to use the bootstrap dataset
#data = np.load("../datasets/dataset_all_h1.npy") # uncomment this line to use the original dataset
X = data[:,:-1]
y = data[:,-1]

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# splitting the data into train and test
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, train_size=0.5)

train_data = TensorDataset(X_train, y_train)
valid_data = TensorDataset(X_valid, y_valid)
test_data = TensorDataset(X_test, y_test)
#setting batch size to have 20 batches per epoch
num_batches = 10
batch_size = round(len(train_data.tensors[0])/num_batches)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size) 
valid_dataloader = DataLoader(valid_data, batch_size=len(valid_data.tensors[0])) # loading the whole test data at once
test_dataloader = DataLoader(test_data, batch_size=len(test_data.tensors[0])) # loading the whole test data at once

# MODEL SPECS -------------------------------------------------------
n_units = 450 # generally 10 to 512
n_layers = 10
a = "ReLU"

model = OneLayer(num_classes, n_units, a)
m = "OneLayer"
print(model)


# LOSS AND OPTIMIZER -------------------------------------------------
lr = 1e-3

# optimzer
o = "Adam"
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# getting ratio between data sizes
i = 0
with open("../dataprep/datasize.txt", 'r') as f:
    for line in f:
        if i == 0:
            biggest = int(line)
            i += 1
        if i == 1:
            injection_size = int(line)

size_ratio = biggest/injection_size
#size_ratio = 1 #uncomment if data is all the same size

# setting weights for loss function
l = "CrossEntropyLoss"
if detector != "V1":
    class_weights = torch.FloatTensor([size_ratio, 1, 1, 1, 1, 1, 1]) #[injection, blips, fast scattering, koyfish, lowfreq, tomte, whistle]
else:
    class_weights = torch.FloatTensor([size_ratio, 1, 1, 1, 1, 1]) #[injection, blips, koyfish, lowfreq, tomte, whistle]
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# smart learning rate
lr_decay_factor = 0.9 # factor by which the learning rate will be multiplied
lr_decay_patience = 50 # number of epochs with no improvement after which learning rate will be reduced

lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = lr_decay_factor, patience = lr_decay_patience)


# TRAINING, TESTING AND PLOTTING ------------------------------------
max_epochs = 200000
train_accuracies, valid_accuracies, final_epoch = train_model(train_dataloader, valid_dataloader, model, loss_fn, optimizer, lr_scheduler, epochs = max_epochs)


# FINAL ACCURACY ----------------------------------------------------
X, y = next(iter(test_dataloader))
pred_labels = torch.argmax(model(X), axis = 1)

test_accuracy = 100 * torch.mean((pred_labels == y).float()).item()
print("Test accuracy: " + str(test_accuracy))

plot_results(train_accuracies, valid_accuracies, test_accuracy, m, a, l, o, lr, final_epoch, n_units, n_layers, detector = detector, num_batches = num_batches)


