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

# creating the model
n_units = 6
n_layers = 1

model = Perceptron()
print(model)

# specifications for compiling the model
epochs = 200
train_accuracies, test_accuracies = [], []
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_model(train_dataloader, test_dataloader, model, lr = 0.01, epochs = 200):
    train_accuracies, test_accuracies = [], []
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for X, y in train_dataloader:
            pred = model(X)
            pred_labels = torch.argmax(pred, axis=1)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())

        X, y = next(iter(test_dataloader))
        pred_labels = torch.argmax(model(X), axis=1)
        test_accuracies.append(100 * torch.mean((pred_labels == y).float()).item())
        print(f"Epoch {epoch+1} | Test accuracy: {test_accuracies[-1]:.2f}%")

    return train_accuracies, test_accuracies

train_accuracies, test_accuracies = train_model(train_dataloader, test_dataloader, model, lr = 0.01, epochs = 200)

def plot_results(train_accuracies, test_accuracies, model):
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrows=2, ncols=1)

    # set plot title
    if model == Perceptron:
        fig.suptitle("Perceptron")
    elif model == VariableNet:
        fig.suptitle("VariableNet " + str(n_units) + " Units " + str(n_layers) + " Layers")
    fig.suptitle(str(model))

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(train_accuracies)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Accuracy")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(test_accuracies)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")

    fig.align_labels()
    if model == Perceptron():
        plt.savefig("../results/" + str(test_accuracies[-1]) + "Acc_Perceptron.png")
    elif model == VariableNet(n_units, n_layers):
        plt.savefig("../results/" + str(test_accuracies[-1]) + "Acc_VariableNet" + str(n_units) + "Units" + str(n_layers) + "Layers.png")

plot_results(train_accuracies, test_accuracies, model)