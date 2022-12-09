# general modules
import os
import numpy as np

# sklearn
from sklearn.model_selection import train_test_split

# torch
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# my modules
import globals as g
from LoadCSV import LoadCSV

blip = LoadCSV("Blip_H1_O3a.csv", g.path_to_data, "Blip")
injection = LoadCSV("Injections_H1_O3a.csv", g.path_to_data, "Injections")

all_data = blip.dataset
y = blip.y

for i in range(len(blip)):
    all_data.append(injection.dataset[i])
    y.append(injection.y[i])

all_data = torch.tensor(all_data, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float)

X_train, X_test, y_train, y_test = train_test_split(all_data, y, train_size=0.8, random_state=42)

train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_data, shuffle=True, batch_size=12)
test_dataloader = DataLoader(test_data, batch_size=len(test_data.tensors[0]))

n_datapoints = len(train_dataloader)

model = nn.Sequential(
    nn.Linear(6, 2),
    nn.Softmax(dim=1)
)

print(model)

epochs = 5
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        l = nn.CrossEntropyLoss()(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = l.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += nn.CrossEntropyLoss()(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct = torch.mean((pred == y).float()).item()
    test_loss /= num_batches
    #correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, optimizer)
    test_loop(test_dataloader, model)
print("Done!")
