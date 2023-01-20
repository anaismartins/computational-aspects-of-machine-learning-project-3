# general modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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
from TwoLayers import TwoLayers
from ThreeLayers import ThreeLayers

from train_model import train_model
from plot_results import plot_results
from cfm import cfm
from save_model import save_model

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

kfold = KFold(n_splits=10, shuffle=False)

all_training_accuracies = []
all_valid_accuracies = []
all_pred_labels = []

fold = 1

for train_index, valid_index in kfold.split(X, y):  
    X_train_fold = X[train_index] 
    y_train_fold = y[train_index] 
    X_valid_fold = X[valid_index] 
    y_valid_fold = y[valid_index] 

    train_fold_data = TensorDataset(X_train_fold, y_train_fold)
    valid_fold_data = TensorDataset(X_valid_fold, y_valid_fold)

    #setting batch size to have num_batches batches per epoch
    num_batches = 10
    batch_size = round(len(train_fold_data.tensors[0])/num_batches)

    train_dataloader = DataLoader(train_fold_data, shuffle=True, batch_size=batch_size) 
    valid_dataloader = DataLoader(valid_fold_data, batch_size=len(valid_fold_data.tensors[0])) # loading the whole test data at once

    # MODEL SPECS -------------------------------------------------------
    max_epochs = 200
    n_units = 512 # generally 10 to 512
    n_layers = 10
    a = "ReLU"

    n_units2 = 10
    n_units3 = 512

    model = OneLayer(num_classes, n_units, a)
    m = "OneLayer"


    # LOSS AND OPTIMIZER -------------------------------------------------
    lr = 1e-3

    # optimzer
    o = "Adam"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # setting weights for loss function
    l = "CrossEntropyLoss"
    if detector != "V1":
        class_weights = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1]) #[injection, blips, fast scattering, koyfish, lowfreq, tomte, whistle]
    else:
        class_weights = torch.FloatTensor([1, 1, 1, 1, 1, 1]) #[injection, blips, koyfish, lowfreq, tomte, whistle]
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # smart learning rate
    lr_decay_factor = 0.9 # factor by which the learning rate will be multiplied
    lr_decay_patience = 50 # number of epochs with no improvement after which learning rate will be reduced

    lr_sch = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = lr_decay_factor, patience = lr_decay_patience)


    print("\n\n\nFold " + str(fold) + " of 10")
    fold += 1


    # TRAINING AND TESTING------------------------------------
    pred_labels, train_accuracies, valid_accuracies, final_epoch = train_model(train_dataloader, valid_dataloader, model, loss_fn, optimizer, lr_sch, epochs = max_epochs)
    all_pred_labels.append(pred_labels)
    all_training_accuracies.append(train_accuracies)
    all_valid_accuracies.append(valid_accuracies)

# FINAL ACCURACY ----------------------------------------------------
# averaged validation accuracies
av_validation_accuracy = 0
for i in range(0, 10):
    av_validation_accuracy += all_valid_accuracies[i][-1]
av_validation_accuracy = av_validation_accuracy/10
print("Average validation accuracy: " + str(av_validation_accuracy))

# averaging the labels - NEED TO FIGURE OUT HOW TO DO THIS
#for train_index, valid_index in kfold.split(X, y):
# ordered_pred_labels[i][valid_index] = all_pred_labels[i]
    

# CONFUISON MATRIX --------------------------------------------------
#cfm(y, pred_labels, m, detector, n_units, n_units2, n_units3, n_layers, a, l, o, lr, epochs = final_epoch, num_batches = num_batches, test_accuracy = av_validation_accuracy)

# PLOT RESULTS ------------------------------------------------------
plot_results(all_training_accuracies, all_valid_accuracies, av_validation_accuracy, m, a, l, o, lr, final_epoch, n_units, n_units2, n_units3, n_layers, detector = detector, num_batches = num_batches)

# SAVE MODEL --------------------------------------------------------
# this is saving the model from the last fold but i dont think melissa really cares abt the model being saved
save_model(model, m, detector, n_units, n_units2, n_units3, n_layers, a, l, o, lr, final_epoch, num_batches, av_validation_accuracy)