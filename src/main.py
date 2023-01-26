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
from Perceptron import Perceptron
from VariableNet import VariableNet
from OneLayer import OneLayer
from TwoLayers import TwoLayers
from ThreeLayers import ThreeLayers
from FourLayers import FourLayers

from train_model import train_model
from plot_results import plot_results
from cfm import cfm
from save_model import save_model
from filename import filename


# DEFINE DETECTOR ----------------------------------------------------------------------------------------
detector = "H1"

if detector != "V1":
    num_classes = 7
else:
    num_classes = 6


# LOAD DATA AND SPLIT INTO TRAIN AND TEST -----------------------------------------------------------------
data = np.load("../datasets/dataset_all_" + detector + "_bootstrap.npy")

# dividing the data into X and y
X = data[:,:-1]
y = data[:,-1]

# passing X and y to torch tensors
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# setting the k for k-fold cross validation
k = 10
kfold = KFold(n_splits=k, shuffle=False)

# prepping the lists to store the results
all_training_accuracies = []
all_valid_accuracies = []
all_train_loss = []
all_valid_loss = []
all_pred_labels = []
all_final_epochs = []
all_test_accuracies = []

# starting iteration
fold = 1

for train_index, valid_index in kfold.split(X_train, y_train):  
    # splitting the data into train and validation for each fold
    X_train_fold = X_train[train_index] 
    y_train_fold = y_train[train_index] 
    X_valid_fold = X_train[valid_index] 
    y_valid_fold = y_train[valid_index] 

    # passing the data to torch datasets
    train_fold_data = TensorDataset(X_train_fold, y_train_fold)
    valid_fold_data = TensorDataset(X_valid_fold, y_valid_fold)

    #setting batch size to have num_batches batches per epoch
    num_batches = 10
    batch_size = round(len(train_fold_data.tensors[0])/num_batches)

    # passing the data to torch dataloaders
    train_dataloader = DataLoader(train_fold_data, shuffle=True, batch_size=batch_size) 
    # loading the whole test data at once
    valid_dataloader = DataLoader(valid_fold_data, batch_size=len(valid_fold_data.tensors[0])) 

    test_dataloader = DataLoader(TensorDataset(X_test, y_test), batch_size=len(X_test))
    
    
    # MODEL SPECS ----------------------------------------------------------------------------------------
    max_epochs = 200

    a = "ReLU"

    n_layers = 2
    # generally 10 to 512 units
    n_units = 128
    n_units2 = 128
    n_units3 = 128
    n_units4 = 128

    # model needs to be called in the loop to reset the weights
    model = Perceptron(num_classes)
    m = "Perceptron"
    #model = OneLayer(num_classes, n_units, a)
    #m = "OneLayer"
    #model = TwoLayers(num_classes, n_units, n_units2, a)
    #m = "TwoLayers"
    #model = ThreeLayers(num_classes, n_units, n_units2, n_units3, a)
    #m = "ThreeLayers"
    model = FourLayers(num_classes, n_units, n_units2, n_units3, n_units4, a)
    m = "FourLayers"

    # LOSS AND OPTIMIZER ---------------------------------------------------------------------------------
    # initial learning rate
    lr = 1e-3

    # optimzer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    o = "Adam"

    # setting weights for loss function
    l = "CrossEntropyLoss"
    if detector != "V1":
        class_weights = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1]) #[injection, blips, koyfish, lowfreq, tomte, whistle, fast scattering]
    else:
        class_weights = torch.FloatTensor([1, 1, 1, 1, 1, 1]) #[injection, blips, koyfish, lowfreq, tomte, whistle]
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    # smart learning rate
    # factor by which the learning rate will be multiplied
    lr_decay_factor = 0.9 
    # number of epochs with no improvement after which learning rate will be reduced
    lr_decay_patience = 100 
    lr_sch = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor = lr_decay_factor, patience = lr_decay_patience)

    print("\n\n\nFold " + str(fold) + " of " + str(k))
    

    # TRAINING AND TESTING--------------------------------------------------------------------------------
    pred_labels, train_accuracies, valid_accuracies, train_loss, valid_loss, test_accuracy, final_epoch, model = train_model(train_dataloader, valid_dataloader, test_dataloader, model, loss_fn, optimizer, lr_sch, epochs = max_epochs)
    
    # saving the results
    all_pred_labels.append(pred_labels)
    all_training_accuracies.append(train_accuracies)
    all_valid_accuracies.append(valid_accuracies)
    all_train_loss.append(train_loss)
    all_valid_loss.append(valid_loss)
    all_test_accuracies.append(test_accuracy)
    all_final_epochs.append(final_epoch)
    
    # saving the best model
    if fold == 1:
        best_model = model
        best_accuracy = valid_accuracies[-1]

    if valid_accuracies[-1] > best_accuracy:
        best_model = model
        best_accuracy = valid_accuracies[-1]

    fold += 1


# FINAL ACCURACY ----------------------------------------------------------------------------------------
# averaged test accuracies
av_test_accuracy = 0
for i in range(0, k):
    av_test_accuracy += all_test_accuracies[i]
av_test_accuracy = av_test_accuracy/k

accuracy_error = 3 * np.std(all_test_accuracies) / np.sqrt(k)
print("Average test accuracy: " + str(round(av_test_accuracy, 2)) + " +/- " + str(round(accuracy_error, 2)) + " %")

# ordered predicted labels to match the order of the original dataset and do the cfm afterwards
av_pred_labels = [0] * len(y_test)
count = [0] * num_classes

for j in range(0, len(y_test)):
    for i in range(0, k):
        count[all_pred_labels[i][j]] += 1
        av_pred_labels[j] = count.index(max(count))
    count = [0] * num_classes
  
# averaging the final epoch
final_epoch = 0
for i in range(0, k):
    final_epoch += all_final_epochs[i]
final_epoch = round(final_epoch/k)

# setting the filename for all
filename = filename(m, detector, a, l, o, lr, final_epoch, num_batches, n_layers, n_units, n_units2, n_units3, n_units4)

size = len(y_train) / num_classes 
# CONFUSION MATRIX --------------------------------------------------
cfm(y_test, av_pred_labels, filename, av_test_accuracy, size, num_classes)


# PLOT RESULTS ------------------------------------------------------
plot_results(all_training_accuracies, all_valid_accuracies, all_train_loss, all_valid_loss, av_test_accuracy, k, filename)


# SAVE MODEL --------------------------------------------------------
save_model(best_model, av_test_accuracy, filename)