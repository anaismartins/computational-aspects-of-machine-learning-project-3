import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from Perceptron import Perceptron
from VariableNet import VariableNet
from OneLayer import OneLayer
from TwoLayers import TwoLayers
from ThreeLayers import ThreeLayers
from FourLayers import FourLayers
from torch_utils import train_model, save_model
from utils import plot_results, cfm, filename, prediction_plots
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

torch.set_printoptions(sci_mode=False)
# DEFINE DETECTOR --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--path', metavar='1', type=str,
                    help='Path to read data')
parser.add_argument('--run', metavar='1', type=str,
                    help='O3a or O3b')
parser.add_argument('--detector', metavar='1', type=str,
                    help='Name of detector')

# Define arguments
args = parser.parse_args()
path = args.path
tw = str(path.split('/')[-2][2:])  # define time window
run = args.run
detector = args.detector
binary = False
k = 9  # k-fold cross validation
print('Start time:', datetime.now())
# Define model params
num_batches = 10  # num batches epoch
lr = 1e-3
o = "Adam"
l = "CrossEntropyLoss"
lr_decay_factor = 0.9
lr_decay_patience = 100
max_epochs = 200000
a = "ReLU"
n_layers = 3
n_units = 350  # generally 10 to 512 units
n_units2 = 350
n_units3 = 350
n_units4 = 128

# LOAD DATA AND SPLIT INTO TRAIN AND TEST ------------------------------------
if not binary:
    data = np.load(path + "dataset_all_" + detector + "_bootstrap_" + run + ".npy")
    if detector != "V1":
        num_classes = 7
    else:
        num_classes = 6
else:
    data = np.load("../datasets/inj_blip_" + detector + ".npy")
    num_classes = 2
class_weights = torch.ones(num_classes, dtype=torch.float)
loss_fn = nn.CrossEntropyLoss(weight=class_weights)

# Pre-processing
print(data.shape)
X, y = data[:, :-1], data[:, -1]

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)
#y = F.one_hot(y, num_classes=7)  # One-hot-encoding
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=42)
kfold = KFold(n_splits=k, shuffle=False)

# prepping the lists to store the results
all_training_accuracies, all_valid_accuracies = [], []
all_train_loss, all_valid_loss = [], []
all_pred_labels, all_final_epochs = [], []
all_test_accuracies = []


fold = 1
for train_index, valid_index in kfold.split(X_train, y_train):

    X_train_fold, y_train_fold = X_train[train_index], y_train[train_index]
    X_valid_fold, y_valid_fold = X_train[valid_index], y_train[valid_index]

    train_fold_data = TensorDataset(X_train_fold, y_train_fold)
    valid_fold_data = TensorDataset(X_valid_fold, y_valid_fold)

    batch_size = round(len(train_fold_data.tensors[0])/num_batches)
    train_dataloader = DataLoader(train_fold_data,
                                  shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_fold_data,
                                  batch_size=len(valid_fold_data.tensors[0]))
    test_dataloader = DataLoader(TensorDataset(X_test, y_test),
                                 batch_size=len(X_test))

    # model needs to be called in the loop to reset the weights
    # FIXME: not sure if this should be called in loop
    #model = Perceptron(num_classes)
    #m = "Perceptron"
    # model = OneLayer(num_classes, n_units, a)
    # m = "OneLayer"
    # model = TwoLayers(num_classes, n_units, n_units2, a)
    # m = "TwoLayers"
    model = ThreeLayers(num_classes, n_units, n_units2, n_units3, a)
    m = "ThreeLayers"
    # model = FourLayers(num_classes, n_units, n_units2, n_units3, n_units4, a)
    # m = "FourLayers"
    # model = VariableNet(num_classes, n_units, n_layers, a)
    # m = "VariableNet"
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_sch = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                            factor=lr_decay_factor,
                                            patience=lr_decay_patience)

    print("\n\n\nFold " + str(fold) + " of " + str(k))

    # TRAINING AND TESTING--------------------------------------------------------------------------------
    pred_labels, train_accuracies, valid_accuracies, train_loss, valid_loss, test_accuracy, final_epoch, model = train_model(train_dataloader, valid_dataloader, test_dataloader, model, loss_fn, optimizer, lr_sch, epochs=max_epochs)

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
    

# FINAL ACCURACY -------------------
av_test_accuracy = 0
# FIXME: I think list.sum()/k will do instead of the loop
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
    final_epoch += all_final_epochs[i]  # FIXME: same
final_epoch = round(final_epoch/k)

filename = filename(m, detector, a, l, o, lr,
                    final_epoch, num_batches,
                    n_layers, n_units, n_units2, n_units3, n_units4)

size = len(y_train) / num_classes

cfm(y_test, av_pred_labels, filename, av_test_accuracy,
    size, num_classes, detector, binary, tw)
print('potato')
storeFolder = plot_results(all_training_accuracies, all_valid_accuracies,
             all_train_loss, all_valid_loss,
             av_test_accuracy, k, filename, binary, tw)
summary = [all_training_accuracies, all_valid_accuracies, all_train_loss, all_valid_loss, all_test_accuracies, all_final_epochs]
np.save(storeFolder + filename + 'summary.npy', summary)
print(storeFolder + filename + '_summary.npy')


prediction_plots(X_test, y_test, av_pred_labels,
                 num_classes, filename, av_test_accuracy, binary, storeFolder)

save_model(best_model, av_test_accuracy, filename, binary, tw)
