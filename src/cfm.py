from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def cfm(y, pred_labels, m, detector, n_units, n_units2, n_units3, n_layers, a, l, o, lr, epochs, num_batches, test_accuracy):

    #classes = ("Injection", "Blip", "Koyfish", "Low Frequency Burst", "Tomte", "Whistle", "Fast Scattering")
    #cf_matrix = confusion_matrix(y, pred_labels)
    #df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes], columns = [i for i in classes])
    #plt.figure(figsize = (12,7))
    #sn.heatmap(df_cm, annot=True)

    sns.heatmap(confusion_matrix(y, pred_labels), annot=True, fmt="d")

    dir_list = os.listdir("../output/cfms/")

    exists = False

    if m == "Perceptron":
        filename = "Acc_" + detector + "_Perceptron_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.png"
    elif m == "VariableNet":
        filename = "Acc_" + detector + "_VariableNet" + str(n_units) + "Units" + str(n_layers) + "Layers_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.png"
    elif m == "OneLayer":
        filename = "Acc_" + detector + "_OneLayer" + str(n_units) + "Units_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.png"
    elif m == "TwoLayers":
        filename = "Acc_" + detector + "_TwoLayers" + str(n_units) + "_" + str(n_units2) + "Units_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.png"
    elif m == "ThreeLayers":
        filename = "Acc_" + detector + "_ThreeLayers" + str(n_units) + "_" + str(n_units2) + "_" + str(n_units3) + "Units_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.png"


    for file in dir_list:
        if filename in file:
            exists = True

        if filename in file and float(file.split("Acc")[0]) < round(test_accuracy, 2):
            os.remove("../output/cfms/" + file)
            plt.savefig("../output/cfms/" + str(round(test_accuracy, 2)) + filename)

    if not exists:
        plt.savefig("../output/cfms/" + str(round(test_accuracy, 2)) + filename)