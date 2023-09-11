from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.figsize'] = [8, 6]
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['figure.dpi'] = 100
plt.style.use("seaborn-colorblind")


def cfm(y, pred_labels, filename, test_accuracy, size, num_classes, detector, binary, tw):
    """
    function that plots the confusion matrix and saves it in the output folder
    
    :param y: the true labels
    :param pred_labels: the predicted labels
    :param filename: the name of the model
    :param test_accuracy: the accuracy of the model
    :return: None
    """

    # confusion matrix
    if not binary:
        if detector != "V1":
            classes = ("Injection", "Blip", "Koyfish", "Low Freq", "Tomte", "Whistle", "Fast Scat")
        else:
            classes = ("Injection", "Blip", "Koyfish", "Low Freq", "Tomte", "Whistle")
    else:
        classes = ("Injection", "Blip")

    cf_matrix = confusion_matrix(y, pred_labels)

    labels = []
    aux = []

    for i in range(0, len(cf_matrix)):
        summ = np.sum(cf_matrix[i])

        for j in range(0, len(cf_matrix[0])):

            cf_matrix[i][j] = float(float(cf_matrix[i][j]) / float(summ) * 100)
            aux.append(str(cf_matrix[i][j]) + "%")
        labels.append(aux)
        aux = []

    labels = np.asarray(labels).reshape(num_classes,num_classes)

    plt.figure(figsize = (12,7))
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', vmin=0, vmax=100)

    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Actual Values")

    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    exists = False
    filename = filename + ".png"

    if not binary:
        folder_path = "../output/"+str(tw)+"/cfms/"
    else:
        folder_path = "../output/"+str(tw)+"/cfms/binary/"

    # getting the directory to store the confusion matrix
    dir_list = os.listdir(folder_path)

    for file in dir_list:
        # checking if the same model already is saved
        if filename in file:
            exists = True

            # check if the accuracy is better and save the model with the best accuracy
            if float(file.split("Acc")[0]) < round(test_accuracy, 2):
                os.remove(folder_path + file)
                plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename)

    # if the model is not saved yet, save it
    if not exists:
        plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename)