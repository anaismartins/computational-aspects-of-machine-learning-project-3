from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def cfm(y, pred_labels, filename, test_accuracy, size, num_classes):
    """
    function that plots the confusion matrix and saves it in the output folder
    
    :param y: the true labels
    :param pred_labels: the predicted labels
    :param filename: the name of the model
    :param test_accuracy: the accuracy of the model
    :return: None
    """

    # confusion matrix
    classes = ("Injection", "Blip", "Koyfish", "Low Freq", "Tomte", "Whistle", "Fast Scat")
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

    print(cf_matrix)

    labels = np.asarray(labels).reshape(num_classes,num_classes)

    print(labels)

    plt.figure(figsize = (12,7))
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='')

    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Actual Values")

    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)

    # getting the directory to store the confusion matrix
    dir_list = os.listdir("../output/cfms/")

    exists = False
    filename = filename + ".png"

    for file in dir_list:
        # checking if the same model already is saved
        if filename in file:
            exists = True

            # check if the accuracy is better and save the model with the best accuracy
            if float(file.split("Acc")[0]) < round(test_accuracy, 2):
                os.remove("../output/cfms/" + file)
                plt.savefig("../output/cfms/" + str(round(test_accuracy, 2)) + filename)

    # if the model is not saved yet, save it
    if not exists:
        plt.savefig("../output/cfms/" + str(round(test_accuracy, 2)) + filename)