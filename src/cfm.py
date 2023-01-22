from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def cfm(y, pred_labels, filename, test_accuracy):
    """
    function that plots the confusion matrix and saves it in the output folder
    
    :param y: the true labels
    :param pred_labels: the predicted labels
    :param filename: the name of the model
    :param test_accuracy: the accuracy of the model
    :return: None
    """

    # confusion matrix
    sns.heatmap(confusion_matrix(y, pred_labels), annot=True, fmt="d")

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