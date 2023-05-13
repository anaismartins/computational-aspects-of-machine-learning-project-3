from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
import shutil

def cfm(y, pred_labels, filename, test_accuracy, size, num_classes, detector, binary):
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
    labels, aux = [], []

    for i in range(0, len(cf_matrix)):
        summ = np.sum(cf_matrix[i])
        for j in range(0, len(cf_matrix[0])):
            cf_matrix[i][j] = float(float(cf_matrix[i][j]) / float(summ) * 100)
            aux.append(str(cf_matrix[i][j]) + "%")
        labels.append(aux)
        aux = []

    labels = np.asarray(labels).reshape(num_classes, num_classes)

    plt.figure(figsize = (12,7))
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', vmin=0, vmax=100)
    ax.set_xlabel("Predicted Values"), ax.set_ylabel("Actual Values")
    ax.xaxis.set_ticklabels(classes), ax.yaxis.set_ticklabels(classes)

    exists = False
    filename = filename + ".png"

    if not binary: folder_path = "../output/cfms/"
    else: folder_path = "../output/cfms/binary/"

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

def filename(m, ifo, a, l, o, lr, epochs, n_batch, n_layers, n_units, n_units2, n_units3, n_units4):
    """
    Creates the filename for the saved model and the saved results
    :param m: model (string with the name)
    :param ifo: detector
    :param a: activation function (string with the name)
    :param l: loss function (string with the name)
    :param o: optimizer (string with the name)
    :param lr: inital set learning rate
    :param epochs: number of epochs that the model took
    :param n_batch: number of batches
    :param n_layers: number of layers
    :param n_units: number of units in the first layer
    :param n_units2: number of units in the second layer
    :param n_units3: number of units in the third layer
    :return: filename (string)
    """

    if m == "Perceptron":
        filename = "Acc_" + ifo + "_Perceptron"
    elif m == "VariableNet":
        filename = "Acc_" + ifo + "_VariableNet" + str(n_units) + "Units" + str(n_layers) + "Layers_" + a
    elif m == "OneLayer":
        filename = "Acc_" + ifo + "_OneLayer" + str(n_units) + "Units_" + a
    elif m == "TwoLayers":
        filename = "Acc_" + ifo + "_TwoLayers" + str(n_units) + "_" + str(n_units2) + "Units_" + a
    elif m == "ThreeLayers":
        filename = "Acc_" + ifo + "_ThreeLayers" + str(n_units) + "_" + str(n_units2) + "_" + str(n_units3) + "Units_" + a
    elif m == "FourLayers":
        filename = "Acc_" + ifo + "_FourLayers" + str(n_units) + "_" + str(n_units2) + "_" + str(n_units3) + "_" + str(n_units4) + "Units_" + a
    
    learning_params = + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(n_batch) + "batches"
    filename = filename + learning_params
    return filename


def plot_results(train_accuracies, valid_accuracies, train_loss, valid_loss, test_accuracy, k, filename, binary):
    """
    function that plots the results of the training and validation accuracy and saves it in the output folder
    :param train_accuracies: the training accuracies
    :param valid_accuracies: the validation accuracies
    :param test_accuracy: the test accuracy
    :param k: the number of folds
    :param filename: the name to give to the saved folder
    """

    # later on to check if the folder with this name was created in this run
    first = True

    # ACCURACY --------------------------------------------------------------------------------------------------
    for i in range(0, k):

        plt.clf()
        plt.plot(train_accuracies[i])
        plt.plot(valid_accuracies[i])
        plt.title("Model Accuracy")
        plt.xlabel("Epoch"), plt.ylabel("Accuracy")
        plt.legend(['Training', 'Validation'], loc='lower right')

        if not binary:
            folder_path = "../output/results/"
        else:
            folder_path = "../output/results/binary/"

        dir_list = os.listdir(folder_path)

        # initializing the exists boolean to False since we haven't found the folder yet
        exists = False

        for folder in dir_list:
            if filename in folder:
                # if there is at least one folder with this name, we set the exists boolean to True to later check if we need to create a new folder
                exists = True

            # if it's running for the first time, check for the existance of the folder for this same model and delete it if the new accuracy is better
            if first:
                if filename in folder and float(folder.split("Acc")[0]) < round(test_accuracy, 2):
                    shutil.rmtree(folder_path + folder)
                    os.makedirs(folder_path + str(round(test_accuracy, 2)) + filename)
                    plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/acc" + str(i) + ".png")
                    first = False
            else:
                plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/acc" + str(i) + ".png")
        
        if first:
            # if there isn't anything with the same name we create the new folder and save the plots
            if not exists:
                os.makedirs(folder_path + str(round(test_accuracy, 2)) + filename)
                plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/acc" + str(i) + ".png")
                first = False
        else:
            plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/acc" + str(i) + ".png")

    # LOSS --------------------------------------------------------------------------------------------------
    for i in range(0, k):

        #clearing the plot before starting
        plt.clf()

        # plotting training accuracy
        plt.plot(train_loss[i])
        
        # plotting validation accuracy
        plt.plot(valid_loss[i])

        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.legend(['Training', 'Validation'], loc='upper right')

        dir_list = os.listdir(folder_path)

        # initializing the exists boolean to False since we haven't found the folder yet
        exists = False

        for folder in dir_list:
            if filename in folder:
                # if there is at least one folder with this name, we set the exists boolean to True to later check if we need to create a new folder
                exists = True

            # if it's running for the first time, check for the existance of the folder for this same model and delete it if the new accuracy is better
            if first:
                if filename in folder and float(folder.split("Acc")[0]) < round(test_accuracy, 2):
                    shutil.rmtree(folder_path + folder)
                    os.makedirs(folder_path + str(round(test_accuracy, 2)) + filename)
                    plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/loss" + str(i) + ".png")
                    first = False
            else:
                plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/loss" + str(i) + ".png")
        
        if first:
            # if there isn't anything with the same name we create the new folder and save the plots
            if not exists:
                os.makedirs(folder_path + str(round(test_accuracy, 2)) + filename)
                plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/loss" + str(i) + ".png")
                first = False
        else:
            plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/loss" + str(i) + ".png")

def prediction_plots(X_test, y_test, av_pred_labels, num_classes, filename, test_accuracy, binary):

    X_real_injections, y_real_injections = [], []

    for i in range(0, len(y_test)):
        if y_test[i] == 0:
            X_real_injections.append(X_test[i])
            y_real_injections.append(av_pred_labels[i])

    m_1, m_2, s_1, s_2 = [], [], [], []

    for i in range(0, len(y_real_injections)):
        m_1.append(X_real_injections[i][2])
        m_2.append(X_real_injections[i][3])
        s_1.append(X_real_injections[i][4])
        s_2.append(X_real_injections[i][5])


    # use colormap
    colormap = np.array(['#d73027', '#fc8d59', '#fee090',
                         '#ffffbf', '#e0f3f8', '#91bfdb', '#4575b4'])
    labels = np.array(["Injection", "Blips", "Koyfish",
                       "Low Freq Burst", "Tomte", "Whistle", "Fast Scat"])

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(0, num_classes):
        x, y = [], []
        for j in range(0, len(y_real_injections)):
            if y_real_injections[j] == i:
                x.append(m_1[j])
                y.append(m_2[j])
        ax.scatter(x, y, s=50, c=colormap[i], label = labels[i])

    plt.title("Classification of Real Injections")
    plt.xlabel("Mass 1"), plt.ylabel("Mass 2")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if not binary: folder_path = "../output/pred_plots/"
    else: folder_path = "../output/pred_plots/binary/"

    # getting the directory to store the confusion matrix
    dir_list = os.listdir(folder_path)

    # initializing the exists boolean to False since we haven't found the folder yet
    exists, first = False, True

    for folder in dir_list:
        if filename in folder:
            # if there is at least one folder with this name, we set the exists boolean to True to later check if we need to create a new folder
            exists = True

        # if it's running for the first time, check for the existance of the folder for this same model and delete it if the new accuracy is better
        if first:
            if filename in folder and float(folder.split("Acc")[0]) < round(test_accuracy, 2):
                shutil.rmtree(folder_path + folder)
                os.makedirs(folder_path + str(round(test_accuracy, 2)) + filename)
                plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/masses.png")
                first = False
        else:
            plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/masses.png")
    
    if first:
        # if there isn't anything with the same name we create the new folder and save the plots
        if not exists:
            os.makedirs(folder_path + str(round(test_accuracy, 2)) + filename)
            plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/masses.png")
            first = False
    else:
        plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/masses.png")

    plt.clf()

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(0, num_classes):
        x = []
        y = []
        for j in range(0, len(y_real_injections)):
            if y_real_injections[j] == i:
                x.append(s_1[j])
                y.append(s_2[j])

        plt.scatter(x, y, s=50, c=colormap[i], label = labels[i])


    plt.title("Classification of Real Injections")
    plt.xlabel("Spin z1")
    plt.ylabel("Spin z2")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # getting the directory to store the confusion matrix
    dir_list = os.listdir(folder_path)

    # initializing the exists boolean to False since we haven't found the folder yet
    exists = False

    for folder in dir_list:
        if filename in folder:
            # if there is at least one folder with this name, we set the exists boolean to True to later check if we need to create a new folder
            exists = True

        # if it's running for the first time, check for the existance of the folder for this same model and delete it if the new accuracy is better
        if first:
            if filename in folder and float(folder.split("Acc")[0]) < round(test_accuracy, 2):
                shutil.rmtree(folder_path + folder)
                os.makedirs(folder_path + str(round(test_accuracy, 2)) + filename)
                plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/spins.png")
                first = False
        else:
            plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/spins.png")
    
    if first:
        # if there isn't anything with the same name we create the new folder and save the plots
        if not exists:
            os.makedirs(folder_path + str(round(test_accuracy, 2)) + filename)
            plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/spins.png")
            first = False
    else:
        plt.savefig(folder_path + str(round(test_accuracy, 2)) + filename + "/spins.png")
        
        