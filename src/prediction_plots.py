import matplotlib.pyplot as plt
import numpy as np
import os
import shutil

def prediction_plots(X_test, y_test, av_pred_labels, num_classes, filename, test_accuracy, binary):

    X_real_injections = []
    y_real_injections = []

    for i in range(0, len(y_test)):
        if y_test[i] == 0:
            X_real_injections.append(X_test[i])
            y_real_injections.append(av_pred_labels[i])

    m_1 = []
    m_2 = []
    s_1 = []
    s_2 = []

    for i in range(0, len(y_real_injections)):
        m_1.append(X_real_injections[i][2])
        m_2.append(X_real_injections[i][3])
        s_1.append(X_real_injections[i][4])
        s_2.append(X_real_injections[i][5])


    # use colormap
    colormap = np.array(['#d73027', '#fc8d59', '#fee090', '#ffffbf', '#e0f3f8', '#91bfdb', '#4575b4'])
    labels = np.array(["Injection", "Blips", "Koyfish", "Low Freq Burst", "Tomte", "Whistle", "Fast Scat"])

    fig = plt.figure()
    ax = plt.subplot(111)

    for i in range(0, num_classes):
        x = []
        y = []
        for j in range(0, len(y_real_injections)):
            if y_real_injections[j] == i:
                x.append(m_1[j])
                y.append(m_2[j])

        ax.scatter(x, y, s=50, c=colormap[i], label = labels[i])

    plt.title("Classification of Real Injections")
    plt.xlabel("Mass 1")
    plt.ylabel("Mass 2")

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if not binary:
        folder_path = "../output/pred_plots/"
    else:
        folder_path = "../output/pred_plots/binary/"

    # getting the directory to store the confusion matrix
    dir_list = os.listdir(folder_path)

    # initializing the exists boolean to False since we haven't found the folder yet
    exists = False
    first = True

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