import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import shutil

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

        #clearing the plot before starting
        plt.clf()

        # plotting training accuracy
        plt.plot(train_accuracies[i])
        
        # plotting validation accuracy
        plt.plot(valid_accuracies[i])

        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")

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
        
        