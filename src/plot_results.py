import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def plot_results(train_accuracies, valid_accuracies, test_accuracy, k, filename):
    """
    function that plots the results of the training and validation accuracy and saves it in the output folder
    :param train_accuracies: the training accuracies
    :param valid_accuracies: the validation accuracies
    :param test_accuracy: the test accuracy
    :param k: the number of folds
    :param filename: the name to give to the saved folder
    """

    # specs for the plot
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrows=2, ncols=1)

    # later on to check if the folder with this name was created in this run
    first = True

    # plotting the results for each fold
    for i in range(0, k):
        # plotting training accuracy
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(train_accuracies[i])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Accuracy")

        # plotting validation accuracy
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(valid_accuracies[i])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")

        fig.align_labels()

        dir_list = os.listdir("../output/results/")

        # initializing the exists boolean to False since we haven't found the folder yet
        exists = False

        for folder in dir_list:
            if filename in folder:
                # if there is at least one folder with this name, we set the exists boolean to True to later check if we need to create a new folder
                exists = True

            # if it's running for the first time, check for the existance of the folder for this same model and delete it if the new accuracy is better
            if first:
                if filename in folder and float(folder.split("Acc")[0]) < round(test_accuracy, 2):
                    os.remove("../output/results/" + folder)
                    os.makedirs("../output/results/" + str(round(test_accuracy, 2)) + filename)
                    plt.savefig("../output/results/" + str(round(test_accuracy, 2)) + filename + "/" + str(i) + ".png")
                    first = False
            else:
                plt.savefig("../output/results/" + str(round(test_accuracy, 2)) + filename + "/" + str(i) + ".png")
        
        if first:
            # if there isn't anything with the same name we create the new folder and save the plots
            if not exists:
                os.makedirs("../output/results/" + str(round(test_accuracy, 2)) + filename)
                plt.savefig("../output/results/" + str(round(test_accuracy, 2)) + filename + "/" + str(i) + ".png")
                first = False
        else:
            plt.savefig("../output/results/" + str(round(test_accuracy, 2)) + filename + "/" + str(i) + ".png")
        
        #clearing the plot for the next run
        plt.clf()