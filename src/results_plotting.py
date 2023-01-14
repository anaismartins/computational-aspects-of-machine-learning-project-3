import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# my modules
import globals as g

def plot_results(train_accuracies, valid_accuracies, test_accuracy, m, a, l, o, lr, epochs, n_units = 0, n_layers = 0, detector = "H1", num_batches = 20):
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrows=2, ncols=1)

    # set plot title
    if m == "Perceptron":
        fig.suptitle("Perceptron")
    elif m == "VariableNet":
        fig.suptitle("VariableNet " + str(n_units) + " Units " + str(n_layers) + " Layers")
    elif m == "OneLayer":
        fig.suptitle("OneLayer " + str(n_units) + " Units")

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(train_accuracies)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Accuracy")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(valid_accuracies)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")

    fig.align_labels()

    dir_list = os.listdir("../results/")

    if m == "Perceptron":
        # bool for checking if we already have this model
        exists = False
        filename = "Acc_" + detector + "_Perceptron_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.png"

        # check if there is a perceptron already and delete it
        for file in dir_list:
            if filename in file:
                exists = True

            if filename in file and float(file.split("Acc")[0]) < round(test_accuracy, 2):
                os.remove("../results/" + file)
                plt.savefig("../results/" + str(round(test_accuracy, 2)) + filename)

        if not exists:
            plt.savefig("../results/" + str(round(test_accuracy, 2)) + filename)

        print("Perceptron results saved to results folder. (If it already existed, it was overwritten.)")

    elif m == "VariableNet":
        # bool for checking if we already have this model
        exists = False
        filename = "Acc_" + detector + "_VariableNet" + str(n_units) + "Units" + str(n_layers) + "Layers_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.png"

        # check if there is a variable net with the same specifications already and delete it
        for file in dir_list:
            if filename in file:
                exists = True

            if filename in file and float(file.split("Acc")[0]) < round(test_accuracy, 2):
                os.remove("../results/" + file)
                plt.savefig("../results/" + str(round(test_accuracy, 2)) + filename)
    
        if not exists: 
            plt.savefig("../results/" + str(round(test_accuracy, 2)) + filename)

        print("VariableNet results saved to results folder. (If it already existed, it was overwritten.)")

    elif m == "OneLayer":
        # bool for checking if we already have this model
        exists = False
        filename = "Acc_" + detector + "_OneLayer" + str(n_units) + "Units_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.png"

        # check if there is a one layer net with the same specifications already and delete it
        for file in dir_list:
            if filename in file:
                exists = True

            if filename in file and float(file.split("Acc")[0]) < round(test_accuracy, 2):
                os.remove("../results/" + file)
                plt.savefig("../results/" + str(round(test_accuracy, 2)) + filename)

        if not exists:
            plt.savefig("../results/" + str(round(test_accuracy, 2)) + filename)

        print("OneLayer results saved to results folder. (If it already existed, it was overwritten.)")