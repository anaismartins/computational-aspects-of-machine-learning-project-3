import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# my modules
import globals as g

def plot_results(train_accuracies, valid_accuracies, test_accuracy, k, m, a, l, o, lr, epochs, n_units = 0, n_units2 = 0, n_units3 = 0, n_layers = 0, detector = "H1", num_batches = 20):
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrows=2, ncols=1)

    # set plot title
    if m == "Perceptron":
        fig.suptitle("Perceptron")
    elif m == "VariableNet":
        fig.suptitle("VariableNet " + str(n_units) + " Units " + str(n_layers) + " Layers")
    elif m == "OneLayer":
        fig.suptitle("OneLayer " + str(n_units) + " Units")

    first = True

    for i in range(0, k):
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(train_accuracies[i])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training Accuracy")

        ax = fig.add_subplot(gs[1, 0])
        ax.plot(valid_accuracies[i])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Accuracy")

        fig.align_labels()

        dir_list = os.listdir("../output/results/")

        if m == "Perceptron":
            filename = "Acc_" + detector + "_Perceptron_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches"
        elif m == "VariableNet":
            filename = "Acc_" + detector + "_VariableNet" + str(n_units) + "Units" + str(n_layers) + "Layers_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches"
        elif m == "OneLayer":
            filename = "Acc_" + detector + "_OneLayer" + str(n_units) + "Units_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches"
        elif m == "TwoLayers":
            filename = "Acc_" + detector + "_TwoLayers" + str(n_units) + "_" + str(n_units2) + "Units_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches"
        elif m == "ThreeLayers":
            filename = "Acc_" + detector + "_ThreeLayers" + str(n_units) + "_" + str(n_units2) + "_" + str(n_units3) + "Units_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches"

        exists = False


        for folder in dir_list:
            if filename in folder:
                exists = True

            if first:
                if filename in folder and float(folder.split("Acc")[0]) < round(test_accuracy, 2):
                    os.remove("../output/results/" + folder)
                    os.makedirs("../output/results/" + str(round(test_accuracy, 2)) + filename)
                    plt.savefig("../output/results/" + str(round(test_accuracy, 2)) + filename + "/" + str(i) + ".png")
                    first = False
            else:
                plt.savefig("../output/results/" + str(round(test_accuracy, 2)) + filename + "/" + str(i) + ".png")
        if first:
            if not exists:
                os.makedirs("../output/results/" + str(round(test_accuracy, 2)) + filename)
                plt.savefig("../output/results/" + str(round(test_accuracy, 2)) + filename + "/" + str(i) + ".png")
                first = False
        else:
            plt.savefig("../output/results/" + str(round(test_accuracy, 2)) + filename + "/" + str(i) + ".png")
        
        ax.clear()