import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_results(train_accuracies, test_accuracies, m, n_units = 0, n_layers = 0):
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrows=2, ncols=1)

    # set plot title
    if m == "perceptron":
        fig.suptitle("Perceptron")
    elif m == "variablenet":
        fig.suptitle("VariableNet " + str(n_units) + " Units " + str(n_layers) + " Layers")
    elif m == "onelayer":
        fig.suptitle("OneLayer " + str(n_units) + " Units")

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(train_accuracies)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Accuracy")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(test_accuracies)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")

    fig.align_labels()
    if m == "perceptron":
        plt.savefig("../results/" + str(round(test_accuracies[-1], 2)) + "Acc_Perceptron.png")
    elif m == "variablenet":
        plt.savefig("../results/" + str(round(test_accuracies[-1], 2)) + "Acc_VariableNet" + str(n_units) + "Units" + str(n_layers) + "Layers.png")
    elif m == "onelayer":
        plt.savefig("../results/" + str(round(test_accuracies[-1], 2)) + "Acc_OneLayer" + str(n_units) + "Units.png")