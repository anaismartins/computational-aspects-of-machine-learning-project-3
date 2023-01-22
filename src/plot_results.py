import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def plot_results(train_accuracies, valid_accuracies, test_accuracy, k, filename):
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(nrows=2, ncols=1)

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