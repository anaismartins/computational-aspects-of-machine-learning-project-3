import torch
import os

def save_model(model, m, detector, n_units, n_units2, n_units3, n_layers, a, l, o, lr, epochs, num_batches, test_accuracy):    

    dir_list = os.listdir("../output/models/")
    
    # bool for checking if we already have this model
    exists = False

    if m == "Perceptron":
        filename = "Acc_" + detector + "_Perceptron_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.pth"
    elif m == "VariableNet":
        filename = "Acc_" + detector + "_VariableNet" + str(n_units) + "Units" + str(n_layers) + "Layers_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.pth"
    elif m == "OneLayer":
        filename = "Acc_" + detector + "_OneLayer" + str(n_units) + "Units_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.pth"
    elif m == "TwoLayers":
        filename = "Acc_" + detector + "_TwoLayers" + str(n_units) + "_" + str(n_units2) + "Units_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.pth"
    elif m == "ThreeLayers":
        filename = "Acc_" + detector + "_ThreeLayers" + str(n_units) + "_" + str(n_units2) + "_" + str(n_units3) + "Units_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches.pth"


        # check if there is this model already and delete it
    for file in dir_list:
        if filename in file:
            exists = True

        if filename in file and float(file.split("Acc")[0]) < round(test_accuracy, 2):
            os.remove("../output/models/" + file)
            torch.save(model.state_dict(), "../output/models/" + str(round(test_accuracy, 2)) + filename)

    if not exists:
        torch.save(model.state_dict(), "../output/models/" + str(round(test_accuracy, 2)) + filename)