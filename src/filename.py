def filename(m, detector, a, l, o, lr, epochs, num_batches, n_layers, n_units, n_units2, n_units3, n_units4):
    """
    Creates the filename for the saved model and the saved results
    :param m: model (string with the name)
    :param detector: detector
    :param a: activation function (string with the name)
    :param l: loss function (string with the name)
    :param o: optimizer (string with the name)
    :param lr: inital set learning rate
    :param epochs: number of epochs that the model took
    :param num_batches: number of batches
    :param n_layers: number of layers
    :param n_units: number of units in the first layer
    :param n_units2: number of units in the second layer
    :param n_units3: number of units in the third layer
    :return: filename (string)
    """

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
    elif m == "FourLayers":
        filename = "Acc_" + detector + "_FourLayers" + str(n_units) + "_" + str(n_units2) + "_" + str(n_units3) + "_" + str(n_units4) + "Units_" + a + "_" + l + "_" + o + "_" + str(lr) + "lr_" + str(epochs) + "epochs_" + str(num_batches) + "batches"

    return filename