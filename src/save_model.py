import torch
import os

def save_model(model, test_accuracy, filename):
    """
    function that saves the model in the output folder
    :param model: the model (string of the model name)
    :param test_accuracy: the accuracy of the model
    :param filename: name to save the model as
    """    

    dir_list = os.listdir("../output/models/")
    
    # bool for checking if we already have this model
    exists = False
    # adding the file extension to the filname
    filename = filename + ".pth"

        # check if there is this model already and delete it
    for file in dir_list:
        if filename in file:
            exists = True

        if filename in file and float(file.split("Acc")[0]) < round(test_accuracy, 2):
            os.remove("../output/models/" + file)
            torch.save(model.state_dict(), "../output/models/" + str(round(test_accuracy, 2)) + filename)

    if not exists:
        torch.save(model.state_dict(), "../output/models/" + str(round(test_accuracy, 2)) + filename)