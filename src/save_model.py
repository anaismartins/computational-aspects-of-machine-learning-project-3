import torch
import os

def save_model(model, test_accuracy, filename):    

    dir_list = os.listdir("../output/models/")
    
    # bool for checking if we already have this model
    exists = False
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