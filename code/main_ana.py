# general modules
import os

#torch
from torch.utils.data import DataLoader

# my modules
from LoadCSV_ana import LoadCSV
import globals_ana as g
from NeuralNetwork_ana import NeuralNetwork
        
train_data_blip = []
test_data_blip = []
train_data_injections = []
test_data_injections = []

filenames = os.listdir(g.path_to_data)

for filename in filenames:
    if ".csv" in filename:
        for classification in g.classifications:
            if classification in filename:
                if "O3a" and "Blip" in filename:
                    train_data_blip.append(LoadCSV(filename, g.path_to_data, classification))
                elif "O3b_old" and "Blip" in filename:
                    test_data_blip.append(LoadCSV(filename, g.path_to_data, classification))
                elif "O3a" and "Injections" in filename:
                    train_data_injections.append(LoadCSV(filename, g.path_to_data, classification))
                elif "O3b_old" and "Injections" in filename:
                    test_data_injections.append(LoadCSV(filename, g.path_to_data, classification))
    
#train_dataloader = DataLoader(train_data, batch_size = g.batch_size, shuffle = True)
#test_dataloader = DataLoader(test_data, batch_size = g.batch_size, shuffle = True)

model = NeuralNetwork().to(device)
print(model)