# general modules
import os

#torch
from torch.utils.data import DataLoader

# my modules
from LoadCSV_ana import LoadCSV
import globals_ana as g
        
train_data = []
test_data = []

filenames = os.listdir(g.path_to_data)

for filename in filenames:
    if ".csv" in filename:
        for classification in g.classifications:
            if classification in filename:
                if "O3a" in filename:
                    train_data.append(LoadCSV(filename, g.path_to_data, classification))
                elif "O3b_old" in filename:
                    test_data.append(LoadCSV(filename, g.path_to_data, classification))

print(train_data[0].classification)
    
train_dataloader = DataLoader(train_data, batch_size = g.batch_size, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = g.batch_size, shuffle = True)