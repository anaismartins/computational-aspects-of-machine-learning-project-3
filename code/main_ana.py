# general modules
import os

#torch
from torch.utils.data import DataLoader

# my modules
from LoadCSV_ana import LoadCSV
import globals_ana as g
        
data = []

filenames = os.listdir(path_to_data)

for filename in filenames:
    if ".csv" in filename:
        data.append(LoadCSV(filename, g.path_to_data))
    
dataloader = DataLoader(data, batch_size = g.batch_size, shuffle = True)