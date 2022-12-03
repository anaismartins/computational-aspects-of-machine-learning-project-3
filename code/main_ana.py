import os

from torch.utils.data import DataLoader

from LoadCSV_ana import LoadCSV

path_to_data = "../../../lopezm/ML_projects/Projects_2022/Project_3/Data/dataframes/" 
        
data = []
i = 0

filenames = os.listdir(path_to_data)

for filename in filenames:
    if ".csv" in filename:
        data.append(LoadCSV(filename, path_to_data))
    
dataloader = DataLoader(data, batch_size=1, shuffle=True)