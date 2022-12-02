import pandas as pd
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

path_to_data = "../../../lopezm/ML_projects/Projects_2022/Project_3/Data/dataframes/"

class Data(Dataset):
    """
    class that reads and initializes the data
    """

    def __init__(self, filename, path_to_data):
        """
        initialization of the object
        """
        self.data = pd.read_csv(path_to_data + filename)
        self.filename = str(filename)
    
    def print(self):
        """
        printing the data
        """
        print(self.filename + str(self.data))
        
        
data = []
i = 0

filenames = os.listdir(path_to_data)

for filename in filenames:
    if ".csv" in filename:
        data.append(Data(filename, path_to_data))
    
dataloader = DataLoader(data, batch_size=1, shuffle=True)

print(dataloader)