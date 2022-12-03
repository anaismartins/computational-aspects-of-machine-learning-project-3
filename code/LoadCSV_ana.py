# general modules
import pandas as pd

# torch
from torch.utils.data import Dataset

class LoadCSV(Dataset):
    """
    class that reads and initializes the data
    """ 

    def __init__(self, filename, path_to_data):  
        """
        runs when the object is created
        """   
        self.data = pd.read_csv(path_to_data + filename) 
        self.filename = str(filename)

    def __len__(self):
        """
        returns the amount of samples in the data
        """

        return len(self.data)
        
    def __print__(self):
        """
        prints the data
        """
        print(self.filename + str(self.data))
