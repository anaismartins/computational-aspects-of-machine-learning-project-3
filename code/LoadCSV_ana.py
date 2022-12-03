# general modules
import pandas as pd

# torch
from torch.utils.data import Dataset

class LoadCSV(Dataset):
    """
    class that reads and initializes the data
    """ 

    def __init__(self, filename, path_to_data, classification):  
        """
        runs when the object is created
        """   
        self.__data__ = pd.read_csv(path_to_data + filename) 
        self.__filename__ = str(filename)
        self.classification = classification

    def __len__(self):
        """
        returns the amount of samples in the data
        """

        return len(self.__data__)
        
    def print(self):
        """
        prints the data
        """
        print(self.__filename__ + str(self.__data__))