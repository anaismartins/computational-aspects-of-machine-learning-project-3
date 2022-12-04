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
        self.__data = pd.read_csv(path_to_data + filename) 
        self.__data = self.__data.sort_values(by = ["Event ID", "Event time"]) 

        self.__filename = str(filename)
        self.classification = classification

        # getting the individual columns
        self.__unnammed = self.__data["Unnamed: 0"]
        self.__trigger_id = self.__data["Trigger ID"]
        self.__trigger_time = self.__data["Trigger time"]
        self.__event_id = self.__data["Event ID"]
        self.__event_time = self.__data["Event time"]
        self.__snr = self.__data["SNR"]
        self.__chisq = self.__data["Chisq"]
        self.__mass_1 = self.__data["Mass_1"]
        self.__mass_2 = self.__data["Mass_2"]
        self.__spin1z = self.__data["Spin1z"]
        self.__spin2z = self.__data["Spin2z"]

    def __len__(self):
        """
        returns the amount of samples in the data
        """

        return len(self.__data)
        
    def __str__(self):
        """
        prints the data
        """
        return str(self.__data)