from torch.utils.data import Dataset

class LoadCSV(Dataset):
    """
    class that reads and initializes the data
    """ 

    def __init__(self, filename, path_to_data):  
        """
        initialize the data
        """   
        self.data = pd.read_csv(path_to_data + filename) 
        self.filename = str(filename)
        
    def print(self):
        """
        print the data
        """
        print(self.filename + str(self.data))
