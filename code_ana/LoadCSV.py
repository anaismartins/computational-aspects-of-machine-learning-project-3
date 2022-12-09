# general modules
import pandas as pd
import numpy as np

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
        # the event ids might be messed up so we just sort it by event time
        self.__data = self.__data.sort_values(by = ["Event ID", "Event time"]) 
        
        self.__data = self.averaging()

        self.snr = self.__data[0]
        self.chisq = self.__data[1]
        self.mass_1 = self.__data[2]
        self.mass_2 = self.__data[3]
        self.spin1z = self.__data[4]
        self.spin2z = self.__data[5]

        self.dataset = np.array([])
        self.y = np.array([])

        for i in range (len(self.snr)):
            aux = np.array([self.snr[i], self.chisq[i], self.mass_1[i], self.mass_2[i], self.spin1z[i], self.spin2z[i]])
            self.dataset = np.append(self.dataset, aux)
            if classification == "Blip":
                self.y = np.append(self.y, 1)
            elif classification == "Injections":
                self.y = np.append(self.y, 0)

    def __len__(self):
        """
        returns the amount of samples in the data
        """

        return len(self.snr)
        
    def __str__(self):
        """
        prints the data
        """
        return (str(self.snr))
    
    def averaging(self):
        blip_sorted = np.array(self.__data.sort_values(by = ['Event ID', 'Event time']).to_numpy())
        event_amount = len(blip_sorted)

        ev_snr = np.array([])
        av_snr = np.array([])
        ev_chisq = np.array([])
        av_chisq = np.array([])
        ev_m1 = np.array([])
        av_m1 = np.array([])
        ev_m2 = np.array([])
        av_m2 = np.array([])
        ev_s1 = np.array([])
        av_s1 = np.array([])
        ev_s2 = np.array([])
        av_s2 = np.array([])
        prev_evtime = blip_sorted[0][4]

        for i in range(event_amount):
            event_id = int(blip_sorted[i][3])

            current_evtime = blip_sorted[i][4]
            if current_evtime == prev_evtime:
                ev_snr = np.append(ev_snr, blip_sorted[i][5])
                ev_chisq = np.append(ev_chisq, blip_sorted[i][6])
                ev_m1 = np.append(ev_m1, blip_sorted[i][7])
                ev_m2 = np.append(ev_m2, blip_sorted[i][8])
                ev_s1 = np.append(ev_s1, blip_sorted[i][9])
                ev_s2 = np.append(ev_s2, blip_sorted[i][10])

                #print('Appending event' ,current_evtime, 'm1', ev_m1)
            else: 
                if len(ev_snr) ==0:
                    av_snr = np.append(av_snr, blip_sorted[i][5])
                    av_chisq = np.append(av_chisq, blip_sorted[i][6])
                    av_m1 = np.append(av_m1, blip_sorted[i][7])
                    av_m2 = np.append(av_m2, blip_sorted[i][8])
                    av_s1 = np.append(av_s1, blip_sorted[i][9])
                    av_s2 = np.append(av_s2, blip_sorted[i][10])
                else:
                    av_snr = np.append(av_snr, np.average(ev_snr))
                    av_chisq = np.append(av_chisq, np.average(ev_chisq))
                    av_m1 = np.append(av_m1, np.average(ev_m1))
                    av_m2 = np.append(av_m2, np.average(ev_m2))
                    av_s1 = np.append(av_s1, np.average(ev_s1))
                    av_s2 = np.append(av_s2, np.average(ev_s2))    

                ev_snr = np.array([])
                ev_chisq = np.array([])
                ev_m1 = np.array([])
                ev_m2 = np.array([])
                ev_s1 = np.array([])
                ev_s2 = np.array([])

                #print('Event time', prev_evtime, 'SNR', av_snr, 'Chisq', av_chisq, 'M1', av_m1, 'M2', av_m2, 'S1', av_s1, 'S2', av_s2)
                prev_evtime = current_evtime

        triggers = (av_snr, av_chisq, av_m1, av_m2, av_s1, av_s2)
                
        return triggers