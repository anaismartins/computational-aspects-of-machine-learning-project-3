import numpy as np
import pandas as pd
import os

blip_files = []
fast_scattering_files = []
injection_files = []
koyfish_files = []
lowfreq_files = []
tomte_files = []
whistle_files = []

#folder_path = '/home/jonno/ML_Course/ML_data/'
folder_path = "/data/gravwav/lopezm/ML_projects/Projects_2022/Project_3/Data/dataframes/" 

# Iterate over all files in the given folder
for filename in os.listdir(folder_path):
    # Check if the file is a CSV file
    if filename.endswith(".csv"):
        #Check what type of data the file contains and put in in the corresponding list.
        if filename.startswith("Blip_H1"):
            blip_files.append((os.path.join(folder_path, filename)))
        elif filename.startswith("Fast_Scattering_H1"):
            fast_scattering_files.append((os.path.join(folder_path, filename)))            
        elif filename.startswith("Fast_Scattering_H1"):
            fast_scattering_files.append((os.path.join(folder_path, filename)))            
        elif filename.startswith("Injections_H1"):
            injection_files.append((os.path.join(folder_path, filename)))       
        elif filename.startswith("KoyFish_H1"):
            koyfish_files.append((os.path.join(folder_path, filename)))
        elif filename.startswith("Low_freq_burst_H1"):
            lowfreq_files.append((os.path.join(folder_path, filename)))
        elif filename.startswith("Tomte_H1"):
            tomte_files.append((os.path.join(folder_path, filename)))
        elif filename.startswith("Whistle_H1"):
            whistle_files.append((os.path.join(folder_path, filename)))    
        elif filename.startswith("Low_freq_burst_H1"):
            lowfreq_files.append((os.path.join(folder_path, filename)))
        elif filename.startswith("Tomte_H1"):
            tomte_files.append((os.path.join(folder_path, filename)))
        elif filename.startswith("Whistle_H1"):
            whistle_files.append((os.path.join(folder_path, filename)))    

#Now this function must be edited so that it reads the csv files from e.g. the koyfish_files array
def read_triggers(read_paths, trigger_id, weighted_average = False):
    """
    Function that reads the CSV trigger files, averages them and converts it to the right array format.
    """
    
    all_triggers = np.array([[False]])
    
    for path in read_paths:
        # For each path we want to read the CSV
        
        path_csv = pd.read_csv(path)

        path_np = path_csv.to_numpy()

        data_sorted = np.array(path_csv.sort_values(by = ['Event ID', 'Event time']))
        event_amount = len(data_sorted)
        
        if path == read_paths[0]:
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

        else: 
            ev_snr = np.array([])
            ev_chisq = np.array([])
            ev_m1 = np.array([])
            ev_m2 = np.array([])
            ev_s1 = np.array([])
            ev_s2 = np.array([])
        # Initialize the first event time
        prev_evtime = data_sorted[0][4]

        
        # Now we will loop over all the event triggers 
        for i in range(event_amount):
            event_id = int(data_sorted[i][3])
            current_evtime = data_sorted[i][4]
            
            # Collect the triggers that belong to the same event.
            if current_evtime == prev_evtime:
                ev_snr = np.append(ev_snr, data_sorted[i][5])
                ev_chisq = np.append(ev_chisq, data_sorted[i][6])
                ev_m1 = np.append(ev_m1, data_sorted[i][7])
                ev_m2 = np.append(ev_m2, data_sorted[i][8])
                ev_s1 = np.append(ev_s1, data_sorted[i][9])
                ev_s2 = np.append(ev_s2, data_sorted[i][10])

            else: 
                # If the evnet only has one trigger, no averaging.
                if len(ev_snr) ==0:
                    triggers = np.array([[data_sorted[i][5], data_sorted[i][6], data_sorted[i][7],
                                       data_sorted[i][8],data_sorted[i][9], data_sorted[i][10], trigger_id]])
                # Otherwise we have multiple triggers and we need to average over them.
                else:
                    # Here we perform the weighted average
                    if weighted_average == True:
                        weights = ev_snr / np.max(ev_snr)
                        triggers = np.array([[np.average(ev_snr, weights = weights), np.average(ev_chisq, weights = weights),
                                              np.average(ev_m1, weights = weights), np.average(ev_m2, weights = weights),
                                              np.average(ev_s1, weights = weights), np.average(ev_s2, weights = weights), 
                                              trigger_id]]) 
                    else: 
                        triggers = np.array([[np.average(ev_snr), np.average(ev_chisq), np.average(ev_m1),
                                        np.average(ev_m2), np.average(ev_s1), 
                                          np.average(ev_s2), trigger_id]]) 
                
                ev_snr = np.array([])
                ev_chisq = np.array([])
                ev_m1 = np.array([])
                ev_m2 = np.array([])
                ev_s1 = np.array([])
                ev_s2 = np.array([])

                prev_evtime = current_evtime
                # Collect all the averaged triggers together
                if all_triggers[0][0] == False:
                    all_triggers = triggers
                else:
                    all_triggers = np.append(all_triggers, triggers, axis = 0)
    
    return all_triggers




injection_triggers = read_triggers(injection_files, 0, weighted_average = True)
blip_triggers = read_triggers(blip_files, 1, weighted_average = True)
#fast_scattering_triggers = read_triggers(fast_scattering_files, 2)
koyfish_triggers = read_triggers(koyfish_files, 2)
#lowfreq_triggers = read_triggers(lowfreq_files, 4)
tomte_triggers = read_triggers(tomte_files, 5)
#whistle_triggers = read_triggers(whistle_files, 6)

num_triggers = tomte_triggers.shape[0]

injection_triggers = injection_triggers[0:num_triggers]
blip_triggers = blip_triggers[0:num_triggers]
koyfish_triggers = koyfish_triggers[0:num_triggers]
#lowfreq_triggers = lowfreq_triggers[0:fast_scattering_triggers.shape[0]]
tomte_triggers = tomte_triggers[0:num_triggers]
#whistle_triggers = whistle_triggers[0:fast_scattering_triggers.shape[0]]

dataset = np.append(injection_triggers, blip_triggers, axis = 0)
#dataset = np.append(dataset, fast_scattering_triggers, axis = 0)
dataset = np.append(dataset, koyfish_triggers, axis = 0)
#dataset = np.append(dataset, lowfreq_triggers, axis = 0)
dataset = np.append(dataset, tomte_triggers, axis = 0)
#dataset = np.append(dataset, whistle_triggers, axis = 0)
np.random.shuffle(dataset)

np.save('dataset_blip_injection_koyfish_tomte_weighted_average_h1.npy', dataset)