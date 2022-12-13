import numpy as np
import pandas as pd

path1 = '/data/gravwav/lopezm/ML_projects/Projects_2022/Project_3/Data/dataframes/Blip_H1_O3a.csv'
path2 = '/data/gravwav/lopezm/ML_projects/Projects_2022/Project_3/Data/dataframes/Blip_H1_O3b_old.csv'
path3 =  '/data/gravwav/lopezm/ML_projects/Projects_2022/Project_3/Data/dataframes/Injections_H1_O3a.csv'


blip_paths= [path1, path2]
injection_paths = [path3]



def read_triggers(read_paths, trigger_id):
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
                                       data_sorted[i][8],data_sorted[i][9], data_sorted[i][10], trigger_id[0], trigger_id[1]]])
                # If we have multiple triggers, we need to average over them.
                else:
                    triggers = np.array([[np.average(ev_snr), np.average(ev_chisq), np.average(ev_m1),
                                        np.average(ev_m2), np.average(ev_s1), 
                                          np.average(ev_s2), trigger_id[0], trigger_id[1]]]) 
                
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

blip_triggers = read_triggers(blip_paths, [0,1])
injection_triggers = read_triggers(injection_paths, [1,0])

np.random.shuffle(injection_triggers)

injection_triggers = injection_triggers[0:blip_triggers.shape[0]]

dataset = np.append(injection_triggers, blip_triggers, axis = 0)
np.random.shuffle(dataset)

np.save('dataset_inj_blip.npy',dataset)