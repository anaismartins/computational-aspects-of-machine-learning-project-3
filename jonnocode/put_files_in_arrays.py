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

folder_path = '/home/jonno/ML_Course/ML_data/'
#folder_path = '../../../lopezm/ML_projects/Projects_2022/Project_3/Data/dataframes/' 

# Iterate over all files in the given folder
for filename in os.listdir(folder_path):
    # Check if the file is a CSV file
    if filename.endswith(".csv"):
        #Check what type of data the file contains and put in in the corresponding list.
        if filename.startswith("Blip"):
            blip_files.append(pd.read_csv(os.path.join(folder_path, filename)))
        elif filename.startswith("Fast"):
            fast_scattering_files.append(pd.read_csv(os.path.join(folder_path, filename)))            
        elif filename.startswith("Injections"):
            injection_files.append(pd.read_csv(os.path.join(folder_path, filename)))       
        elif filename.startswith("KoyFish"):
            koyfish_files.append(pd.read_csv(os.path.join(folder_path, filename)))
        elif filename.startswith("Low"):
            lowfreq_files.append(pd.read_csv(os.path.join(folder_path, filename)))
        elif filename.startswith("Tomte"):
            tomte_files.append(pd.read_csv(os.path.join(folder_path, filename)))
        elif filename.startswith("Whistle"):
            whistle_files.append(pd.read_csv(os.path.join(folder_path, filename)))       