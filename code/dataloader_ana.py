import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

path_to_data = "../../../lopezm/ML_projects/Projects_2022/Project_3/Data/dataframes/"

blip_h1_o3a = pd.read_csv(path_to_data + "Blip_H1_O3a.csv")

blip_h1_o3a = blip_h1_o3a.sort_values(by=["Event ID"])
print(blip_h1_o3a)