import pandas as pd
import matplotlib.pyplot as plt
import torch

csv_file = pd.read_csv('/home/jonno/ML_Course/ML_data/Blip_H1_O3a.csv')

print(csv_file.keys())
print(csv_file)

csv_file.plot(kind="scatter", x="Mass_1", y="Mass_2")
plt.show()


