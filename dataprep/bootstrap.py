import random
import numpy as np

# Set the detector to use
detector = "H1"

# Load the data
injections = np.load("../datasets/injection_triggers_" + detector + ".npy").tolist()
blips = np.load("../datasets/blip_triggers_" + detector + ".npy").tolist()
koyfishes = np.load("../datasets/koyfish_triggers_" + detector + ".npy").tolist()
lowfreqs = np.load("../datasets/lowfreq_triggers_" + detector + ".npy").tolist()
tomtes = np.load("../datasets/tomte_triggers_" + detector + ".npy").tolist()
whistles = np.load("../datasets/whistle_triggers_" + detector + ".npy").tolist()
if detector != "V1":
    fast_scatterings = np.load("../datasets/fast_scattering_triggers_" + detector + ".npy").tolist()

# Find the biggest dataset
if detector != "V1":
    biggest = max(len(blips), len(fast_scatterings), len(koyfishes), len(lowfreqs), len(tomtes), len(whistles))
else:
    biggest = max(len(blips), len(koyfishes), len(lowfreqs), len(tomtes), len(whistles))

def bootstrap(glitch):
    """
    Bootstraps the data to make it the same size as the biggest dataset
    :param glitch: the dataset to be bootstrapped
    :return: the bootstrapped dataset
    """
    # setting the number of rows to the size of the biggest dataset rounded to the hundreds
    n_rows = round(biggest/100)*100

    rows = [[]]
    aux = []

    # giving rows the size it needs for the sampling
    for i in range(n_rows):
        for j in range(7):
            aux.append(0)
        rows.append(aux)
        aux = []

    for it in range(0, 100):
        # sampling the data 100 times
        rows[round(biggest/100)*it:round(biggest/100)*(it + 1)] = random.sample(glitch, round(biggest/100))

    rows.pop(-1)

    return rows

blip_boot = bootstrap(blips)
koyfish_boot = bootstrap(koyfishes)
lowfreq_boot = bootstrap(lowfreqs)
tomte_boot = bootstrap(tomtes)
whistle_boot = bootstrap(whistles)
if detector != "V1":
    fast_scattering_boot = bootstrap(fast_scatterings)

# making the injection dataset the same size as the biggest dataset
injection_boot = injections[0:round(biggest/100)*100]

# joining the separate datasets for each class into one
dataset = np.append(injection_boot, blip_boot, axis = 0)
dataset = np.append(dataset, koyfish_boot, axis = 0)
dataset = np.append(dataset, lowfreq_boot, axis = 0)
dataset = np.append(dataset, tomte_boot, axis = 0)
dataset = np.append(dataset, whistle_boot, axis = 0)
if detector != "V1":
    dataset = np.append(dataset, fast_scattering_boot, axis = 0)

# shuffling the dataset and saving it
np.random.shuffle(dataset)
print("Dataset size: " + str(dataset.shape))
np.save('../datasets/dataset_all_' + detector + '_bootstrap.npy', dataset)