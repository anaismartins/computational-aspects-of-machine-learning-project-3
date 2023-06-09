import random
import numpy as np
import pandas as pd
import argparse


def bootstrap(glitch, biggest):
    """
    Bootstraps the data to make it the same size as the biggest dataset
    :param glitch: the dataset to be bootstrapped
    :return: the bootstrapped dataset
    """
    # setting the number of rows to the size of the
    # biggest dataset rounded to the hundreds
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
        up = round(biggest / 100) * it
        down = round(biggest / 100) * (it + 1)
        rows[up:down] = random.sample(glitch, round(biggest/100))
    rows.pop(-1)

    return rows


def dataFrametoList(path_to_file):
    data = pd.read_csv(path_to_file)
    data = data.values.tolist()
    return data


parser = argparse.ArgumentParser()
parser.add_argument('--path', metavar='1', type=str,
                    help='Path to read data')
parser.add_argument('--run', metavar='1', type=str,
                    help='O3a or O3b')
parser.add_argument('--detector', metavar='1', type=str,
                    help='Name of detector')

# Define arguments
args = parser.parse_args()
path = args.path
run = args.run
detector = args.detector

injections = dataFrametoList(path + 'Av_Injections_' + detector + '_' + run + '.csv')
blips = dataFrametoList(path + 'Av_Blip_' + detector + '_' + run + '.csv')
koyfishes = dataFrametoList(path + 'Av_Koi_Fish_' + detector + '_' + run + '.csv')
lowfreqs = dataFrametoList(path + 'Av_Low_Frequency_Burst_' + detector + '_' + run + '.csv')
tomtes = dataFrametoList(path + 'Av_Tomte_' + detector + '_' + run + '.csv')
whistles = dataFrametoList(path + 'Av_Whistle_' + detector + '_' + run + '.csv')
if detector != "V1":
    fast_scatterings = dataFrametoList(path + 'Av_Fast_Scattering_' + detector + '_' + run + '.csv')

# Find the biggest dataset
if detector != "V1":
    biggest = max(len(blips), len(fast_scatterings),
                  len(koyfishes), len(lowfreqs),
                  len(tomtes), len(whistles))
else:
    biggest = max(len(blips), len(koyfishes),
                  len(lowfreqs), len(tomtes),
                  len(whistles))

blip_boot, koyfish_boot = bootstrap(blips, biggest), bootstrap(koyfishes, biggest)
lowfreq_boot, tomte_boot = bootstrap(lowfreqs, biggest), bootstrap(tomtes, biggest)
whistle_boot = bootstrap(whistles, biggest)
if detector != "V1":
    fast_scattering_boot = bootstrap(fast_scatterings, biggest)

# making the injection dataset the same size as the biggest dataset
injection_boot = injections[0:round(biggest / 100) * 100]

# joining the separate datasets for each class into one
dataset = np.vstack([injection_boot, blip_boot,
                     koyfish_boot, lowfreq_boot,
                     tomte_boot, whistle_boot])
if detector != "V1":
    dataset = np.vstack([dataset, fast_scattering_boot])

# shuffling the dataset and saving it
np.random.shuffle(dataset)
print("Dataset size: " + str(dataset.shape))
np.save(path + 'dataset_all_' + detector + '_bootstrap_' + run + '.npy', dataset)
