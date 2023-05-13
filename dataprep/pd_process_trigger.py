import numpy as np
import pandas as pd
import os
import argparse

def classTag(file_name):
    if 'Injections' in file_name:
        t = 0
    if 'Blip' in file_name:
        t = 1
    if 'Koi_Fish' in file_name:
        t = 2
    if 'Low_Frequency_Burst' in file_name:
        t = 3
    if 'Tomte' in file_name:
        t = 4
    if 'Whistle' in file_name:
        t = 5
    if 'Fast_Scattering' in file_name:
        t = 6
    return t


def averageFeature(cols, tmp, t):
    """
       We average the clusters weighting them by the SNR
    Input
    -----
    cols: (list) columns for feature Data Frame
    tmp: (Data Frame) cluster of features to average
    t: (int) class tag
    Output
    ------
    feature: (Data Frame) averaged cluster
    """
    cols.extend(['Class'])
    # Weight by the SNR
    weights = tmp['SNR'] / np.max(tmp['SNR'])

    # Compute average and return a DataFrame
    feature = np.zeros((1, len(cols)))
    for c in range(len(cols)-1):
        feature[0, c] = (tmp[cols[c]] * weights).sum() / weights.sum()
    # T
    feature[0, len(cols) - 1] = int(t)
    feature = pd.DataFrame(feature, columns=cols)

    return feature


def iterateClusters(file_name, path):
    """
        We iterate over the clusters of triggers.
    Input
    -----
    file_name: file to read
    path: path where the file is located

    Output
    ------
    None, the function stores two DataFrames
    features: contains the feature arrays to feed in the method
    IDs: contains the IDs of each cluster to follow up the methods results
    """

    # Load data and group by
    classes = classTag(file_name)
    df = pd.read_csv(path + file_name)
    df_ = df.groupby(['Cluster ID', 'Cluster time'])

    drop_cols = ['Unnamed: 0', 'ifo', 'Trigger ID', 'Trigger time',
                 'Cluster ID', 'Cluster time', 'Template ID']
    dest = '/home/melissa.lopez/GlitchBank/LargeAnalysis/O3aGB/AvDataFrames/'  # FIXME
    cluster_id, cluster_time = list(), list()
    k = 0

    for group_name, df_group in df_:

        # Store Cluster ID and Cluster time
        cluster_id.append(group_name[0])
        cluster_time.append(group_name[1])

        # Drop irrelevant columns
        tmp = df_group.drop(drop_cols, axis=1)

        feature = averageFeature(list(tmp.columns), tmp, classes)

        if k == 0:
            features = feature
        else:
            features = pd.concat([features, feature])
        features = features.loc[:, ~features.columns.str.contains('^Unnamed')]
        k = k + 1

    features = features.reset_index(drop=True)
    features = features.loc[:, ~features.columns.str.contains('^Unnamed')]
    features.to_csv(dest + 'Av_' + file_name[:-4] + '.csv', index=False)

    IDs = pd.DataFrame({'Cluster ID': cluster_id,
                        'Cluster time': cluster_time})
    IDs = IDs.loc[:, ~IDs.columns.str.contains('^Unnamed')]
    IDs.to_csv(dest + 'Av_' + file_name[:-4] + '_ID.csv', index=False)


parser = argparse.ArgumentParser()
parser.add_argument('--folder', metavar='1', type=str,
                    help='Path to read data')
parser.add_argument('--detector', metavar='1', type=str,
                    help='Name of detector')

# Define arguments
args = parser.parse_args()
folder_path = args.folder
detector = args.detector

file_list = np.sort(os.listdir(folder_path))
for file_name in file_list:
    if detector in file_name:
        print(file_name)
        iterateClusters(file_name, folder_path)