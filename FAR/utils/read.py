import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(1, '../')
#sys.path.insert(1, '../../computational-aspects-of-machine-learning-project-3')
from src.torch_utils import load_model, predictions
from FAR.utils.measure import Measure


class DataProcessor:
    def __init__(self, path_pred, path_unknown, path_inj):
        self.path_pred = path_pred
        self.path_unknown = path_unknown
        self.path_inj = path_inj

    def getDataFrame(self, ifo):
        # Existing code for reading and processing data
        preds = pd.read_csv(self.path_pred + f'pred_logs_unknown_{ifo}.csv')
        preds = preds.sort_values(by='Cluster time')
        preds = preds.loc[preds['Num triggers'] >= 10]
        preds = preds.loc[:, ~preds.columns.str.contains('^Unnamed')]

        maxcluster = pd.read_csv('/data/gravwav/lopezm/Projects/GlitchBank/time_slides/' + f'maxcluster_logs_{ifo}.csv')
        maxcluster = maxcluster.sort_values(by='Cluster time')
        maxcluster = maxcluster.loc[:, ~maxcluster.columns.str.contains('^Unnamed')]
        
        df = pd.merge(preds, maxcluster, on=['Cluster time', 'Cluster ID'], suffixes=("", "_max"))
        df.sort_values(by='Cluster time', inplace=True)
        return df

    def getDataFrame_eqmatch(self, ifo):
        """
            This is the zero lag data that keeps track of the original clusters
        """
        # Existing code for reading and processing data
        preds = pd.read_pickle(self.path_pred + f'pred_logs_unknown_eqmatch_reduced_{ifo}.csv')
        preds = preds.sort_values(by='Cluster time')
        preds = preds.loc[preds['Num triggers'] >= 10] #FIXME: need to add reduced
        preds = preds[preds['Pinj'].notna()]  # remove NaNs to make the calculation lighter
        return preds

    def LoadZeroLag(self, ifo, reset=False):
        unknown = pd.read_pickle(self.path_pred + f'pred_logs_unknown_eqmatch_reduced_{ifo.capitalize()}.csv')
        if reset:
            # We reset the index to pass it to coincTriggers
            unknown = unknown.reset_index(drop=True)
            # Correction of clusters
            unknown['Cluster idx'] = unknown.index
        maxcluster = pd.read_csv('/data/gravwav/lopezm/Projects/GlitchBank/time_slides/' + f'maxcluster_logs_{ifo.capitalize()}.csv')
        maxcluster = maxcluster.sort_values(by='Cluster time')
        maxcluster = maxcluster.loc[:, ~maxcluster.columns.str.contains('^Unnamed')]
        
        unknown = pd.merge(unknown, maxcluster, on=['Cluster time', 'Cluster ID'], suffixes=("", "_max"))

        # For some reason it is not adding the max suffix.. FIXME: this is done my hand
        unknown = unknown.rename(columns={col: f"{col}_max" for col in unknown.columns[14:]})

        with open(self.path_unknown + f'clusters_unknown_{ifo}_reduced.pkl', 'rb') as f:
            unknown_clusters = pickle.load(f)

        return unknown, unknown_clusters

    def getInjections(self):
        tmp = pd.read_pickle(self.path_inj + 'imbh_injections_chunk'+str(1)+'.csv')
        all_tmps = pd.DataFrame(columns=tmp.columns)
        train_inj_h1 = RecoveryData().get_training_data('H1')
        train_inj_l1 = RecoveryData().get_training_data('L1')
        train_inj_v1 = RecoveryData().get_training_data('V1')
        for i in range(1, 22):
            C1, C2 = [], []
            tmp = pd.read_pickle(self.path_inj + 'imbh_injections_chunk'+str(i)+'.csv')
            for ifo, train_inj in zip(['H1', 'L1', 'V1'], 
                                      [train_inj_h1, trin_inj_l1, train_inj_v1]):
                tmp, c1, c2 = RecoveryData().get_recovery_data(tmp, train_inj, ifo)
                C1.append(c1)
                C2.append(c2)
            all_tmps = pd.concat([all_tmps, tmp])
        for ifo in ['H1', 'L1', 'V1', 'H1L1', 'H1V1', 'L1V1', 'H1L1V1']:
            if len(ifo) == 2:
                all_tmps = LogitToStat(all_tmps, 1, ['Prob', ifo], 'Pinj_'+ifo)
            if len(ifo) == 4:
                all_tmps['Pinj_'+ifo] = Measure.harmonic2coinc(all_tmps['Pinj_'+ifo[:2]],
                                                       all_tmps['Pinj_'+ifo[2:]])
            if len(ifo) == 6:
                all_tmps['Pinj_'+ifo] = Measure.harmonic3coinc(all_tmps['Pinj_H1'],
                                                       all_tmps['Pinj_L1'],
                                                       all_tmps['Pinj_V1'])
            all_tmps = Measure.prob2stat(all_tmps, ifo, col='Pinj_'+ifo)
        return all_tmps, C1, C2


    def shiftedDataSet(self, tmp, window):
        # Rename the 'Cluster time' column to 'Cluster time old'
        tmp.rename(columns={'Cluster time': 'Cluster time old'}, inplace=True)
        
        # Create a new 'Cluster time' column by adding the window value to 'Cluster time old'
        tmp['Cluster time'] = tmp['Cluster time old'] + window
        
        # Return the modified DataFrame
        return tmp


def LogitToStat_old(data, col_pinj=0):

    tmp = data[[col for col in data.columns if 'Prob' in col]]
    tmp = torch.tensor(tmp.values)
    # Find the maximum value per row
    max_per_row, _ = torch.max(tmp, dim=1)
    # Find the indices of rows where the maximum value is in column A
    max_idx = torch.nonzero(tmp[:, col_pinj] == max_per_row).squeeze().numpy()
    # Select rows where column A has the maximum value per row
    data = data.iloc[max_idx]
    return data


def LogitToStat(tmp, temperature, substrings, label_stat, col_pinj=0):
    # Initialize a new column with NaN values
    tmp[label_stat] = np.nan
    
    # Extract columns containing all substrings
    z = tmp[[col for col in tmp.columns if all(sub in col for sub in substrings)]].values
    
    # Calculate softmax
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]  # Necessary step to enable broadcasting
    e_x = np.exp((z - s) / temperature)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]  # Ditto
    stat = e_x / div
    stat = torch.tensor(stat)

    ### OLD
    # Find the maximum value per row FIXME
    #max_per_row, _ = torch.max(stat, dim=1) 
    # Find the indices of rows where the maximum value is in column 'col_pinj'
    #max_idx = torch.nonzero(stat[:, col_pinj] == max_per_row).squeeze().numpy().tolist()
    ###
    
    # We want everything that is not NAN
    nonzero_idx = torch.nonzero(stat[:, col_pinj]).squeeze().numpy().tolist()
    # Assign the calculated softmax values to the new column, along with the index
    tmp[label_stat].iloc[nonzero_idx] = stat[nonzero_idx, 0].numpy().tolist()

    return tmp

class Merge:
    def __init__(self, path_store, timeslides_file):
        self.path_store = path_store
        self.timeslides_file = timeslides_file

    def getBKGtime(self, ifos):
        tmp = pd.read_csv(self.timeslides_file)

        t = 0 
        for i in range(len(tmp)):  # Iterate over the length of the dataframe
            t = t + tmp['time' + ifos.replace("1", "")].iloc[i]
            if t > 1000:
                break
        return np.round(t), i

    def mergeTriggers(self, ifos, new_stat=False):
        # We want to get the analysis time
        time_bkg, upper = self.getBKGtime(ifos)

        tmp = None  # Initialize tmp here
        for c in range(upper):
            #file = self.path_store + f'triggers{ifos}_run{c}.csv'
            file = self.path_store + f"triggers{ifos.replace('1', '')}_run_eq_match{c}.csv"

            if c != 0:
                tmp = pd.concat([tmp, pd.read_csv(file, index_col=0)])
            if c == 0:
                tmp = pd.read_csv(file, index_col=0)
            c = c + 1
        if len(ifos) == 4:
            ifo1, ifo2 = ifos[:2], ifos[2:]

            if new_stat:
                stat1 = tmp[f'Pinj_{ifo1}'] / (tmp[f'Chisq_max_{ifo1}'] / tmp[f'SNR_max_{ifo1}']**2)
                stat2 = tmp[f'Pinj_{ifo2}'] / (tmp[f'Chisq_max_{ifo2}'] / tmp[f'SNR_max_{ifo2}']**2)
                tmp['barPinj'] = Measure.harmonic2coinc(stat1, stat2)
                tmp['rank_stat_'+ifo1+ifo2] = tmp['barPinj']
            else:
                stat1 = tmp[f'Pinj_{ifo1}']
                stat2 = tmp[f'Pinj_{ifo2}']
                tmp['barPinj'] = Measure.harmonic2coinc(stat1, stat2)
                tmp = Measure.prob2stat(tmp, ifos[:2]+ifos[2:], col='barPinj')
        if len(ifos) == 6:

            if new_stat:
                stat1 = tmp[f'Pinj_H1'] / (tmp[f'Chisq_max_H1'] / tmp[f'SNR_max_H1']**2)
                stat2 = tmp[f'Pinj_L1'] / (tmp[f'Chisq_max_L1'] / tmp[f'SNR_max_L1']**2)
                stat3 = tmp[f'Pinj_V1'] / (tmp[f'Chisq_max_V1'] / tmp[f'SNR_max_V1']**2)
            else:
                stat1, stat2, stat3 = tmp[f'Pinj_H1'], tmp[f'Pinj_L1'], tmp[f'Pinj_V1']

            tmp['barPinj'] = Measure.harmonic3coinc(stat1, stat2, stat3)
            tmp = Measure.prob2stat(tmp, 'H1L1V1', col='barPinj')
       
        return time_bkg, tmp.reset_index(drop=True)
class RecoveryData:
    def __init__(self, tw=0.05):
        self.tw = tw
    
    def get_training_data(self, ifo, run='O3a'):
        path = '/data/gravwav/lopezm/Projects/GlitchBank/new_boostrapped/tw0.05/'
        data = np.load(path + "dataset_all_" + ifo + "_bootstrap_" + run + ".npy")

        # Pre-processing
        X, y = data[:, :-2], data[:, -2]

        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.1,
                                                            random_state=42)
        train_data = torch.cat((X_train, y_train.reshape(-1, 1)), dim=1)
        train_inj = train_data[train_data[:, 6] == 0]

        test_data = torch.cat((X_test, y_test.reshape(-1, 1)), dim=1)
        test_inj = test_data[test_data[:, 6] == 0]
        return train_inj
    
    def add_prediction(self, data, ifo, ids, preds):
        for i in range(preds.shape[1]):
            data['Prob'+str(i)+'_'+ifo] = np.ones(len(data))*np.nan
            mask = data.index.isin(ids)
            data.loc[mask, 'Prob'+str(i)+'_'+ifo] = preds[:, i]
        return data

    def get_recovery_data(self, data, training_data, ifo):
        ids = []
        triggs = []
        c1, c2 = 0, 0
        a = training_data[:, :-1].numpy()
        for i in range(len(data)):
            if not np.isnan(data.iloc[i]['feature_'+ifo]).any():
                b = data.iloc[i]['feature_'+ifo][:6].T
                subs = np.abs(a - b).sum(axis=1)
                if np.min(subs) > 10**-5:
                    # If it is smaller it belongs to the training set
                    ids.append(i)
                    triggs.append(b)
                else:
                    c1 += 1 # within training set
            else:
                c2 += 1 # not detected by GstLAL
        triggs = np.matrix(triggs)
        model = load_model(self.tw, ifo)
        ypred = predictions(triggs, model)
        data = self.add_prediction(data, ifo, ids, ypred)
        return data, c1, c2

    @staticmethod
    def get_recovery_data_nopred(data, training_data, ifo):
        """
        Get recovery data by comparing features in the given data against the training data.
        
        Parameters:
        self (object): The instance of the class.
        data (DataFrame): The input data containing features for comparison.
        training_data (Tensor): The training data used for comparison.
        ifo (str): The interferometer identifier.
    
        Returns:
        DataFrame: The input data with predictions added for non-training set points.
        int: The count of data points within the training set.
        int: The count of data points not detected by GstLAL.
        """
        # Convert the training data to a numpy array, excluding the last column
        matrix2 = np.round(training_data[:, :-1].numpy(), 4)
        
        # Extract non-NaN 'feature_h1' values and their indices
        not_nan_mask = data['feature_'+ifo].notna()
        ids_matrix1 = data.loc[not_nan_mask, 'feature_'+ifo].index
        
        # Convert non-NaN 'feature_h1' values to a numpy array and round
        matrix1 = np.round(np.vstack(data.loc[not_nan_mask, 'feature_'+ifo].values), 4)
        
        # Initialize a list to store the indices
        ids = []
        
        # Use broadcasting to find matching rows
        for i, row1 in enumerate(matrix1):
            if np.any(np.all(np.isclose(row1, matrix2, rtol=1e-4), axis=1)):
                ids.append(ids_matrix1[i])  # Store the original index
        
        mask = ~data.index.isin(ids)
        data = data.iloc[mask]  
        # Return the modified data and the counts of training set and not detected points
        return data


def getMaxValues(ifo, path, data):
    maxcluster = pd.read_csv(path + f'maxcluster_logs_{ifo}.csv')
    maxcluster = maxcluster.sort_values(by='Cluster time')
    maxcluster = maxcluster.loc[:, ~maxcluster.columns.str.contains('^Unnamed')]
    columns_to_drop = ['ifo', 'Chisq', 'Trigger ID', 'Template ID', 'Trigger time', 'Spin1z', 'Spin2z']
    new_column_names = {'Cluster time': 'Cluster time_'+ifo,
                        'Cluster ID': 'Cluster ID_'+ifo, 'Mass_1': 'Mass_1_max_'+ifo,
                        'Mass_2': 'Mass_2_max_'+ifo, 'SNR': 'SNR_max_'+ifo}
    maxcluster = maxcluster.drop(columns=columns_to_drop)
    maxcluster = maxcluster.rename(columns=new_column_names)
    common_columns = ['Cluster time_'+ifo, 'Cluster ID_'+ifo]
    # We round the data to avoid mistakes
    data['Cluster time_'+ifo] = data['Cluster time_'+ifo].round(5)
    maxcluster['Cluster time_'+ifo] = maxcluster['Cluster time_'+ifo].round(5)

    data = pd.merge(data, maxcluster, on=['Cluster time_'+ifo, 'Cluster ID_'+ifo], suffixes=("", "_max"), how='left')
    return data

def PrepareZeroLag(ifo):
    df = pd.read_csv('../output_new/tw0.05/predictions/pred_logs_unknown_'+ifo+'.csv', index_col=0)
    
    # Remove things larger than light time travel
    if len(ifo) == 4:
        if ifo == 'H1L1': t_window = 0.010+0.005
        if ifo == 'H1V1': t_window = 0.027+0.005
        if ifo == 'L1V1': t_window = 0.026+0.005

        cond1 = (np.abs(df['Cluster time_'+ifo[2:]] - df['Cluster time_'+ifo[:2]]) <= t_window)
        df = df[cond1]
        df = df[(df['Num triggers_'+ifo[2:]] >= 10) & (df['Num triggers_'+ifo[:2]] >= 10)]
        ifos = [ifo[:2], ifo[2:]]
    
    if len(ifo) == 6:
        cond1 = (np.abs(df['Cluster time_H1'] - df['Cluster time_L1']) <=0.010+0.005)
        cond2 = (np.abs(df['Cluster time_H1'] - df['Cluster time_V1']) <=0.027+0.005)
        cond3 = (np.abs(df['Cluster time_L1'] - df['Cluster time_V1']) <=0.026+0.005)
        df = df[cond1 & cond2 & cond3]
        df = df[(df['Num triggers_H1'] >= 10) & (df['Num triggers_L1'] >= 10) & (df['Num triggers_V1'] >= 10)]
        ifos = ['H1', 'L1', 'V1']
        
    for i in ifos:
       df = LogitToStat(df, 1, ['Prob', i], 'Pinj_'+i)
       df = Measure.prob2stat(df, i, col='Pinj_'+i)
       df = getMaxValues(i, '/data/gravwav/lopezm/Projects/GlitchBank/time_slides/', df)

    if len(ifo) == 4:
        df['barPinj'] = Measure.harmonic2coinc(df['Pinj_'+ifo[:2]],
                                               df['Pinj_'+ifo[2:]], epsilon=1e-20)
    if len(ifo) == 6:
        df['barPinj'] = Measure.harmonic3coinc(df['Pinj_H1'], df['Pinj_L1'],
                                               df['Pinj_V1'], epsilon=1e-20)
    df = Measure.prob2stat(df, ifo, col='barPinj')
    return df
