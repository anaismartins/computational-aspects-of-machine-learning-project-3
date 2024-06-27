import numpy as np
import pandas as pd
from fractions import Fraction
from itertools import product

class Measure:
    def __init__(self):
        pass  # Placeholder to avoid syntax error

    @staticmethod
    def FAR2stat(desired_y, x, y_obs):
        index_closest_y = np.abs(y_obs - desired_y).argmin()
        stat = x[index_closest_y]
        return stat
    
    @staticmethod
    def stat2FAR(desired_x, x, y_obs):
        if len(np.where(y_obs == 0)[0]) > 0:
            maximum = x[np.where(y_obs == 0)[0][0]]
        else:
            maximum = max(x)
        if desired_x > maximum:
            FAR_obs = 1e-3
            fraction_FAR = "1/{}".format(int(1 / FAR_obs))
        elif np.isnan(desired_x): 
            FAR_obs, fraction_FAR = np.nan, np.nan
        else:
            index_closest_x = np.abs(x - desired_x).argmin()
            FAR_obs = y_obs[index_closest_x]
            #fraction_FAR = "1/{}".format(int(1 / FAR_obs)) #FIXME
            fraction_FAR = np.nan
        # Represent the reciprocal as a fraction
        return FAR_obs, fraction_FAR
    @staticmethod
    def prob2stat(tmp, ifo, col):
        ranking_stat = -np.log(1- tmp[col] + 1e-20)
        tmp['rank_stat_'+ifo] = ranking_stat
        return tmp
        
    @staticmethod
    def harmonic2coinc(val1, val2, epsilon=1e-20):
        # FIXME: this will need to be fixed in the NN
        val1 = val1.astype(np.float128) + epsilon
        val2 = val2.astype(np.float128) + epsilon

        harmonic = 2 / (1/val1 + 1/val2)
        return harmonic

    @staticmethod
    def harmonic3coinc(val1, val2, val3, epsilon=1e-20):
        # We need higher resolution
        # FIXME: this will need to be fixed in the NN
        val1 = val1.astype(np.float128) + epsilon
        val2 = val2.astype(np.float128) + epsilon
        val3 = val3.astype(np.float128) + epsilon

        harmonic = 3 / (1/val1 + 1/val2 + 1/val3)
        return harmonic

    @staticmethod
    def limitTriggers(ifo, data):
        # First loss in efficiency is due to # of triggers
        if len(ifo) == 4:
            cond1 = (data['Num triggers_'+ifo[2:]] >= 10)
            cond2 = (data['Num triggers_'+ifo[:2]] >= 10)
            data = data[cond1 & cond2]
        else: 
            cond1 = (data['Num triggers_H1'] >= 10)
            cond2 = (data['Num triggers_L1'] >= 10)
            cond3 = (data['Num triggers_V1'] >= 10)
            data = data[cond1 & cond2 & cond3]
        return data
    @staticmethod 
    def TPR(data, ifo, p_star):
        
        """
        Calculate the True Positive Rate (TPR) for a given set of data, interferometer configuration, 
        and decision threshold.
    
        Parameters:
        - data (DataFrame): The DataFrame containing the detection data and statistics.
        - ifo (str): A string identifying the interferometer configuration. This affects which columns 
                     are used to check for data completeness and calculation.
        - p_star (float): The decision threshold on the rank statistic used to determine true positives.
    
        Returns:
        - tpr_gstlal (float): The True Positive Rate calculated using only the GSTLAL-detected events 
                              that pass the p_star threshold.
        - tpr_all (float): The True Positive Rate calculated over all data that passes the p_star threshold.
    
        Note:
        - This method assumes the presence of 'rank_stat_' prefixed columns in the data for detection statistics.
        - The data should have 'Cluster time_' prefixed columns based on the `ifo` configuration to check 
          for missing values before calculation.
        """
        # We only take into account the injections that ocurred with detector ON
        data = data[(data['Inside'+ifo] == 1)]
        pos_all = len(data)

        # We select injections detected by GstLAL
        columns_to_check = ['Cluster time_'+ifo[2:], 'Cluster time_'+ifo[:2]] if len(ifo) == 4 else ['Cluster time_H1', 'Cluster time_L1', 'Cluster time_V1']
        data_gstlal = data.dropna(subset=columns_to_check)
        pos_gstlal = len(data_gstlal)

        # First loss in efficiency is due to # of triggers
        data, data_gstlal = Measure.limitTriggers(ifo, data), Measure.limitTriggers(ifo, data_gstlal)
        
        tp_all = len(data[(data['rank_stat_'+ifo] >= p_star)])
        tp_gstlal = len(data_gstlal[data_gstlal['rank_stat_'+ifo] >= p_star])
        
        return tp_gstlal, pos_gstlal, tp_all, pos_all

    @staticmethod
    def FPR(data, ifo, p_star):
        """
        Calculate the False Positive Rate (FPR) for a given set of data and decision threshold.
    
        Parameters:
        - data (DataFrame): The DataFrame containing the detection data.
        - ifo (str): A string identifying the interferometer configuration used to index the rank statistics.
        - p_star (float): The decision threshold on the rank statistic above which detections are considered.
    
        Returns:
        - fpr (float): The False Positive Rate calculated as the percentage of all detections that are false 
                       positives (i.e., above the threshold p_star but not actual detections).
    
        Note:
        - This method assumes the data contains a 'rank_stat_' prefixed column corresponding to the ifo configuration.
        """
        neg_all = len(data)
        fp = len(data[data['rank_stat_'+ifo] >= p_star])
        fpr = fp / neg_all * 100
        return fpr

    @staticmethod
    def getFARandThreshold(data, ifo, t_bkg, t_search, FAR=0.01):
        binning = Binning()

        x, y, l, p = binning.fit_and_binning(data=data['rank_stat_'+ifo], rank=2)
        y_obs, ylabel = binning.yVar(y, l, 'far', t_bkg, t_search)
        p_star = Measure.FAR2stat(FAR, x, y_obs)
        return x, y_obs, p_star, ylabel
    
class Styling:
    @staticmethod
    def load_triggers_2(d1, d2):
        tmp = [
            d1['Mass_1'], d2['Mass_1'],
            d1['Mass_2'], d2['Mass_2'],
            d1['SNR'], d2['SNR'],
            d1['Chisq'], d2['Chisq'],
            d1['Mass_1_max'], d2['Mass_1_max'],
            d1['Mass_2_max'], d2['Mass_2_max'],
            d1['SNR_max'], d2['SNR_max'],
            d1['Chisq_max'], d2['Chisq_max'],
            #d1['Prob0'], d2['Prob0'],
            d1['Pinj'], d2['Pinj'],
            d1['Num triggers'], d2['Num triggers'],
            d1['Cluster time'], d2['Cluster time'],
            d1['Cluster time old'], d2['Cluster time old'],
            d1['Cluster ID'], d2['Cluster ID'],
            d1['Trigger time'], d2['Trigger time'],
            d1['Trigger ID'], d2['Trigger ID'],
            d1['Template ID'], d2['Template ID']
        ]
        return tmp

    @staticmethod
    def load_triggers_3(d1, d2, d3):
        tmp = [
            d1['Mass_1'], d2['Mass_1'], d3['Mass_1'],
            d1['Mass_2'], d2['Mass_2'], d3['Mass_2'],
            d1['SNR'], d2['SNR'], d3['SNR'],
            d1['Chisq'], d2['Chisq'], d3['Chisq'],
            d1['Mass_1_max'], d2['Mass_1_max'], d3['Mass_1_max'],
            d1['Mass_2_max'], d2['Mass_2_max'], d3['Mass_2_max'],
            d1['SNR_max'], d2['SNR_max'], d3['SNR_max'],
            d1['Chisq_max'], d2['Chisq_max'], d3['Chisq_max'],
            #d1['Prob0'], d2['Prob0'], d3['Prob0'],
            d1['Pinj'], d2['Pinj'], d3['Pinj'],
            d1['Num triggers'], d2['Num triggers'], d3['Num triggers'],
            d1['Cluster time'], d2['Cluster time'], d3['Cluster time'],
            d1['Cluster time old'], d2['Cluster time old'], d3['Cluster time old'],
            d1['Cluster ID'], d2['Cluster ID'], d3['Cluster ID'],
            d1['Trigger time'], d2['Trigger time'], d3['Trigger time'],
            d1['Trigger ID'], d2['Trigger ID'], d3['Trigger ID'],
            d1['Template ID'], d2['Template ID'], d3['Template ID']
        ]
        return tmp

    @staticmethod
    def get_columns(ifos):
        if len(ifos) == 2:
            cols = [
                'Mass_1_' + ifos[0], 'Mass_1_' + ifos[1],
                'Mass_2_' + ifos[0], 'Mass_2' + ifos[1],
                'SNR_' + ifos[0], 'SNR_' + ifos[1],
                'Chisq_' + ifos[0], 'Chisq_' + ifos[1],
                'Mass_1_max_' + ifos[0], 'Mass_1_max_' + ifos[1],
                'Mass_2_max_' + ifos[0], 'Mass_2_max_' + ifos[1],
                'SNR_max_' + ifos[0], 'SNR_max_' + ifos[1],
                'Chisq_max_' + ifos[0], 'Chisq_max_' + ifos[1],
                'Pinj_' + ifos[0], 'Pinj_' + ifos[1],
                'Num_triggers_' + ifos[0], 'Num_triggers_' + ifos[1],
                'Cluster_time_' + ifos[0], 'Cluster_time_' + ifos[1],
                'Cluster_time_old_' + ifos[0], 'Cluster_time_old_' + ifos[1],
                'Cluster_ID_' + ifos[0], 'Cluster_ID_' + ifos[1],
                'Trigger_time_' + ifos[0], 'Trigger_time_' + ifos[1],
                'Trigger_ID_' + ifos[0], 'Trigger_ID_' + ifos[1],
                'Template_ID_' + ifos[0], 'Template_ID_' + ifos[1]
            ]
        elif len(ifos) == 3:
            cols = [
                'Mass_1_H1', 'Mass_1_L1', 'Mass_1_V1',
                'Mass_2_H1', 'Mass_2_L1', 'Mass_2_V1',
                'SNR_H1', 'SNR_L1', 'SNR_V1',
                'Chisq_H1', 'Chisq_L1', 'Chisq_V1',
                'Mass_1_max_H1', 'Mass_1_max_L1', 'Mass_1_max_V1',
                'Mass_2_max_H1', 'Mass_2_max_L1', 'Mass_2_max_V1',
                'SNR_max_H1', 'SNR_max_L1', 'SNR_max_V1',
                'Chisq_max_H1', 'Chisq_max_L1', 'Chisq_max_V1',
                'Pinj_H1', 'Pinj_L1', 'Pinj_V1',
                'Num_triggers_H1', 'Num_triggers_L1', 'Num_triggers_V1',
                'Cluster_time_H1', 'Cluster_time_L1', 'Cluster_time_V1',
                'Cluster_time_old_H1', 'Cluster_time_old_L1', 'Cluster_time_old_V1',
                'Cluster_ID_H1', 'Cluster_ID_L1', 'Cluster_ID_V1',
                'Trigger_time_H1', 'Trigger_time_L1', 'Trigger_time_V1',
                'Trigger_ID_H1', 'Trigger_ID_L1', 'Trigger_ID_V1',
                'Template_ID_H1', 'Template_ID_L1', 'Template_ID_V1'
            ]
        return cols

class Binning:
    def __init__(self):
        pass
    @staticmethod
    def fit_and_binning(data, rank=None):
        # Compute histogram counts and bin edges
        bin_array = np.arange(0, 50, 0.01)  
        counts, bins = np.histogram(data, bins=bin_array)
        cumulative_counts = np.cumsum(counts[::-1])[::-1]
        # Check thesis Eq. 3.43
        length = len(data)
        N = np.sum(data.values[:, None] >= bin_array, axis=0)
        x, y = bin_array.copy(), N.copy()
        
        if rank is not None:
            # Perform quadratic fit
            # We need to do this in log and add an epsilon to avoid crashing
            coefficients = np.polyfit(x, y, rank)
            p = np.poly1d(coefficients)
            return x, y, length, p
        else:
            return x, y, length

    @staticmethod
    def yVar(y, length, var, t_bkg=None, t_search=None):
        if var == 'counts':
            y 
            return y, 'Number of counts'
        if var == 'far':
            # Check Eq. 3.43 of my thesis
            return y / (length * t_search), r'FAR (yr$^{-1}$)'
        if var == 'events':
            return y * t_search / t_bkg, 'Number of events'

class CoincMass:
    def __init__(self, tmp, ifos, sigma1, sigma2):
        self.tmp = tmp
        self.ifos = ifos
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def filterCoincMass(self):
        self.tmp = self.tmp[self.tmp['Mass_1_max_' + self.ifos[1]].between(
            self.tmp['Mass_1_max_' + self.ifos[0]] - self.sigma1,
            self.tmp['Mass_1_max_' + self.ifos[0]] + self.sigma1)]
        self.tmp = self.tmp[self.tmp['Mass_2_max_' + self.ifos[1]].between(
            self.tmp['Mass_2_max_' + self.ifos[0]] - self.sigma2,
            self.tmp['Mass_2_max_' + self.ifos[0]] + self.sigma2)]
        return self.tmp

def SNRveto(snr_star, data, ifo):
    if len(ifo) == 2:
        print('SNR_max_'+ifo[0], data.columns)
        cond1 = (data['SNR_max_'+ifo[0]] >= snr_star)
        cond2 = (data['SNR_max_'+ifo[1]] >= snr_star)
        data = data[cond1 & cond2]
    if len(ifo) == 3:
        cond1 = (data['SNR_max_H1'] >= snr_star)
        cond2 = (data['SNR_max_L1'] >= snr_star)
        cond3 = (data['SNR_max_V1'] >= snr_star)
        data = data[cond1 & cond2 & cond3]
    return data

def rename_columns(col, sub_old, sub_new):
    if col.endswith(sub_old):
        return col[:-len(sub_old)] + sub_new
    return col
    
class ClusterProcessor:
    def __init__(self):
        pass

    @staticmethod
    def mergeClusters(cluster1, cluster2, suffixes, label, shift1=0, shift2=0):
        """
        Merge two clusters based on 'Template ID' and calculate the time difference.

        Parameters:
        cluster1 (pd.DataFrame): First cluster DataFrame.
        cluster2 (pd.DataFrame): Second cluster DataFrame.
        suffixes (list): List of suffixes to add to the column names.
        label (str): Name of the column to store the time difference.
        shift1 (float): Time shift for the first cluster.
        shift2 (float): Time shift for the second cluster.

        Returns:
        pd.DataFrame: Merged DataFrame with time difference calculated.
        """
        # Merge clusters on 'Template ID', dropping duplicate entries
        tmp = pd.merge(cluster1.drop_duplicates(subset=['Template ID']),
                       cluster2.drop_duplicates(subset=['Template ID']),
                       on=['Template ID'], how='inner', suffixes=suffixes)
        # Calculate the absolute time difference with optional shifts
        tmp[label] = np.abs((tmp['Trigger time' + suffixes[0]] + shift1) - (tmp['Trigger time' + suffixes[1]] + shift2))
        return tmp
    
    @staticmethod
    def mergeClusters3(cluster1, cluster2, cluster3, shift2=0, shift3=0):
        tmp = pd.merge(cluster1.drop_duplicates(subset=['Template ID']),
                       cluster2.drop_duplicates(subset=['Template ID']),
                       on=['Template ID'], how='inner', suffixes=['_H1', '_L1'])
        # This is a hack for '_v1' suffixes
        cluster3 = cluster3.rename(columns={col: col + '_V1' if col != 'Template ID' else col for col in cluster3.columns})
        tmp = pd.merge(tmp.drop_duplicates(subset=['Template ID']),
                       cluster3.drop_duplicates(subset=['Template ID']),
                       on=['Template ID'], how='inner', suffixes=['', '_V1'])
    
        # We shift the clusters in time
        tmp['Trigger time_L1'] = tmp['Trigger time_L1'] + shift2
        tmp['Trigger time_V1'] = tmp['Trigger time_V1'] + shift3

        # And calculate delta time
        tmp['Delta t_H1L1'] = np.abs(tmp['Trigger time_H1'] - tmp['Trigger time_L1'])
        tmp['Delta t_H1V1'] = np.abs(tmp['Trigger time_H1'] - tmp['Trigger time_V1'])
        tmp['Delta t_L1V1'] = np.abs(tmp['Trigger time_L1'] - tmp['Trigger time_V1'])

        return tmp

    def coincTriggers2(self, x, data_x, data_y, cluster_x, cluster_y, suffixes, shift1=0, shift2=0, window=1):
        """
        Find coincident triggers between two data sets within a time window.

        Parameters:
        x (int): Index of the trigger in data_x.
        data_x (pd.DataFrame): DataFrame containing the first set of data.
        data_y (pd.DataFrame): DataFrame containing the second set of data.
        cluster_x (dict): Dictionary containing clusters for data_x.
        cluster_y (dict): Dictionary containing clusters for data_y.
        suffixes (list): List of suffixes to add to the column names.
        shift1 (float): Time shift for the first cluster.
        shift2 (float): Time shift for the second cluster.
        window (float): Time window for finding coincident triggers.

        Returns:
        tuple: Index x, index y, and delta_t if conditions are met; otherwise, returns x, None, None.
        """
        # Directly access the cluster time for the trigger at index x
        cluster_time_x = data_x.at[x, 'Cluster time']

        # Create a mask to filter data_y within the time window around cluster_time_x
        mask = (data_y['Cluster time'] >= cluster_time_x - window) & (data_y['Cluster time'] <= cluster_time_x + window)
        tmp_ = data_y[mask]

        # If no data points are found in the time window, return None values
        if tmp_.empty:
            return x, None, None

        # Iterate through the filtered data points in tmp_
        for i in range(len(tmp_)):
            y = tmp_.index[i]
            cluster1, cluster2 = cluster_x[x], cluster_y[y]
            # Check if clusters have sufficient length
            if len(cluster1) < 10 or len(cluster2) < 10:
                continue
    
            # Merge the clusters and calculate delta_t
            tmp = self.mergeClusters(cluster1, cluster2, suffixes, 'Delta t', shift1, shift2)
            delta_t = tmp['Delta t'].mean()

            # Check delta_t based on suffixes and thresholds
            if (suffixes == ['_H1', '_L1']) and (delta_t <= 0.015):
                return x, y, delta_t
            if (suffixes == ['_H1', '_V1']) and (delta_t <= 0.032):
                return x, y, delta_t
            if (suffixes == ['_L1', '_V1']) and (delta_t <= 0.031):
                return x, y, delta_t

        # If no valid coincident triggers are found, return None values
        return x, None, None

    def coincTriggers3(self, x, data_x, data_y, data_z, cluster_x, cluster_y, cluster_z, shift1=0, shift2=0, shift3=0, window=1):
        """
        Find coincident triggers between three data sets within a time window.

        Parameters:
        x (int): Index of the trigger in data_x.
        data_x (pd.DataFrame): DataFrame containing the first set of data.
        data_y (pd.DataFrame): DataFrame containing the second set of data.
        data_z (pd.DataFrame): DataFrame containing the third set of data.
        cluster_x (dict): Dictionary containing clusters for data_x.
        cluster_y (dict): Dictionary containing clusters for data_y.
        cluster_z (dict): Dictionary containing clusters for data_z.
        shift1 (float): Time shift for the first cluster.
        shift2 (float): Time shift for the second cluster.
        shift3 (float): Time shift for the third cluster.
        window (float): Time window for finding coincident triggers.

        Returns:
        tuple: Index x, index y, index z, and delta_t values if conditions are met; otherwise, returns None values.
        """
       
        # Directly access the value instead of using iloc
        cluster_time_x = data_x.at[x, 'Cluster time']

        # Vectorized condition check
        mask_y = (data_y['Cluster time'] >= cluster_time_x - window) & (data_y['Cluster time'] <= cluster_time_x + window)
        tmp_y = data_y[mask_y]

        mask_z = (data_z['Cluster time'] >= cluster_time_x - window) & (data_z['Cluster time'] <= cluster_time_x + window)
        tmp_z = data_z[mask_z]

        if (tmp_y.empty) or (tmp_z.empty):
            return x, None, None, None, None, None

        for i, j in product(range(len(tmp_y)), range(len(tmp_z))):
            y = tmp_y.index[i]
            z = tmp_z.index[j]
            #tx, ty, tz = cluster_time_x, tmp_y['Cluster time'].iloc[i], tmp_z['Cluster time'].iloc[j]
            #print(tx, ty, tx)
            cluster1, cluster2, cluster3 = cluster_x[x], cluster_y[y], cluster_z[z]
    
            # Check the length conditions
            if len(cluster1) < 10 or len(cluster2) < 10 or len(cluster3) < 10:
                return x, None, None, None, None, None

            tmp = self.mergeClusters3(cluster1, cluster2, cluster3, shift2=shift2, shift3=shift3)

            delta_t1, delta_t2, delta_t3 = tmp['Delta t_H1L1'].mean(), tmp['Delta t_H1V1'].mean(), tmp['Delta t_L1V1'].mean()
            # if len(tmp) > 0:
            #     print(delta_t1, delta_t2, delta_t3)
            if (delta_t1 <= 0.015) and (delta_t2 <= 0.032) and (delta_t3 <= 0.031):
                #print(delta_t1, delta_t2, delta_t3)
                return x, y, z, delta_t1, delta_t2, delta_t3
    
        return x, None, None, None, None, None

    def mergeData2(self, data1, data2, idx1, idx2, suffixes, add_value_h1=0, add_value_v1=0):
        """
        Merge two data sets according to the provided indices and suffixes, and optionally add values to the data.

        Parameters:
        data1 (pd.DataFrame): First input DataFrame.
        data2 (pd.DataFrame): Second input DataFrame.
        idx1 (list): List of indices for data1.
        idx2 (list): List of indices for data2.
        suffixes (tuple): Tuple of suffixes to add to the column names.

        Returns:
        pd.DataFrame: Concatenated DataFrame with appropriate suffixes.
        """
        # Add suffixes to the column names
        cols1 = [col + suffixes[0] for col in data1.columns]
        cols2 = [col + suffixes[1] for col in data2.columns]
        cols1.extend(['Cluster idx' + suffixes[0]])
        cols2.extend(['Cluster idx' + suffixes[1]])
        print(cols1, cols2)
        # Initialize matrices with NaN values
        # Object dtype is used due to different data formats within the data frame
        matrix1 = pd.DataFrame(np.full((len(idx1), len(data1.columns) + 1), np.nan), dtype=object)
        matrix2 = pd.DataFrame(np.full((len(idx2), len(data2.columns) + 1), np.nan), dtype=object)
       
        # Convert index lists to numpy arrays for faster indexing
        idx1, idx2 = np.array(idx1), np.array(idx2)
        # Filter out None values and get valid indices
        valid_idx1, valid_idx2 = ~pd.isna(idx1), ~pd.isna(idx2)
        
        # Populate matrices using valid indices
        matrix1.loc[valid_idx1] = np.concatenate([data1.loc[idx1[valid_idx1]].values,
                                                  np.asarray([idx1[valid_idx1]]).reshape(-1, 1)], axis=1)
        matrix2.loc[valid_idx2] = np.concatenate([data2.loc[idx2[valid_idx2]].values,
                                                  np.asarray([idx2[valid_idx2]]).reshape(-1, 1)], axis=1)

        # Concatenate matrices along the columns
        matrix_concat = np.concatenate([matrix1, matrix2], axis=1)

        # Create DataFrame with concatenated matrix and appropriate column names
        df = pd.DataFrame(matrix_concat, columns=cols1 + cols2)
        df = df.dropna(subset=['rank_stat' + suffixes[0], 'rank_stat' + suffixes[1]])
        # df.rename(columns=lambda col: rename_columns(col, suffixes[0], suffixes[0].capitalize()), inplace=True)
        # df.rename(columns=lambda col: rename_columns(col, suffixes[1], suffixes[1].capitalize()), inplace=True)
        print(df.columns)

        df['barPinj'] = Measure.harmonic2coinc(df['Pinj' + suffixes[0]], df['Pinj' + suffixes[1]])
        df = Measure.prob2stat(df, suffixes[0][1:] + suffixes[1][1:], col='barPinj')
        return df

    def mergeData3(self, data1, data2, data3, idx1, idx2, idx3):
        """
        Merge three data sets according to the provided indices and suffixes.

        Parameters:
        data1 (pd.DataFrame): First input DataFrame.
        data2 (pd.DataFrame): Second input DataFrame.
        data3 (pd.DataFrame): Third input DataFrame.
        idx1 (list): List of indices for data1.
        idx2 (list): List of indices for data2.
        idx3 (list): List of indices for data3.

        Returns:
        pd.DataFrame: Concatenated DataFrame with appropriate suffixes.
        """
        # Add suffixes to the column names
        cols1 = [col + '_H1' for col in data1.columns]
        cols2 = [col + '_L1' for col in data2.columns]
        cols3 = [col + '_V1' for col in data3.columns]
        cols1.extend(['Cluster idx_H1'])
        cols2.extend(['Cluster idx_L1'])
        cols3.extend(['Cluster idx_H1'])

        # Initialize matrices with NaN values
        # Object dtype is used due to different data formats within the data frame
        matrix1 = pd.DataFrame(np.full((len(idx1), len(data1.columns) + 1), np.nan), dtype=object)
        matrix2 = pd.DataFrame(np.full((len(idx2), len(data2.columns) + 1), np.nan), dtype=object)
        matrix3 = pd.DataFrame(np.full((len(idx3), len(data3.columns) + 1), np.nan), dtype=object)
        
        # Convert index lists to numpy arrays for faster indexing
        idx1, idx2, idx3 = np.array(idx1), np.array(idx2), np.array(idx3)
        # Filter out None values and get valid indices
        valid_idx1, valid_idx2, valid_idx3 = ~pd.isna(idx1), ~pd.isna(idx2), ~pd.isna(idx3)
        
        # Populate matrices using valid indices
        matrix1.loc[valid_idx1] = np.concatenate([data1.loc[idx1[valid_idx1]].values,
                                                  np.asarray([idx1[valid_idx1]]).reshape(-1, 1)], axis=1)
        matrix2.loc[valid_idx2] = np.concatenate([data2.loc[idx2[valid_idx2]].values,
                                                  np.asarray([idx2[valid_idx2]]).reshape(-1, 1)], axis=1)
        matrix3.loc[valid_idx3] = np.concatenate([data3.loc[idx3[valid_idx3]].values,
                                                  np.asarray([idx3[valid_idx3]]).reshape(-1, 1)], axis=1)

        # Concatenate matrices along the columns
        matrix_concat = np.concatenate([matrix1, matrix2, matrix3], axis=1)

        # Create DataFrame with concatenated matrix and appropriate column names
        df = pd.DataFrame(matrix_concat, columns=cols1 + cols2 + cols3)
        df = df.dropna(subset=['rank_stat_H1',
                               'rank_stat_L1', 
                               'rank_stat_V1'])
        ifo1, ifo2, ifo3 = 'H1', 'L1', 'V1'
        # df.rename(columns=lambda col: rename_columns(col, ifo1, ifo1.capitalize()), inplace=True)
        # df.rename(columns=lambda col: rename_columns(col, ifo2, ifo2.capitalize()), inplace=True)
        # df.rename(columns=lambda col: rename_columns(col, ifo3, ifo3.capitalize()), inplace=True)

        df['barPinj'] = Measure.harmonic3coinc(df['Pinj_H1'], df['Pinj_L1'], df['Pinj_V1'])
        df = Measure.prob2stat(df, ifo1 + ifo2 + ifo3, col='barPinj')
   
        return df

