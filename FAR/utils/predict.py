import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../..')
from FAR.utils.measure import Measure
from FAR.utils.read import LogitToStat, RecoveryData
from src.torch_utils import load_model, predictions
from FAR.utils.utilities import rename_columns

def predictFeatures(path_to_file, ifos, types):

    # We load the data
    inj_df = pd.read_pickle(path_to_file)

    if types == 'inj':
        # If we are dealing with the injextions
        # remove the data from training before prediction
        train_inj =  RecoveryData().get_training_data(ifos[0].capitalize())
        inj_df = RecoveryData().get_recovery_data_nopred(inj_df, train_inj, ifos[0])

    for ifo in ifos:
        # Injections and zero lag have different naming conventions (FIXME?)
        if types == 'inj':
            num_trigg_label, feature_label = 'Num triggers_' + ifo, 'feature_' + ifo
        if types == 'zero_lag':
            num_trigg_label, feature_label = 'Num triggers', 'feature'

        # We need to select non NaN values for the prediction
        not_nan = inj_df[num_trigg_label].notna().values
        features = np.array([feature.flatten() for feature in inj_df.loc[not_nan, feature_label].values])
        model = load_model('0.05', ifo.capitalize())
        ypred = predictions(features, model) 
        tmp = np.full((len(inj_df), ypred.shape[1]), np.nan)
        tmp[not_nan] = ypred
        columns = [f'Prob_{i}_{ifo}' for i in range(ypred.shape[1])]
        inj_df[columns] = pd.DataFrame(tmp, index=inj_df.index)

        # This applies Softmax and then we transform Pinj into ranking stat
        inj_df = LogitToStat(inj_df, 1, ['Prob', ifo], 'Pinj_' + ifo)
        inj_df = Measure.prob2stat(inj_df, ifo, col='Pinj_' + ifo)

    if types == 'inj':
        # If we are dealing with injections we want to followup with coincidences
        if len(ifos) == 2:
            # we need the joint statistic
            ifo2 = ifos[0].replace('1', '')+ifos[1].replace('1', '')
            har2 = Measure.harmonic2coinc(inj_df['Pinj_' + ifos[0]], inj_df['Pinj_' + ifos[1]])
            inj_df['Pinj_'+ifo2] = har2
            inj_df = Measure.prob2stat(inj_df, ifo2, col='Pinj_' + ifo2)
            # and the joint SNR
            i1, i2 = 'injSNR'+ifos[0].capitalize(), 'injSNR'+ifos[1].capitalize()
            inj_df['injSNR'+ifo2] = np.sqrt(inj_df[i1]**2 + inj_df[i2]**2)
        if len(ifos) == 3:
             # we need the joint statistic
            ifo3 = ifos[0].replace('1', '')+ifos[1].replace('1', '')+ifos[2].replace('1', '')
            har3 = Measure.harmonic3coinc(inj_df['Pinj_' + ifos[0]], inj_df['Pinj_' + ifos[1]], inj_df['Pinj_' + ifos[2]])
            inj_df['Pinj_'+ifo3] = har3
            inj_df = Measure.prob2stat(inj_df, ifo3, col='Pinj_' + ifo3)
            # and the joint SNR
            i1, i2, i3 = 'injSNR'+ifos[0].capitalize(), 'injSNR'+ifos[1].capitalize(), 'injSNR'+ifos[2].capitalize()
            inj_df['injSNR'+ifo3] = np.sqrt(inj_df[i1]**2 + inj_df[i2]**2 + inj_df[i3]**2)
            
    return inj_df