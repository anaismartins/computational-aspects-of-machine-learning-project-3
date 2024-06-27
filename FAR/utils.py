import pandas as pd

def getDataFrame(ifo):
    path_pred = '/data/gravwav/lopezm/Projects/GlitchBank/computational-aspects-of-machine-learning-project-3/output_new/tw0.05/predictions/'
    preds = pd.read_csv(path_pred + 'pred_unknown_'+ifo+'.csv', index_col=0)
    preds = preds.loc[:, ~preds.columns.str.contains('^Unnamed')]
    preds = preds.loc[preds['Num triggers'] > 10] # limit number of triggers
    
    maxcluster = pd.read_csv('/data/gravwav/lopezm/Projects/GlitchBank/time_slides/maxcluster_'+ifo+'.csv', index_col=0)
    maxcluster = maxcluster.loc[:, ~maxcluster.columns.str.contains('^Unnamed')]
    
    df = pd.merge(preds, maxcluster, on=['Cluster time', 'Cluster ID'],
             suffixes=("", "_max"))
    df.sort_values(by='Cluster time', inplace=True)
    # We also shuffle
    #df = df.sample(frac=1.0)
    return df

def harmonic2coinc(val1, val2):
    harmonic = 2/(1/val1 + 1/val2)
    return harmonic

def harmonic3coinc(val1, val2, val3):
    harmonic = 3/(1/val1 + 1/val2 + 1/val3)
    return harmonic

def shiftedDataSet(tmp, window=1):
    min_time = min(tmp['Cluster time'].values)
    last_id = tmp.loc[tmp['Cluster time'] < min_time + window].index[-1] + 1
    tmp = pd.concat([tmp[last_id:], tmp[:last_id]])
    tmp['Cluster time'] = tmp['Cluster time'] + window
    tmp['Cluster time old'] = tmp['Cluster time'] - window
    return tmp

def coincMass(coinc, mass1, mass2, sigmaM1, sigmaM2):
        coinc = coinc.loc[(coinc['Mass_1_max'] > mass1 - sigmaM1) & (coinc['Mass_1_max'] < mass1 + sigmaM1)]
        coinc = coinc.loc[(coinc['Mass_2_max'] > mass2 - sigmaM2) & (coinc['Mass_2_max'] < mass2 + sigmaM2)]
        return coinc
    
def LoadTriggers2(d1, d2):  
    
    tmp =  [d1['Mass_1'], d2['Mass_1'], 
            d1['Mass_2'], d2['Mass_2'],
            d1['SNR_max'], d2['SNR_max'],
            d1['Prob0'], d2['Prob0'],
            d1['Num triggers'], d2['Num triggers'],
            d1['Cluster time'], d2['Cluster time'],
            d1['Cluster ID'], d2['Cluster ID'],
            d1['Trigger time'], d2['Trigger time'],
            d1['Trigger ID'], d2['Trigger ID'],
            d1['Template ID'], d2['Template ID']]
    return tmp

def LoadTriggers3(d1, d2, d3):  
    
    tmp =  [d1['Mass_1'], d2['Mass_1'], d3['Mass_1'], 
            d1['Mass_2'], d2['Mass_2'], d3['Mass_2'],
            d1['SNR_max'], d2['SNR_max'], d3['SNR_max'], 
            d1['Prob0'], d2['Prob0'], d3['Prob0'],
            d1['Num triggers'], d2['Num triggers'], d3['Num triggers'],
            d1['Cluster time'], d2['Cluster time'], d3['Cluster time'],
            d1['Cluster ID'], d2['Cluster ID'], d3['Cluster ID'],
            d1['Trigger time'], d2['Trigger time'], d3['Trigger time'],
            d1['Trigger ID'], d2['Trigger ID'], d3['Trigger ID'], 
            d1['Template ID'], d2['Template ID'], d3['Template ID']]
    return tmp

def getColumns(ifos):
    
    if len(ifos) == 2:
        cols = ['Mass_1_max_'+ifos[0], 'Mass_1_max_'+ifos[1],
                'Mass_2_max_'+ifos[0], 'Mass_2_max_'+ifos[1],
                'SNR_max_'+ifos[0], 'SNR_max_'+ifos[1],
                'Pinj_'+ifos[0],'Pinj_'+ifos[1],
                'Num_triggers_'+ifos[0], 'Num_triggers_'+ifos[1],
                'Cluster_time_'+ifos[0], 'Cluster_time_'+ifos[1], 
                'Cluster_ID_'+ifos[0], 'Cluster_ID_'+ifos[1], 
                'Trigger_time_'+ifos[0], 'Trigger_time_'+ifos[1], 
                'Trigger_ID_'+ifos[0], 'Trigger_ID_'+ifos[1],
                'Template_ID_'+ifos[0], 'Template_ID_'+ifos[1]]
        
    if len(ifos) == 3:
        cols = ['Mass_1_max_H1', 'Mass_1_max_L1', 'Mass_1_max_V1',
                'Mass_2_max_H1', 'Mass_2_max_L1', 'Mass_2_max_V1',
                'SNR_max_H1', 'SNR_max_L1', 'SNR_max_V1',
                'Pinj_H1','Pinj_L1', 'Pinj_V1',
                'Num_triggers_H1','Num_triggers_L1', 'Num_triggers_V1',
                'Cluster_time_H1', 'Cluster_time_L1', 'Cluster_time_V1', 
                'Cluster_ID_H1', 'Cluster_ID_L1', 'Cluster_ID_V1', 
                'Trigger_time_H1', 'Trigger_time_L1', 'Trigger_time_V1',
                'Trigger_ID_H1', 'Trigger_ID_L1', 'Trigger_ID_V1',
                'Template_ID_H1', 'Template_ID_L1', 'Template_ID_V1']
    return cols

