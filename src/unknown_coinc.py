import pandas as pd
import numpy as np
import argparse

def doubleCoincidence(data1, data2, ifo1, ifo2):
    
    # Define the columns of the data frame
    cols1, cols2 = data1.columns, data2.columns
    cols1, cols2 = [c + '_'+ifo1 for c in cols1], [c + '_'+ifo2 for c in cols2]
    columns=[cols1 + cols2]
    tmps = list()
    for i in range(len(data1)): 
        tmp1 = data1.iloc[i]
        tmp2 = data2.loc[(data2['Cluster time'] >= tmp1['Cluster time'] - 0.03) & (data2['Cluster time'] <= tmp1['Cluster time'] + 0.03)]

        if len(tmp2) == 1:
            tmps.append(list(tmp1.values)+ list(tmp2.values[0]))
            
    tmp = pd.DataFrame(tmps, columns=columns)
    return tmp

def tripleCoincidence(data1, data2, data3, ifo1, ifo2, ifo3):
    
    # Define the columns of the data frame
    cols1, cols2, cols3 = data1.columns, data2.columns, data3.columns
    cols1 = [c + '_'+ifo1 for c in cols1]
    cols2 = [c + '_'+ifo2 for c in cols2]
    cols3 = [c + '_'+ifo3 for c in cols3]
    columns=[cols1 + cols2 + cols3]
    
    # Iterate over the times
    tmps = list()
    for i in range(len(data1)): 
        tmp1 = data1.iloc[i]
        tmp2 = data2.loc[(data2['Cluster time'] >= tmp1['Cluster time'] - 0.03) & (data2['Cluster time'] <= tmp1['Cluster time'] + 0.03)]
        tmp3 = data3.loc[(data3['Cluster time'] >= tmp1['Cluster time'] - 0.03) & (data3['Cluster time'] <= tmp1['Cluster time'] + 0.03)]

        if (len(tmp2) == 1) and (len(tmp3) == 1):
            tmps.append(list(tmp1.values)+ list(tmp2.values[0]) + list(tmp3.values[0]))
    
    tmp = pd.DataFrame(tmps, columns=columns)
    return tmp

parser = argparse.ArgumentParser()
parser.add_argument('--tw', metavar='1', type=str, nargs=1,
                    help='Time window', default=1)
parser.add_argument('--ifos', metavar='1', type=str, nargs=1,
                    help='Interferometer combination', default=1)

args = parser.parse_args()
tw = args.tw[0]
ifos = args.ifos[0]

path = '/data/gravwav/lopezm/Projects/GlitchBank/computational-aspects-of-machine-learning-project-3/output_new/tw'+str(tw)+'/predictions/'


if len(ifos) == 4:
    data1 = pd.read_csv(path + 'pred_unknown_'+ifos[:2]+'.csv')
    data1 = data1.loc[:, ~data1.columns.str.match('Unnamed')]
    
    data2 = pd.read_csv(path + 'pred_unknown_'+ifos[2:]+'.csv')
    data2 = data2.loc[:, ~data2.columns.str.match('Unnamed')]
    print('Data loaded!')    
    data = doubleCoincidence(data1, data2, ifos[:2], ifos[2:])
    print(data)
    data.to_csv(path + 'pred_unknown_'+ifos+'.csv')
if len(ifos) == 6:
    
    pH1 = pd.read_csv(path + 'pred_unknown_H1.csv')
    pH1 = pH1.loc[:, ~pH1.columns.str.match('Unnamed')]

    pL1 = pd.read_csv(path + 'pred_unknown_L1.csv')
    pL1 = pL1.loc[:, ~pL1.columns.str.match('Unnamed')]

    pV1 = pd.read_csv(path + 'pred_unknown_V1.csv')
    pV1 = pV1.loc[:, ~pV1.columns.str.match('Unnamed')]
    pH1L1V1 = tripleCoincidence(pH1, pL1, pV1, 'H1', 'L1', 'V1')

    pH1L1V1.to_csv(path + 'pred_unknown_H1L1V1.csv')
