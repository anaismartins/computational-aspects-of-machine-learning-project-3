import time
import sys
sys.path.insert(0, './utils')
import pandas as pd
import numpy as np
from utils.measure import Measure, Styling
from utils.read import DataProcessor, LogitToStat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--runs', metavar='1', type=int, 
                    help='run ID', default=1)
parser.add_argument('--n', metavar='1', type=int, 
                    help='number of inner runs', default=1)
parser.add_argument('--zero_lag', action='store_true',
                    help='Include iff foreground')
args = parser.parse_args()

path = '/data/gravwav/lopezm/Projects/GlitchBank/'
timeslides_file = 'timeslides_1132.9y.csv'
t_window_HV = 0.027 + 0.005 # time between H1 - V1 + fluctuations
t_window_HL = 0.010 + 0.005 # time between H1 - V1 + fluctuations
t_window_LV = 0.026 + 0.005 # time between H1 - V1 + fluctuations
zero_lag = args.zero_lag
runs = args.runs # run number
n = args.n # run number
inner_runs = runs + 1 * np.arange(n) # ID of run

if not zero_lag:
    path_store = path_store = path + 'runs/background/results/'
    sls = np.linspace((runs+1)*3, (runs+n)*3, n)  # L shifts
    svs = np.linspace((runs+1)*3, (runs+n)*3, n) + 6# V shifts
else:
    path_store = path + 'runs/zero_lag/'
    sls, svs = [0], [0]
print(zero_lag, 'zero_lag')

dp = DataProcessor(path + 'computational-aspects-of-machine-learning-project-3/output_new/tw0.05/predictions/', path + 'time_slides/', path + 'runs/injections')
dH1, dL1, dV1 = dp.getDataFrame('H1'),  dp.getDataFrame('L1'),  dp.getDataFrame('V1')
# WARNING: LogitToStat selects rows where Pinj is maximum
temperature = 1
dH1, dL1, dV1 = LogitToStat(dH1, temperature, 'Prob', 'Pinj'), LogitToStat(dL1, temperature, 'Prob', 'Pinj'), LogitToStat(dV1, temperature, 'Prob', 'Pinj')
print(len(dH1), len(dL1), len(dV1))
if not zero_lag:
    dH1s, dL1s_fix = dH1.copy(), dL1.copy()
    dH1s['Cluster time old'] = dH1['Cluster time']
    dL1s_fix['Cluster time old'] = dL1['Cluster time']
else:
    dH1s, dL1s, dL1s_fix, dV1s = dH1.copy(), dL1.copy(), dL1.copy(), dV1.copy()
    dH1s['Cluster time old'] = dH1s['Cluster time']
    dL1s['Cluster time old'] = dL1s['Cluster time']
    dL1s_fix['Cluster time old'] = dL1['Cluster time']
    dV1s['Cluster time old'] = dV1s['Cluster time']

for run, sl, sv in zip(inner_runs, sls, svs):

    start = time.time()
    print(run, sl, sv)
    # We read the data
    if not zero_lag:
        dL1s, dV1s = dp.shiftedDataSet(dL1, window=sl), dp.shiftedDataSet(dV1, window= sv)

    end = time.time()
    print('Data read in ', np.round(end - start, 2), ' s')

    var, tmp1, tmp2 = dH1s.copy(), dL1s.copy(), dV1s.copy()
    print('H L V', len(var), len(tmp1), len(tmp2))
    triggersHL, triggersHV, triggersHLV = list(), list(), list()
    start_time = time.time()
    for v in range(len(var)):
    
        # we get a time from H1 and its masses
        t_star = var.iloc[v]['Cluster time']

        # Time coincidence H1L1 & H1V1
        coinc1 = tmp1.loc[(tmp1['Cluster time'] >= t_star - t_window_HL) & (tmp1['Cluster time'] <= t_star + t_window_HL)]
        coinc2 = tmp2.loc[(tmp2['Cluster time'] >= t_star - t_window_HV) & (tmp2['Cluster time'] <= t_star + t_window_HV)]

        # if len(coinc1) > 0: # store HL coincidence
        #     triggerHL = Styling.load_triggers_2(var.iloc[v], coinc1.iloc[0])
        #     triggersHL.append(triggerHL)
        
        # if len(coinc2) > 0: # store HV coincidence
        #     triggerHV = Styling.load_triggers_2(var.iloc[v], coinc2.iloc[0])
        #     triggersHV.append(triggerHV)
        
        if (len(coinc1) > 0) and (len(coinc2) > 0): 
            c1, c2 = coinc1['Cluster time'].values[0], coinc2['Cluster time'].values[0]

            if (np.abs(c1 - c2) <= t_window_LV): # check also L1 and V1 are time coincident
                triggerHLV = Styling.load_triggers_3(var.iloc[v], coinc1.iloc[0], coinc2.iloc[0])
                print(triggerHLV)
                triggersHLV.append(triggerHLV) # store HLV coincidence
    print(len(triggersHL), len(triggersHV), len(triggersHLV))
    triggersHL = pd.DataFrame(triggersHL, columns=Styling.get_columns('HL'))
    triggersHV = pd.DataFrame(triggersHV, columns=Styling.get_columns('HV'))
    triggersHLV = pd.DataFrame(triggersHLV, columns=Styling.get_columns('HLV'))

    del var, tmp1, tmp2, coinc1, coinc2
    
    var, tmp1 = dL1s_fix.copy(), dV1s.copy()
    triggersLV = list()

    for v in range(len(var)):
        # we get a time from L1 and its masses
        t_star = var.iloc[v]['Cluster time']

        # Coincidence L1V1
        coinc1 = tmp1.loc[(tmp1['Cluster time'] >= t_star - t_window_LV) & (tmp1['Cluster time'] <= t_star + t_window_LV)]

        if len(coinc1) > 0: # store LV coincidence
            triggerLV = Styling.load_triggers_2(var.iloc[v], coinc1.iloc[0])
            triggersLV.append(triggerLV)
            
    triggersLV = pd.DataFrame(triggersLV, columns=Styling.get_columns('LV'))

    # if not zero_lag:
    #     triggersHL.to_csv(path_store + 'triggersHL_run'+str(run)+'.csv')
    #     triggersHV.to_csv(path_store + 'triggersHV_run'+str(run)+'.csv')
    #     triggersLV.to_csv(path_store + 'triggersLV_run'+str(run)+'.csv')
    #     triggersHLV.to_csv(path_store + 'triggersHLV_run'+str(run)+'.csv')
    # else:
    #     triggersHL.to_csv(path_store + 'triggersHL_zero_lag.csv')
    #     triggersHV.to_csv(path_store + 'triggersHV_zero_lag.csv')
    #     triggersLV.to_csv(path_store + 'triggersLV_zero_lag.csv')
    #     triggersHLV.to_csv(path_store + 'triggersHLV_zero_lag.csv')

    end_time = time.time()
    print("Time elapsed of one job:", end_time - start_time, "seconds")
