import time
import sys
import argparse
import pandas as pd
import numpy as np
sys.path.insert(0, './utils')
from utils.measure import ClusterProcessor
from utils.read import DataProcessor
from utils.utilities import selectInProperTime, selectInProper3Time

parser = argparse.ArgumentParser()
parser.add_argument('--runs', metavar='1', type=int,
                    help='run ID', default=1)
parser.add_argument('--n', metavar='1', type=int,
                    help='number of inner runs', default=1)
parser.add_argument('--zero_lag', action='store_true',
                    help='Include iff foreground')
args = parser.parse_args()

path = '/data/gravwav/lopezm/Projects/GlitchBank/'
timeslides = pd.read_csv(path + 'computational-aspects-of-machine-learning-project-3/FAR/timeslides_1132.9y.csv')
t_window_HV = 0.027 + 0.005  # time between H1 - V1 + fluctuations
t_window_HL = 0.010 + 0.005  # time between H1 - V1 + fluctuations
t_window_LV = 0.026 + 0.005  # time between H1 - V1 + fluctuations
zero_lag = args.zero_lag
runs = args.runs # run number
n = args.n # run number
inner_runs = runs + 1 * np.arange(n) # ID of run
cp = ClusterProcessor()
if not zero_lag:
    path_store = path + 'runs/background/results/'
else:
    path_store = path + 'runs/zero_lag/'

print(zero_lag, 'zero_lag')

dp = DataProcessor(path + 'computational-aspects-of-machine-learning-project-3/output_new/tw0.05/predictions/', path + 'runs/zero_lag/', path + 'runs/injections')

# We call the frames extracted from CIT
path_frames = '/data/gravwav/lopezm/Projects/GlitchBank/runs/frames/'
fh = pd.read_csv(path_frames + 'framesH1.csv')  # tu.getFrameTimes('H', 'H1_HOFT_CLEAN_SUB60HZ_C01', start, end)
fl = pd.read_csv(path_frames + 'framesL1.csv')  # tu.getFrameTimes('L', 'L1_HOFT_CLEAN_SUB60HZ_C01', start, end)
fv = pd.read_csv(path_frames + 'framesV1.csv')  # tu.getFrameTimes('V', 'V1Online', start, end)

# We load the datasets 
dH1, dH1_c = dp.LoadZeroLag('h1', reset=True)
dL1, dL1_c = dp.LoadZeroLag('l1', reset=True)
dV1, dV1_c = dp.LoadZeroLag('v1', reset=True)
print(len(dH1), len(dL1), len(dV1))
print(dH1.columns)
print(f'We are going to do {len(inner_runs)} runs')

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

for run in inner_runs:

    start = time.time()
    
    # We read the data
    if not zero_lag:
        sh, sl, sv = timeslides['slideH'].iloc[run], timeslides['slideL'].iloc[run], timeslides['slideV'].iloc[run]
        dL1s, dV1s = dp.shiftedDataSet(dL1, window=sl), dp.shiftedDataSet(dV1, window=sv)
    else:
        sh, sl, sv = 0, 0, 0 
    print(run, sl, sv)
    # For each iteration we copy from original zero lag or time shifted data
    var, tmp1, tmp2 = dH1s.copy(), dL1s.copy(), dV1s.copy()
    print('H L V', len(var), len(tmp1), len(tmp2))
    triggersHL, triggersHV, triggersHLV = list(), list(), list()
    start_time = time.time()

    H1, L1, H2, V2, L3, V3, T1, T2, T3 = [], [], [], [], [], [], [], [], []
    H4, L4, V4, T13, T23, T33 = [], [], [], [], [], []
    for i in range(len(var)):

        h1, l1, t1 = cp.coincTriggers2(i, var, tmp1, dH1_c, dL1_c, ['_H1', '_L1'], shift1=sh, shift2=sl)
        h2, v2, t2 = cp.coincTriggers2(i, var, tmp2, dH1_c, dV1_c, ['_H1', '_V1'], shift1=sh, shift2=sv)

        h4, l4, v4, t13, t23, t33 = cp.coincTriggers3(i, var, tmp1,
                                                      tmp2, dH1_c,
                                                      dL1_c, dV1_c, shift1=sh, shift2=sl, shift3=sv)
        H1.append(h1); L1.append(l1); H2.append(h2);
        V2.append(v2); T1.append(t1); T2.append(t2);
        H4.append(h4); L4.append(l4); V4.append(v4)
        T13.append(t13); T23.append(t23); T33.append(t33)

    triggersHL = cp.mergeData2(var, tmp1, H1, L1, ['_H1', '_L1'])

    triggersHV = cp.mergeData2(var, tmp2, H2, V2, ['_H1', '_V1'])

    triggersHLV = cp.mergeData3(var, tmp1, tmp2, H4, L4, V4)

    del var, tmp1, tmp2

    var, tmp1 = dL1s_fix.copy(), dV1s.copy()
    
    for i in range(len(var)):

        l3, v3, t3 = cp.coincTriggers2(i, var, tmp1, dL1_c, dV1_c, ['_L1', '_V1'], shift1=0, shift2=sv)
    
        L3.append(l3); V3.append(v3); T3.append(t3)
    triggersLV = cp.mergeData2(var, tmp1, L3, V3, ['_L1', '_V1'])
    end = time.time()
    print('Single iteration took ', np.round(end - start, 2), ' s')

        # We define double detection in double time, and in tripple time
    selectInProperTime('H1L1', triggersHL, sh, sl, sv, run, path_store, zero_lag)
    selectInProperTime('H1V1', triggersHV, sh, sl, sv, run, path_store, zero_lag)
    selectInProperTime('L1V1', triggersLV, sh, sl, sv, run, path_store, zero_lag)
    selectInProper3Time('H1L1V1', triggersHLV, sh, sl, sv, run, path_store, zero_lag)
    # triggersHL.to_csv(path_store + f'triggersHL_run_eq_match{run}_{ptime}.csv')
    # triggersHV.to_csv(path_store + f'triggersHV_run_eq_match{run}_{ptime}.csv')
    # triggersLV.to_csv(path_store + f'triggersLV_run_eq_match{run}_{ptime}.csv')
    # triggersHLV.to_csv(path_store + f'triggersHLV_run_eq_match{run}_{ptime}.csv')
    end_time = time.time()
    print("Time elapsed of one job:", end_time - start_time, "seconds")
