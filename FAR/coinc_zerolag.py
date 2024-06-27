import pandas as pd
import pickle
import sys
sys.path.insert(0, './utils')
from measure import ClusterProcessor
from read import DataProcessor
import time

path_pred = '/data/gravwav/lopezm/Projects/GlitchBank/computational-aspects-of-machine-learning-project-3/output_new/tw0.05/predictions/'
path_store = '/data/gravwav/lopezm/Projects/GlitchBank/runs/zero_lag/'

start_time = time.time()
u_h1, uc_h1 = DataProcessor.LoadZeroLag(path_pred, path_store, 'h1', reset=True)
u_l1, uc_l1 = DataProcessor.LoadZeroLag(path_pred, path_store, 'l1', reset=True)
u_v1, uc_v1 = DataProcessor.LoadZeroLag(path_pred, path_store, 'v1', reset=True)
end_time = time.time()
print(f'Loading data took {end_time-start_time} s')

cp = ClusterProcessor()

H1, L1, H2, V2, L3, V3, T1, T2, T3 = [], [], [], [], [], [], [], [], []
H4, L4, V4, T13, T23, T33 = [], [], [], [], [], []

for i in range(len(u_h1)):

    h1, l1, t1 = cp.coincTriggers2(i, u_h1, u_l1, uc_h1, uc_l1, ['_h1', '_l1'])
    h2, v2, t2 = cp.coincTriggers2(i, u_h1, u_v1, uc_h1, uc_v1, ['_h1', '_v1'])
    h4, l4, v4, t13, t23, t33 = cp.coincTriggers3(i, u_h1, u_l1,
                                                  u_v1, uc_h1,
                                                  uc_l1, uc_v1)
    H1.append(h1); L1.append(l1); H2.append(h2);
    V2.append(v2); T1.append(t1); T2.append(t2);
    H4.append(h4); L4.append(l4); V4.append(v4)
    T13.append(t13); T23.append(t23); T33.append(t33)

for i in range(len(u_l1)):
    l3, v3, t3 = cp.coincTriggers2(i, u_l1, u_v1, uc_l1, uc_v1, ['_l1', '_v1'])

    H1.append(h1); L1.append(l1); H2.append(h2)
    V2.append(v2); L3.append(l3); V3.append(v3)
    T1.append(t1); T2.append(t2); T3.append(t3)

df_hl = cp.mergeData2(u_h1, u_l1, H1, L1, ['_h1', '_l1'])
df_hv = cp.mergeData2(u_h1, u_v1, H2, V2, ['_h1', '_v1'])
df_lv = cp.mergeData2(u_l1, u_v1, L3, V3, ['_l1', '_v1'])
df_hlv = cp.mergeData3(u_v1, u_l1, u_v1, H4, L4, V4)

df_hl.to_pickle(path_store + 'unknown_hl_reduced.csv')
df_hv.to_pickle(path_store + 'unknown_hv_reduced.csv')
df_lv.to_pickle(path_store + 'unknown_lv_reduced.csv')
df_hlv.to_pickle(path_store + 'unknown_hlv_reduced.csv')