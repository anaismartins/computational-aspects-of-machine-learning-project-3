from gwdatafind import find_urls
from gwpy.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time
import numpy as np
import argparse
import sys
sys.path.insert(0, './utils')
from utils import segments as tu
pd.options.display.float_format = '{:.2f}'.format

parser = argparse.ArgumentParser()
parser.add_argument('--job_start', metavar='1', type=int,
                    help='Start of jobs')
parser.add_argument('--N', metavar='1', type=int,
                    help='Number of jobs')
parser.add_argument('--zerolag', action='store_true',
                    help='Include iff foreground')

args = parser.parse_args()
s = args.job_start
N = args.N
zerolag = args.zerolag
path = '/data/gravwav/lopezm/Projects/GlitchBank/runs/frames/times/'
start, end = 1238166018, 1269363618

# We call the frames extracted from CIT
path_frames = '/data/gravwav/lopezm/Projects/GlitchBank/runs/frames/'
dh = pd.read_csv(path_frames + 'framesH1.csv') #tu.getFrameTimes('H', 'H1_HOFT_CLEAN_SUB60HZ_C01', start, end)
dl = pd.read_csv(path_frames + 'framesL1.csv') #tu.getFrameTimes('L', 'L1_HOFT_CLEAN_SUB60HZ_C01', start, end)
dv = pd.read_csv(path_frames + 'framesV1.csv') #tu.getFrameTimes('V', 'V1Online', start, end)

start = time.time()
if zerolag:
    # For zero lag we just need to check the search time
    matrix = np.empty((1, 11))
    (thlv, thl_nv, thv_nl, tlv_nh, coinc_hlv,
     coinc_hl_nv, coinc_hv_nl, coinc_lv_nh) = tu.AllCoincTime(dh, dl, dv)
    matrix[c, 0],  matrix[c, 1], matrix[c, 2],  matrix[c, 3], matrix[c, 4], matrix[c, 5] = 0, sl, sv, 0, th, tl
    matrix[c, 6],  matrix[c, 7],  matrix[c, 8], matrix[c, 9], matrix[c, 10], = tv, thl_nv, thv_nl, thv_nl, thlv
    np.save(path + 'time_search_zerolag.npy', matrix)
else:
    # These are the time slides iterations
    # GstLAL uses this time but FIXME
    shiftL = np.arange(3*s, 3*(N+s), 3)
    shiftV = np.arange(3*s, 3*(N+s), 3) + 6
    
    matrix = np.empty((len(shiftL), 11))
    
    th, tl, tv = tu.SingleCoincTime(dh), tu.SingleCoincTime(dl), tu.SingleCoincTime(dv)
    
    for c, sl, sv in zip(range(len(shiftL)), shiftL, shiftV):
        if c % 1 == 0:
            print(c)
        start = time.time()
        (thlv, thl_nv, thv_nl, tlv_nh, coinc_hlv,
         coinc_hl_nv, coinc_hv_nl, coinc_lv_nh) = tu.AllCoincTime(dh, tu.Sliding(dl, sl), tu.Sliding(dl, sv), dl)
        
        coinc_hlv = tu.createDataFrame(coinc_hlv)
        coinc_hl_nv = tu.createDataFrame(coinc_hl_nv)
        coinc_hv_nl = tu.createDataFrame(coinc_hv_nl)
        coinc_lv_nh = tu.createDataFrame(coinc_lv_nh)
        
        coinc_hlv.to_csv(path + f'coinc_hlv_sh0_sl{sl}_sv{sv}')
        coinc_hl_nv.to_csv(path + f'coinc_hl_nv_sh0_sl{sl}_sv{sv}')
        coinc_hv_nl.to_csv(path + f'coinc_hv_nl_sh0_sl{sl}_sv{sv}')
        coinc_lv_nh.to_csv(path + f'coinc_lv_nh_sh0_sl{sl}_sv{sv}')
        
        print(coinc_hlv.shape, coinc_hl_nv.shape, coinc_hv_nl.shape, coinc_lv_nh.shape)
        coinc_hlv 
        print(0, sl, sv, 0, thlv, thl_nv, thv_nl, tlv_nh)
        matrix[c, 0],  matrix[c, 1], matrix[c, 2],  matrix[c, 3], matrix[c, 4], matrix[c, 5] = 0, sl, sv, 0, th, tl
        matrix[c, 6],  matrix[c, 7],  matrix[c, 8], matrix[c, 9], matrix[c, 10], = tv, thl_nv, thv_nl, tlv_nh, thlv
        end = time.time()
        print(end - start)
    end = time.time()
    end = time.time()
    print(end - start)
    np.save(path + f'time_background_{N}_{s}.npy', matrix)
    #matrix.to_csv('timeslides_'+str(np.round(matrix['timeHLV'].sum(), 2))+'y.csv')