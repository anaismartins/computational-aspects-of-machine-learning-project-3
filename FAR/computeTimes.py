from gwdatafind import find_urls
from gwpy.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time
import numpy as np
import utils.timeslides as tu
pd.options.display.float_format = '{:.2f}'.format

start = 1238166018
end = 1269363618
dh = tu.getFrameTimes('H', 'H1_HOFT_CLEAN_SUB60HZ_C01', start, end)
dl = tu.getFrameTimes('L', 'L1_HOFT_CLEAN_SUB60HZ_C01', start, end)
dv = tu.getFrameTimes('V', 'V1Online', start, end)

N = 6100
start = time.time()
# GstLAL uses this time but FIXME
shiftL = np.arange(3, N, 3)
shiftV = np.arange(3, N, 3) + 6

matrix = np.empty((len(shiftL), 10))

th, tl, tv = tu.SingleCoincTime(dh), tu.SingleCoincTime(dl), tu.SingleCoincTime(dv)
for c, sl, sv in zip(range(len(shiftL)), shiftL, shiftV):
    if c % 10 == 0:
        print(c)
    start = time.time()
    thl = tu.DoubleCoincTime(dh, tu.Sliding(dl, sl))
    thv = tu.DoubleCoincTime(tu.Sliding(dl, sl), tu.Sliding(dv, sv))
    tlv = tu.DoubleCoincTime(dh, tu.Sliding(dv, sv))
    thlv = tu.TripleCoincTime(dh, tu.Sliding(dl, sl), tu.Sliding(dv, sv))

    matrix[c, 0],  matrix[c, 1],  matrix[c, 2], matrix[c, 3], matrix[c, 4] = 0, sl, sv, th, tl
    matrix[c, 5],  matrix[c, 6],  matrix[c, 7], matrix[c, 8], matrix[c, 9], = tv, thl, thv, tlv, thlv
    end = time.time()
    print(end - start)
end = time.time()
print(end - start)
matrix = pd.DataFrame(matrix, columns=['slideH', 'slideL', 'slideV',
                              'timeH', 'timeL', 'timeV', 
                              'timeHL', 'timeHV', 'timeLV', 'timeHLV'])

matrix.to_csv('timeslides_'+str(np.round(matrix['timeHLV'].sum(), 2))+'y.csv')