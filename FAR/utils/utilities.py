import pandas as pd
import numpy as np
import pickle
import sys
sys.path.insert(1, '../')
from FAR.utils.measure import Measure

def getFARCatalog(foreground, xbkg_obs, ybkg_obs, catalog, ifo, label='FAR_', offset=0.1 ):
    fars = []
    for e in range(len(catalog)):
        t_star = catalog['GPS'].iloc[e]
        cond1 = (foreground['Cluster time_'+ifo[:2]] >= t_star - offset)
        cond2 = (foreground['Cluster time_'+ifo[:2]] <= t_star + offset)
        tmp = foreground[cond1 & cond2]
        if len(tmp) ==  1:
            ranking = tmp['rank_stat_'+ifo].values[0]
            far, _ = Measure.stat2FAR(ranking, xbkg_obs, ybkg_obs) #FIXME
            if far <= 0.1:
                print(far, catalog['Name'].iloc[e], Measure.FAR2stat(0.01, xbkg_obs, ybkg_obs))
                display(tmp)
        else:
            far = np.nan
        fars.append(far)
    catalog[label+ifo] = fars
    return catalog

def searchTime(data, time, col):
    cond1 = (data[col] >= time - 0.2)
    cond2 = (data[col] <= time + 0.2)
    tmp = data[cond1 & cond2]
    return tmp

def rename_columns(col, sub_old, sub_new):
    if col.endswith(sub_old):
        return col[:-len(sub_old)] + sub_new
    return col