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

def remove_little_dogs(catalog, bkg, ifo, offset = 0.1):
    ids = []
    for c in range(len(catalog)):
        t_star = catalog.iloc[c]['GPS']
        if len(ifo) == 4: 
            ifo1, ifo2 = ifo[:2], ifo[2:]
            ifo_list = [ifo1, ifo2]
        if len(ifo) == 6:
            ifo1, ifo2, ifo3 = ifo[:2], ifo[2:4], ifo[4:]
            ifo_list = [ifo1, ifo2, ifo3]

        for i in ifo_list:
            cond1 = (bkg['Cluster time old_' + i] >= t_star - offset)
            cond2 = (bkg['Cluster time old_' + i] <= t_star + offset)
            tmp = bkg[cond1 & cond2]
            if len(tmp)>0:
                # print(i)
                # display(tmp[['Pinj_'+ifo1, 'Pinj_'+ifo2, 'rank_stat_'+ifo]])
                ids.append(tmp.index.values[0])
    ids = np.unique(ids)
    return bkg.drop(ids)

def rename_data(ifo, data):

    if len(ifo) == 4:
        ifo1, ifo2 = ifo[:2], ifo[2:]
    if len(ifo) == 6:
        ifo1, ifo2, ifo3 = ifo[:2], ifo[2:4], ifo[4:]
        
    data.rename(columns=lambda col: rename_columns(col, ifo1.lower(), ifo1), inplace=True)
    data.rename(columns=lambda col: rename_columns(col, ifo2.lower(), ifo2), inplace=True)
    data.rename(columns=lambda col: rename_columns(col, 
                                                   ifo.lower().replace('1', ""), ifo), inplace=True)
    if len(ifo) == 6:
        data.rename(columns=lambda col: rename_columns(col, ifo3.lower(), ifo3), inplace=True)
    return data