import pandas as pd
import numpy as np
import pickle
import os
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

def is_within_segments(cluster_time, segments):
    return any((cluster_time >= row['start']) and (cluster_time <= row['end']) for _, row in segments.iterrows())


def selectProperTriggers(df_clusters, df_segments, ifo):
    print(df_clusters.columns, 'potato', 'Cluster time_'+ifo)
    # Apply the function to each row of df_clusters
    df_clusters['within_segment'] = df_clusters['Cluster time_'+ifo].apply(lambda x: is_within_segments(x, df_segments))
    df_clusters = df_clusters[df_clusters['within_segment'] == True]
    return df_clusters


def selectInProperTime(ifo, data, sh, sl, sv, run, path_store, zero_lag=False):

    path_frames = '/data/gravwav/lopezm/Projects/GlitchBank/runs/frames/times/'
    str1, str2 = ifo.replace('1', '').lower(), 'hlv'
    no_ifo = list(set(str2) - set(str1))[0]
    ifo_2ptime = ifo.replace('1', '').lower() + '_n' + no_ifo
    ifo_3ptime = str2
    
    pdt_2ptime = pd.read_csv(path_frames + f'coinc_{ifo_2ptime}_sh{int(sh)}_sl{int(sl)}_sv{int(sv)}', index_col=0)
    pdt_3ptime = pd.read_csv(path_frames + f'coinc_{ifo_3ptime}_sh{int(sh)}_sl{int(sl)}_sv{int(sv)}', index_col=0)

    triggers_2ptime, triggers_3ptime = data.copy(), data.copy()
    triggers_2ptime = selectProperTriggers(triggers_2ptime, pdt_2ptime, ifo[:2])
    triggers_3ptime = selectProperTriggers(triggers_3ptime, pdt_3ptime, ifo[:2])
    print('original', len(data), 'double time', len(triggers_2ptime), 'tripple time', len(triggers_3ptime))

    if zero_lag:
        zl = '_zero_lag'
    else:
        zl = ''
    data.to_csv(path_store + 'triggers'+ifo.replace('1', '')+f'{zl}_run_eq_match{run}_original.csv')
    triggers_2ptime.to_csv(path_store + 'triggers'+ifo.replace('1', '')+f'{zl}_run_eq_match{run}_{ifo_2ptime}.csv')
    triggers_3ptime.to_csv(path_store + 'triggers'+ifo.replace('1', '')+f'{zl}_run_eq_match{run}_{ifo_3ptime}.csv')

def selectInProper3Time(ifo, data, sh, sl, sv, run, path_store, zero_lag=False):

    path_frames = '/data/gravwav/lopezm/Projects/GlitchBank/runs/frames/times/'
    ifo_3ptime = ifo.replace('1', '').lower()
    pdt_3ptime = pd.read_csv(path_frames + f'coinc_{ifo_3ptime}_sh{int(sh)}_sl{int(sl)}_sv{int(sv)}', index_col=0) 

    if zero_lag:
        zl = '_zero_lag'
    else:
        zl = ''
    data.to_csv(path_store + 'triggers'+ifo.replace('1', '')+f'{zl}_run_eq_match{run}_original.csv')

    triggers_3ptime = data.copy()
    triggers_3ptime = selectProperTriggers(triggers_3ptime, pdt_3ptime, ifo[:2])
    triggers_3ptime.to_csv(path_store + 'triggers'+ifo.replace('1', '')+f'{zl}_run_eq_match{run}_{ifo_3ptime}.csv')

def CountProperTime(path_tproper = '/data/gravwav/lopezm/Projects/GlitchBank/runs/frames/times/'):
    """
        This function counts the proper detector time per time slide.
    """
    c = 0
    for file in os.listdir(path_tproper):
        if 'time_background' in file:
            tmp = np.load(path_tproper + file)
            if c > 0:
                tproper = np.vstack([tproper, tmp])
            else:
                tproper = tmp.copy()
            c = c + 1
    #  print(0, sl, sv, 0, thlv, thl_nv, thv_nl, tlv_nh)

    tproper = pd.DataFrame(tproper, columns=['dH1s', 'dL1s', 'dV1s','dL1s_fix',
                                             'H', 'L', 'V', 'HL_nV', 'HV_nL', 'LV_nV', 'HLV'])
    tproper = tproper.sort_values(by=['dL1s', 'dV1s'])
    return tproper
