from gwdatafind import find_urls
from gwpy.timeseries import TimeSeries
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time
import numpy as np
pd.options.display.float_format = '{:.2f}'.format

def getFrameTimes(ifo, channel, start, end):
    urls = find_urls(ifo, channel, start, end, host="datafind.ldas.cit:80",)
    starts, durations, ends = list(), list(), list()
    for u in range(len(urls)):
        tmp = urls[u]
        infile = "/" + "/".join(tmp.split('/')[3:])

        start = int(infile.split('/')[-1].split('-')[-2])
        duration = int(infile.split('/')[-1].split('-')[-1][:-4])
        end = start + duration
        starts.append(start)
        durations.append(duration)
        ends.append(end)
    df = pd.DataFrame({'start':starts, 'end':ends, 'duration':durations})
    df = df.sort_values(by='start')
    return df


def Sliding(data, t):
    tmp = pd.DataFrame()
    tmp['start'] = data['start'] + t
    tmp['end'] = data['end'] + t
    tmp['duration'] = data['duration']
    return tmp

    
def SingleCoincTime(data):
    analysis_time = data['duration'].sum()
    analysis_time = analysis_time /(365*24*60*60) # to give results in years
    return analysis_time

def DoubleCoincTime(data1, data2):
    analysis_time = 0

    # Precompute interval values
    intervals1 = np.array(data1[['start', 'end']])
    intervals2 = np.array(data2[['start', 'end']])

    # Perform vectorized intersection check
    start_max = np.maximum(intervals1[:, 0][:, None], intervals2[:, 0])
    end_min = np.minimum(intervals1[:, 1][:, None], intervals2[:, 1])
    intersect = start_max <= end_min

    # Compute analysis time
    dframes = np.where(intersect, end_min - start_max, 0)
    analysis_time = np.sum(dframes) / (365*24*60*60)  # Convert to years

    return analysis_time

def TripleCoincTime(data1, data2, data3):
    analysis_time = 0

    # Precompute interval values for each dataset
    intervals1 = np.array(data1[['start', 'end']])
    intervals2 = np.array(data2[['start', 'end']])
    intervals3 = np.array(data3[['start', 'end']])

    # Compute intersection of data1 and data2
    intersect12 = compute_intersection(intervals1, intervals2)

    # Compute intersection of data2 and data3
    intersect23 = compute_intersection(intervals2, intervals3)

    # Compute intersection of intersect12 and intersect23
    intersect123 = compute_intersection(intersect12, intersect23)

    # Compute analysis time
    analysis_time = np.sum(intersect123[:, 1] - intersect123[:, 0]) / (365 * 24 * 60 * 60)  # Convert to years

    return analysis_time

def compute_intersection(intervals1, intervals2):
    """
    Compute intersection of two sets of intervals.
    """
    start_max = np.maximum(intervals1[:, 0][:, None], intervals2[:, 0])
    end_min = np.minimum(intervals1[:, 1][:, None], intervals2[:, 1])
    intersect = start_max <= end_min

    # Filter intersecting intervals
    intersect_intervals = np.column_stack((start_max[intersect], end_min[intersect]))

    return intersect_intervals