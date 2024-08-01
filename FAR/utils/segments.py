import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time
import numpy as np
pd.options.display.float_format = '{:.2f}'.format

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

def SingleCoincTime(data):
    analysis_time = data['duration'].sum()
    analysis_time = analysis_time /(365*24*60*60) # to give results in years
    return analysis_time

def Sliding(data, t):
    tmp = pd.DataFrame()
    tmp['start'] = data['start'] + t
    tmp['end'] = data['end'] + t
    tmp['duration'] = data['duration']
    return tmp
    
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
    start1, end1 = intervals1[:, 0], intervals1[:, 1]
    start2, end2 = intervals2[:, 0], intervals2[:, 1]
    
    intersect_start = np.maximum(start1[:, None], start2)
    intersect_end = np.minimum(end1[:, None], end2)
    
    valid = intersect_start < intersect_end
    intersections = np.vstack([intersect_start[valid], intersect_end[valid]]).T
    
    return intersections

def compute_difference(intervals, intervals_to_exclude):
    result = []
    for start, end in intervals:
        temp_intervals = [[start, end]]
        for ex_start, ex_end in intervals_to_exclude:
            new_temp_intervals = []
            for temp_start, temp_end in temp_intervals:
                if ex_end <= temp_start or ex_start >= temp_end:
                    # No overlap
                    new_temp_intervals.append([temp_start, temp_end])
                else:
                    # Overlap, split the interval
                    if temp_start < ex_start:
                        new_temp_intervals.append([temp_start, ex_start])
                    if temp_end > ex_end:
                        new_temp_intervals.append([ex_end, temp_end])
            temp_intervals = new_temp_intervals
        result.extend(temp_intervals)
    return np.array(result)

def AllCoincTime(data1, data2, data3, data2o):
    to_years = (365 * 24 * 60 * 60)
    # Precompute interval values for each dataset
    intervals1 = np.array(data1[['start', 'end']])
    intervals2 = np.array(data2[['start', 'end']]) # time slided
    intervals3 = np.array(data3[['start', 'end']])
    intervals2o = np.array(data2o[['start', 'end']]) # not time slided

    # Compute intersection of data1 and data2
    intersect12 = compute_intersection(intervals1, intervals2)

    # Compute intersection of intersect12 and data3 for triple coincidence
    intersect123 = compute_intersection(intersect12, intervals3)
    analysis_time_triple = 0
    if len(intersect123) > 0:
        analysis_time_triple = np.sum(intersect123[:, 1] - intersect123[:, 0]) / to_years

    # Compute difference of intersect12 and intervals where data3 is present
    start = time.time()
    coinc_12_n3 = compute_difference(intersect12, intervals3)
    end = time.time()
    print(end - start, 'difference 12 to 3')
    analysis_time_double_12_n3 = 0
    if len(coinc_12_n3) > 0:
        analysis_time_double_12_n3 = np.sum(coinc_12_n3[:, 1] - coinc_12_n3[:, 0]) / to_years

    # Compute intersection of data1 and data3
    intersect13 = compute_intersection(intervals1, intervals3)

    # Compute difference of intersect13 and intervals where data2 is present
    start = time.time()
    coinc_13_n2 = compute_difference(intersect13, intervals2)
    end = time.time()
    print(end - start, 'difference 13 to 2')
    analysis_time_double_13_n2 = 0
    if len(coinc_13_n2) > 0:
        analysis_time_double_13_n2 = np.sum(coinc_13_n2[:, 1] - coinc_13_n2[:, 0]) / to_years

    #     # Compute intersection of data2 and data3
    #     intersect23 = compute_intersection(intervals2, intervals3)

    #     # Compute difference of intersect23 and intervals where data1 is present
    #     start = time.time()
    #     coinc_23_n1 = compute_difference(intersect23, intervals1)
    #     end = time.time()
    #     print(coinc_23_n1, 'difference 23 to 1')
    #     analysis_time_double_23_n1 = 0
    #     if len(coinc_23_n1) > 0:
    #         analysis_time_double_23_n1 = np.sum(coinc_23_n1[:, 1] - coinc_23_n1[:, 0]) / to_years
        
    # Compute intersection of data2o and data3
    intersect2o3 = compute_intersection(intervals2o, intervals3)

    # Compute difference of intersect2o3 and intervals where data1 is present
    start = time.time()
    coinc_2o3_n1 = compute_difference(intersect2o3, intervals1)
    end = time.time()
    print(end - start, 'difference 2o3 to 1')
    analysis_time_double_2o3_n1 = 0
    if len(coinc_2o3_n1) > 0:
        analysis_time_double_2o3_n1 = np.sum(coinc_2o3_n1[:, 1] - coinc_2o3_n1[:, 0]) / to_years

    return (analysis_time_triple, analysis_time_double_12_n3,
            analysis_time_double_13_n2, analysis_time_double_2o3_n1,
            intersect123, coinc_12_n3, coinc_13_n2, coinc_2o3_n1)

def createDataFrame(data):
    if len(data) > 0:
        start = data[:, 0]
        end = data[:, 1]
        duration = data[:, 1] -  data[:, 0]
    else:
        start, end, duration = [None], [None], [None]
    data = pd.DataFrame({'start': start, 'end': end, 'duration': duration})
    return data
