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
