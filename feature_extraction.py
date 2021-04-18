import numpy as np
import pandas as pd
import math

from config import *
from data_preprocessing import *
from feature_extraction import *
from helpers import *

from scipy.stats import skew 

from tsfresh.feature_extraction import feature_calculators

def feature_extract(df, window_size):
    full_features = np.array([])
    axis = ['accel1', 'accel2', 'accel3', 'gyro1', 'gyro2', 'gyro3']
    titles = np.ravel(np.array([i+'_'+j for i in feature_list for j in axis]))
    if not deployed:
        titles = np.append(titles,["tag"])

    # print("Begin Feature Extraction")
    windows = set_sliding_windows(df, overlap, window_size)
    # windows = set_windows(df, window_size)
    
    print("Window count:", len(windows))
    for j, window in enumerate(windows):
        if j % 100 == 0:
            print("Stage:", j)
        for i,ax in enumerate(window.T):
            if i == 6 and not deployed:
                # Adding tag
                full_features = np.append(full_features, ax[0])
            else:
                full_features = np.append(full_features, add_mean(ax))
                full_features = np.append(full_features, add_max(ax))
                full_features = np.append(full_features, add_min(ax))
                full_features = np.append(full_features, add_median(ax))
                full_features = np.append(full_features, add_gradient(ax))
                full_features = np.append(full_features, add_std(ax))
                full_features = np.append(full_features, add_iqr(ax))
                # full_features = np.append(full_features, add_skew(ax))
                full_features = np.append(full_features, add_zero_crossing_count(ax))
                # full_features = np.append(full_features, add_cwt(ax))
                full_features = np.append(full_features, add_no_peaks(ax))
                full_features = np.append(full_features, add_recurring_dp(ax))
                # full_features = np.append(full_features, add_ratio_v_tsl(ax))
                # full_features = np.append(full_features, add_sum_recurring_dp(ax))
                full_features = np.append(full_features, add_var_coeff(ax))
                full_features = np.append(full_features, add_kurtosis(ax)) 
                
    if deployed:
        full_features = full_features.reshape(
            -1,
            len(feature_list) * 6,
        )
    else:
        full_features = full_features.reshape(
            -1,
            len(feature_list) * 6 + 1,
        )        
    full_features_df = pd.DataFrame(full_features)
    full_features_df.columns = titles
    return full_features_df

def set_windows(df, window_size):
    max_rows = window_size * math.floor(len(df)/window_size)
    df = df[:max_rows]
    return np.array(df).reshape(-1, window_size, df.shape[1])

def set_sliding_windows(df, overlap, window_size):
    if overlap >= window_size:
        return set_windows(df, window_size)
    window_count = math.ceil((len(df)-window_size+1)/(window_size-overlap))
    slides = np.array([])
    print("Set sliding windows:", window_count)
    for i in range(window_count):
        if i % 100 == 0:
            print("Sliding:", i)
        slides = np.append(slides, df[i*(window_size-overlap):i*(window_size-overlap)+window_size])
    slides = slides.reshape(
        window_count,
        window_size,
        df.shape[1]
    )
    return slides


def _roll(a, shift):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])

def add_max(data):
    return np.max(data)

def add_min(data):
    return np.min(data)

def add_median(data):
    return np.median(data)

def add_mean(data):
    return np.mean(data)

def add_gradient(data):
    return np.mean(np.gradient(data))

def add_std(data):
    return np.std(data)

def add_iqr(data):
    return np.percentile(data, 75) - np.percentile(data, 25)

def add_zero_crossing_count(data):
    zero_crossings = np.where(np.diff(np.signbit(data)))[0]
    return len(zero_crossings)

def add_dominant_frequency(data):
    w = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data))
    i = np.argmax(abs(w))
    dom_freq = abs(freqs[i])
    return dom_freq

def number_peaks(x, n):
    x_reduced = x[n:-n]

    res = None
    for i in range(1, n + 1):
        result_first = (x_reduced > _roll(x, i)[n:-n])

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= (x_reduced > _roll(x, -i)[n:-n])
    return np.sum(res)

def add_no_peaks(data):
    return number_peaks(data, 3)

def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    if len(x) == 0:
        return np.nan

    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    value_counts = x.value_counts()
    reoccuring_values = value_counts[value_counts > 1].sum()

    if np.isnan(reoccuring_values):
        return 0

    return reoccuring_values / x.size

def add_recurring_dp(data):
    return percentage_of_reoccurring_datapoints_to_all_datapoints(data)

def variation_coefficient(x):
    """
    Returns the variation coefficient (standard error / mean, give relative value of variation around mean) of x.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan

def add_var_coeff(data):
    return variation_coefficient(data)

def kurtosis(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)

def add_kurtosis(data):
    return kurtosis(data)

def sum_of_reoccurring_data_points(x):
    """
    Returns the sum of all data points, that are present in the time series
    more than once.

    For example

        sum_of_reoccurring_data_points([2, 2, 2, 2, 1]) = 8

    as 2 is a reoccurring value, so all 2's are summed up.

    This is in contrast to ``sum_of_reoccurring_values``,
    where each reoccuring value is only counted once.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :return: the value of this feature
    :return type: float
    """
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    return np.sum(counts * unique)