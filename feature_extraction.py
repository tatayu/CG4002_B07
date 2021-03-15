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

    print("Begin Feature Extraction")
    windows = set_sliding_windows(df, window_size//10, window_size)
    # windows = set_windows(df, window_size)
    
    for window in windows:
        for i,ax in enumerate(window.T):
            if i == 6 and not deployed:
                # Adding tag
                full_features = np.append(full_features, ax[0])
            else:
                full_features = np.append(full_features, add_mean(ax))
                full_features = np.append(full_features, add_max(ax))
                full_features = np.append(full_features, add_min(ax))
                full_features = np.append(full_features, add_median(ax))
                # full_features = np.append(full_features, add_gradient(ax))
                full_features = np.append(full_features, add_std(ax))
                # full_features = np.append(full_features, add_iqr(ax))
                # full_features = np.append(full_features, add_skew(ax))
                # full_features = np.append(full_features, add_zero_crossing_count(ax))
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

def set_sliding_windows(df, shift, window_size):
    window_count = math.ceil((len(df)-window_size+1)/shift)
    slides = np.array([])
    for i in range(window_count):
        slides = np.append(slides, df[i*shift:i*shift+window_size])
    slides = slides.reshape(
        window_count,
        window_size,
        df.shape[1]
    )
    return slides

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

def add_skew(data):
    return skew(data)

def add_zero_crossing_count(data):
    zero_crossings = np.where(np.diff(np.signbit(data)))[0]
    return len(zero_crossings)

def add_dominant_frequency(data):
    w = np.fft.fft(data)
    freqs = np.fft.fftfreq(len(data))
    i = np.argmax(abs(w))
    dom_freq = abs(freqs[i])
    return dom_freq

def add_cwt(data):
    return feature_calculators.number_cwt_peaks(data, 3)

def add_no_peaks(data):
    return feature_calculators.number_peaks(data, 3)

def add_recurring_dp(data):
    return feature_calculators.percentage_of_reoccurring_datapoints_to_all_datapoints(data)

def add_ratio_v_tsl(data):
    return feature_calculators.ratio_value_number_to_time_series_length(data)

def add_sum_recurring_dp(data):
    return feature_calculators.sum_of_reoccurring_data_points(data)

def add_var_coeff(data):
    return feature_calculators.variation_coefficient(data)

def add_sample_entropy(data):
    return feature_calculators.sample_entropy(data)

def add_abs_energy(data):
    return feature_calculators.abs_energy(data)/1000

def add_kurtosis(data):
    return feature_calculators.kurtosis(data)