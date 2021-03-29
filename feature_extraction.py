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
    # print("Set sliding windows:", window_count)
    for i in range(window_count):
        slides = np.append(slides, df[i*(window_size-overlap):i*(window_size-overlap)+window_size])
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

def time_reversal_asymmetry_statistic(x, lag):
    """
    This function calculates the value of

    .. math::

        \\frac{1}{n-2lag} \\sum_{i=1}^{n-2lag} x_{i + 2 \\cdot lag}^2 \\cdot x_{i + lag} - x_{i + lag} \\cdot  x_{i}^2

    which is

    .. math::

        \\mathbb{E}[L^2(X)^2 \\cdot L(X) - L(X) \\cdot X^2]

    where :math:`\\mathbb{E}` is the mean and :math:`L` is the lag operator. It was proposed in [1] as a
    promising feature to extract from time series.

    .. rubric:: References

    |  [1] Fulcher, B.D., Jones, N.S. (2014).
    |  Highly comparative feature-based time-series classification.
    |  Knowledge and Data Engineering, IEEE Transactions on 26, 3026–3037.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param lag: the lag that should be used in the calculation of the feature
    :type lag: int
    :return: the value of this feature
    :return type: float
    """
    n = len(x)
    x = np.asarray(x)
    if 2 * lag >= n:
        return 0
    else:
        one_lag = _roll(x, -lag)
        two_lag = _roll(x, 2 * -lag)
        return np.mean((two_lag * two_lag * one_lag - one_lag * x * x)[0:(n - 2 * lag)])