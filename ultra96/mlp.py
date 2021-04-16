import numpy as np
import pandas as pd

import math
import torch

from queue import Queue

from collections import Counter 

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, Softmax, BatchNorm1d, Dropout, Flatten

import brevitas.nn as nn

import warnings

warnings.filterwarnings("ignore")

window_size = 80

feature_list = [
    'mean', 
    'max', 
    'min', 
    'median', 
    'gradient', 
    'std', 
    'iqr', 
    # 'skew', 
    'zero_crossing',
    # 'cwt', 
    'no_peaks', 
    'recurring_dp', 
    # 'ratio_v_tsl', 
    # 'sum_recurring_dp', 
    'var_coeff', 
    'kurtosis'
]

class MLP(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(MLP, self).__init__()
        self.d_in = d_in

        self.linear1 = nn.QuantLinear(d_in, d_hidden, bias=True)
        self.linear2 = nn.QuantLinear(d_hidden, d_hidden//4, bias=True)
        self.linear3 = nn.QuantLinear(d_hidden//4, d_hidden//4, bias=True)
        self.linear4 = nn.QuantLinear(d_hidden//4, d_hidden//8, bias=True)
        self.linear5 = nn.QuantLinear(d_hidden//8, d_out, bias=False)

        # self.dropout = torch.nn.Dropout(p=0.15)
        self.relu = torch.nn.ReLU()
        
    def forward(self, X):
        X = X.view(-1, self.d_in)
        X = self.relu(self.linear1(X.float()))
        # X = self.dropout(X)
        X = self.relu(self.linear2(X))
        # X = self.dropout(X)
        X = self.relu(self.linear3(X))
        # X = self.dropout(X)
        X = self.relu(self.linear4(X))
        # X = self.dropout(X)
        X = self.linear5(X)
        return torch.nn.functional.log_softmax(X, dim=1)
    
    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def predict(self, X):
        outputs = self(X.float())
        _, predicted = torch.max(outputs, 1)
        return predicted

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
        # if i % 100 == 0:
            # print("Sliding:", i)
        slides = np.append(slides, df[i*(window_size-overlap):i*(window_size-overlap)+window_size])
    slides = slides.reshape(
        window_count,
        window_size,
        df.shape[1]
    )
    return slides

def feature_extract(df, window_size):
    full_features = np.array([])
    axis = ['accel1', 'accel2', 'accel3', 'gyro1', 'gyro2', 'gyro3']
    titles = np.ravel(np.array([i+'_'+j for i in feature_list for j in axis]))

    # print("Begin Feature Extraction")
    windows = set_sliding_windows(df, 75, window_size)
    # windows = set_windows(df, window_size)

    for window in windows:
        for _,ax in enumerate(window.T):
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

    full_features = full_features.reshape(
        -1,
        len(feature_list) * 6,
    )   
    full_features_df = pd.DataFrame(full_features)
    full_features_df.columns = titles
    return full_features_df

def feature_extraction(data):
    if isinstance(data, Queue):
        data = np.array(list(data.queue))
        if data.shape[0] % 6 != 0:
            data = data[:len(data) - data.shape[0]%6]
        arr = data.reshape(-1,6)
        data = pd.DataFrame(arr, columns=['accel1', 'accel2', 'accel3', 'gyro1', 'gyro2', 'gyro3'])
        # del data['timestamp']
    else:
        data = pd.DataFrame.from_dict(data)
        if 'dance' in data:
            del data['dance']

    # print(data)

    df = data.apply(pd.to_numeric).interpolate(method='polynomial', order=2)
    col = df.columns
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)
    df_scaled = df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # df_scaled = min_max_scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=col)
    df.reset_index(drop=True, inplace=True)

    # smoothed_dataset = _savgol_filter(df)
    # print(df)
    features = feature_extract(df, window_size=window_size).reset_index(drop=True)
    # print(features.shape)
    return features


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

def mlp_process(data):
    mlp_model = MLP(72, 70, 3)
    mlp_model.load('MLP_Model_moves_3_windowsize50_overlap48_epoch1')
    mlp_model.eval()
    df_target = torch.from_numpy(np.array(data))
    mlp_result = mlp_model.predict(df_target)
    proba_dict = {}

    dances = ['hair', 'sidepump', 'gun']
    for x in mlp_result:
        x = dances[int(x)-1]
        if x not in proba_dict:
            proba_dict[x] = 1
        else:
            proba_dict[x] += 1
    for k in proba_dict.keys():
        proba_dict[k] /= len(mlp_result) / 100
        proba_dict[k] = float("%.2f" % proba_dict[k])
    sorted_dict = dict(sorted(proba_dict.items(), key=lambda item: -item[1]))
    print("Percentage per prediction:", sorted_dict)
    return mlp_result

def check_confidence_level_2(pos_arr, *arrays):
    # 3: move left, 4: move right, 5: stationary
    is_moving = [4,5,6]
    full_arr = []

    IS_DANCE = 1
    IS_MOVE = 2
    IS_NEITHER = 3

    if isinstance(pos_arr, str):
        pos_arr = [int(x) for x in pos_arr.split()]
    pos_arr = [int(x) for x in pos_arr]
    # Returns if mode of 3 ppl equals 85% conf level.
    for arr in arrays:
        arr = np.array(arr)
        full_arr.extend(arr)
    full_counter = Counter(full_arr)  
    if len(full_counter.most_common(1)) == 0:
        # print("Skipped. No mode.")
        return pos_arr, None, IS_NEITHER
    full_mode = full_counter.most_common(1)[0][0]
    full_count = full_counter.most_common(1)[0][1]
    if full_count / len(full_arr) >= 0.6: # and full_mode not in is_moving:
        return pos_arr, int(full_mode), IS_DANCE

    return pos_arr, None, IS_NEITHER

def mlp_predict(moves, *arrays):
    collections = list(np.array([]))
    for x in arrays:
        x = feature_extraction(x)
        collection = mlp_process(x)
        collections.append(collection)
    pos_arr, dance_move, return_type = check_confidence_level_2(
        [1,2,3], 
        *collections,
    )
    return pos_arr, dance_move, return_type 
