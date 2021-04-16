import numpy as np
import pandas as pd

import math
import torch

from queue import Queue

from collections import Counter 

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, Softmax, BatchNorm1d, Dropout, Flatten

from config import *
from hardware_accelerator import HardwareAccelerator

from sklearn import preprocessing

window_size = 120
overlap = 80

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

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, Softmax, BatchNorm1d, Dropout, Flatten

class CNN(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(CNN, self).__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = torch.nn.Conv1d(in_channels=64,out_channels=64, kernel_size=5)
        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.conv4 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.conv5 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.conv6 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
#         self.lstm1 = torch.nn.LSTM(
#             input_size=14,
#             hidden_size=32,
#             num_layers=2,
#             batch_first=False,
#         )
        self.fc1 = torch.nn.Linear(48, 26)
        self.fc2 = torch.nn.Linear(26, d_out)
        
#         self.dropout = torch.nn.Dropout(p=0.3) 
        
    def forward(self,x):
        x = x.float().unsqueeze(dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = x[:, -1]
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
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
    windows = set_sliding_windows(df, overlap, window_size)
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
    print(full_features_df)
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
    
    df = data.apply(pd.to_numeric)#.interpolate(method='polynomial', order=3)
    col = df.columns
    max_arr = [16377, 15638, 16383, 357, 363, 421]
    min_arr = [-16382, -14090, -8726, -340, -337, -412]
    max_arr = [x/1.75 for x in max_arr]
    min_arr = [x/1.75 for x in min_arr]
    def normalize(df):
        result = df.copy()
        for i, feature_name in enumerate(df.columns):
            max_val = max_arr[i]
            min_val = min_arr[i]
            result[feature_name] = (df[feature_name] - min_val) / (max_val - min_val)
        return result
    df_scaled = normalize(df)
    # print(df_scaled)
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)
    # df_scaled = df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
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

def cnn_process(data):
    # cnn_model = CNN(72, 64, 8)
    # cnn_model.load('CNN_Model_moves_8_ws120_ol110_epoch1')
    # cnn_model.eval()
    cnn_model = HardwareAccelerator()#DESIGN_PATH, INPUT_SIZE, OUTPUT_SIZE)
    # df_target = torch.from_numpy(np.array(data))
    # cnn_result = cnn_model.predict(df_target)
    '''
    try:
        print(np.array(data).shape)
    except Exception as e:
        print(e)
    '''
    data = np.array(data)
    cnn_result = []
    try:
        for x in data:
            x = x.reshape(1,-1)
            cnn_result.extend(cnn_model.predict(x))
    except Exception as e:
        print(e)
    return cnn_result

def check_performance(result):
    print(result)
    dances = ['dab', 'elbowkick', 'gun', 'hair', 'listen', 'pointhigh', 'sidepump', 'wipetable']
    proba_dict = {}
    # dances = ['hair', 'sidepump', 'gun']
    for x in result:
        x = dances[int(x)]
        if x not in proba_dict:
            proba_dict[x] = 1
        else:
            proba_dict[x] += 1
    for k in proba_dict.keys():
        proba_dict[k] /= len(result) / 100
        proba_dict[k] = float("%.2f" % proba_dict[k])
    sorted_dict = dict(sorted(proba_dict.items(), key=lambda item: -item[1]))
    # print("Predicting CNN")
    print("Percentage per prediction:", sorted_dict)
    return

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
        arr = [int(x) for x in arr]
        full_arr.extend(arr)
    full_counter = Counter(full_arr)  
    if len(full_counter.most_common(1)) == 0:
        return pos_arr, None, IS_NEITHER

    full_mode = full_counter.most_common(1)[0][0]
    full_count = full_counter.most_common(1)[0][1]

    for arr in arrays:
        arr = np.array(arr)
        arr = [int(x) for x in arr]
        counter = Counter(arr)
        mode = counter.most_common(1)[0][0]
        count = counter.most_common(1)[0][1]
        if count / len(arr) == 1 and len(arr) > 1 and mode != 1:
            return pos_arr, int(mode), IS_DANCE
    # print("Mode:", full_mode, "Count:", full_count, "Full length:", len(full_arr))
    if full_mode != 1: # and full_mode not in is_moving:
        return pos_arr, int(full_mode), IS_DANCE

    has_elbowkick = 0
    for arr in arrays:
        arr = np.array(arr)
        arr = [int(x) for x in arr]
        if 1 in arr:
            has_elbowkick += 1
        # arr_counter = Counter(arr)
        # mode = arr_counter.most_common(1)[0][0]
        # count = arr_counter.most_common(1)[0][1]
    if has_elbowkick < 2 and int(full_mode) == 1:
        return pos_arr, int(full_counter.most_common(2)[0][0]),IS_DANCE

    return pos_arr, int(full_mode), IS_DANCE

def cnn_predict(moves, *arrays):
    collections = list(np.array([]))
    for x in arrays:
        x = feature_extraction(x)
        collection = cnn_process(x)
        collections.append(collection)
    pos_arr, dance_move, return_type = check_confidence_level_2(
        [1,2,3], 
        *collections,
    )
    return pos_arr, dance_move, return_type 

from results import knn_process

def predict(moves, *arrays):
    collections = list(np.array([]))
    for i, x in enumerate(arrays):
        print("Data ", i+1, x.qsize())
        if x.qsize() < 120:
            continue
        inner_collections = list(np.array([]))
        x = feature_extraction(x)
        collection = cnn_process(x)
        inner_collections.extend(int(x) for x in list(collection))
        collection = knn_process(x)
        collection = [x-1 for x in collection]
        inner_collections.extend((int(x) for x in list(collection)))
        check_performance(inner_collections)
        collections.append(inner_collections)
    pos_arr, dance_move, return_type = check_confidence_level_2(
        [1,2,3], 
        *collections,
    )
    return pos_arr, dance_move, return_type 
