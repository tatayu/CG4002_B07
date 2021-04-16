import numpy as np
import pandas as pd
import random
import time
import joblib
import json

from statistics import mode
# from scipy import stats
# from scipy.signal import savgol_filter
import torch

import brevitas.nn as nn

from collections import Counter 

# from sklearn import preprocessing

from feature_extraction import *

from queue import Queue

clustering = False
deployed = True

class CNN(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(CNN, self).__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = torch.nn.Conv1d(in_channels=64,out_channels=64, kernel_size=5)
        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
#         self.conv4 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
#         self.conv5 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.lstm1 = torch.nn.LSTM(
            input_size=14,
            hidden_size=32,
            num_layers=2,
            batch_first=False,
        )
        self.fc1 = torch.nn.Linear(60, 26)
        self.fc2 = torch.nn.Linear(26, d_out)
        
        self.dropout = torch.nn.Dropout(p=0.3) 
        
    def forward(self,x):
        x = x.float().unsqueeze(dim=1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
#         x = self.relu(self.conv4(x))
#         x = self.dropout(x)
#         x = self.relu(self.conv5(x))
#         x,_ = self.lstm1(x)
        x = x[:, -1]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def predict(self, X):
        outputs = self(X.float())
        _, predicted = torch.max(outputs, 1)
        return predicted

class MLP(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(MLP, self).__init__()
        self.d_in = d_in

        self.linear1 = nn.QuantLinear(d_in, d_hidden, bias=True)
        self.linear2 = nn.QuantLinear(d_hidden, d_hidden, bias=True)
        self.linear3 = nn.QuantLinear(d_hidden, d_hidden//2, bias=True)
        self.linear4 = nn.QuantLinear(d_hidden//2, d_hidden//2, bias=True)
        self.linear5 = nn.QuantLinear(d_hidden//2, d_out, bias=False)

        self.dropout = torch.nn.Dropout(p=0.65)
        self.relu = torch.nn.ReLU()
        
    def forward(self, X):
        X = X.view(-1, self.d_in)
        X = self.relu(self.linear1(X.float()))
        X = self.dropout(X)
        X = self.relu(self.linear2(X))
        X = self.dropout(X)
        X = self.relu(self.linear3(X))
        X = self.dropout(X)
        X = self.relu(self.linear4(X))
        X = self.dropout(X)
        X = self.linear5(X)
        return torch.nn.functional.log_softmax(X, dim=1)
    
    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def predict(self, X):
        outputs = self(X.float())
        _, predicted = torch.max(outputs, 1)
        return predicted

class MLP_8(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(MLP_8, self).__init__()
        self.d_in = d_in

        self.linear1 = nn.QuantLinear(d_in, d_hidden, bias=True)
        self.linear2 = nn.QuantLinear(d_hidden, d_hidden//4, bias=True)
        self.linear3 = nn.QuantLinear(d_hidden//4, d_hidden//4, bias=True)
        self.linear4 = nn.QuantLinear(d_hidden//4, d_hidden//8, bias=True)
        self.linear5 = nn.QuantLinear(d_hidden//8, d_out, bias=False)

        self.dropout = torch.nn.Dropout(p=0.15)
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

def knn_test(data):
    if isinstance(data, pd.DataFrame):
        data = data.values
    knn = joblib.load(open('knn_model_8.pkl', 'rb'))
    try:
        result = list(knn.predict(data))
    except ValueError:
        result = []
    '''
    knn_2 = joblib.load(open('knn_model_8_2.pkl', 'rb'))
    try:
        result_2 = list(knn_2.predict(data))
    except ValueError:
        result_2 = []
    '''
    # knn_3 = joblib.load(open('knn_model_8_3.pkl', 'rb'))
    # try:
    #     result_3 = list(knn_3.predict(data))
    # except ValueError:
    #     result_3 = []
    # result.extend(result_2)
    # result.extend(result_3)
    return result

# from sequentia.classifiers import KNNClassifier
# def knn_dtw_test(data):
#     if isinstance(data, pd.DataFrame):
#         data = data.to_numpy()
#     # knn = joblib.load(open('knn_model_dtw.pkl', 'rb'))
#     # print([type(x) for _,x in knn.items()])
#     knn = KNNClassifier(k=30, classes=[x for x in range(1,9)], use_c=True)
#     knn_model = knn.load('knn_model_dtw.pkl')
#     try:
#         result = knn_model.predict(data)
#     except ValueError:
#         result = []
#     return result

def rf_test(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    rfc = joblib.load(open('rf_model.pkl', 'rb'))
    try:
        result = rfc.predict(data)
    except ValueError:
        result = []
    return result

# def _savgol_filter(dataset):
#     return savgol_filter(dataset, 5, 3, mode='nearest')

# def smoothing(dataset):
#     dataset = dataset[dataset.columns.difference(['dance'])]
#     return _savgol_filter(dataset)

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

def process(data):
    knn_result = knn_test(data)
    # if len(knn_result) > 0:
    #     print("Predicted tag for KNN:", stats.mode(knn_result).mode[0], end='  ')

    # rf_result = rf_test(data)
    # if len(rf_result) > 0:
    #     print("Predicted tag for RF:", stats.mode(rf_result).mode[0], end='  ')

    # mlp_model = MLP_8(72, 70, 8)
    # mlp_model.load('MLP_Model_moves_8_windowsize50_overlap48_epoch1')
    # mlp_model.eval()

    def print_weights():
        def lazy_print(row):
            print('{', end='')
            for _, x in enumerate(row):
                if _ == len(row) - 1:
                    print(x, end='}')
                else:
                    print(x, end=',')
            print(',')
        linear1_weights = mlp_model.linear1.weight.detach().numpy()
        linear1_bias = mlp_model.linear1.bias.detach().numpy()
        w_df = pd.DataFrame(linear1_weights)
        b_df = pd.DataFrame(linear1_bias)
        w_df.apply(lambda row : lazy_print(row))
        b_df.apply(lambda row : lazy_print(row))
    
    # test_torch = torch.from_numpy(np.array(data))
    # mlp_result = np.array(mlp_model.predict(test_torch))
    # mlp_result += 1
    # print("Predicted tag for MLP:", stats.mode(mlp_result).mode[0], end='  ')

    # cnn_model = CNN(72, 70, 8)
    # cnn_model.load('CNN_Model_moves_8_ws50_ol48_epoch15')
    # cnn_model.eval()
    # test_torch = torch.from_numpy(np.array(data))
    # cnn_result = np.array(cnn_model.predict(test_torch))
    # cnn_result += 1
    # print("Predicted tag for CNN:", stats.mode(cnn_result).mode[0])

    full_result = np.array([])
    full_result = np.append(full_result, knn_result)
    # full_result = np.append(full_result, rf_result)
    # full_result = np.append(full_result, mlp_result)
    # full_result = np.append(full_result, cnn_result)

    proba_dict = {}

    dances = ['dab', 'elbowkick', 'gun', 'hair', 'listen', 'pointhigh', 'sidepump', 'wipetable']
    # dances = ['gun', 'hair', 'sidepump']
    for x in full_result:
        x = dances[int(x)-1]
        if x not in proba_dict:
            proba_dict[x] = 1
        else:
            proba_dict[x] += 1
    for k in proba_dict.keys():
        proba_dict[k] /= len(full_result) / 100
        proba_dict[k] = float("%.2f" % proba_dict[k])
    sorted_dict = dict(sorted(proba_dict.items(), key=lambda item: -item[1]))
    # print("Percentage per prediction:", sorted_dict)

    return full_result

# Returns new pos_arr, dance move if is_dance and state (1: Dance, 2: Move, 3: Neither)
def check_confidence_level(pos_arr, *arrays):
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
    if full_count / len(full_arr) >= 0.4: # and full_mode not in is_moving:
        return pos_arr, int(full_mode), IS_DANCE

    # Return if mode of at least 1 equals 85% conf level.
    modes = []
    for arr in arrays:
        arr = np.array(arr)
        test_list1 = Counter(arr)  
        if len(test_list1.most_common(1)) == 0:
            continue
        mode = test_list1.most_common(1)[0][0]
        count = test_list1.most_common(1)[0][1]
        modes.append([mode,count/len(arr)])
        if count / len(arr) > 0.75 and mode == full_mode: # and full_mode not in is_moving:
            return pos_arr, int(full_mode), IS_DANCE
    
    # print("Skipped")
    return pos_arr, None, IS_NEITHER

    if full_count / len(full_arr) < 0.8 and full_mode not in is_moving:
        return pos_arr, None, IS_NEITHER
    elif full_count / len(full_arr) < 0.8 and full_mode == 5:
        return pos_arr, None, IS_NEITHER

    if len(modes) == 6:
        # 1,2,3 -> 2,3,1
        if ((modes[3][1] > 0.7 and modes[3][0] == 4) and 
            (modes[4][1] > 0.7 and modes[4][0] == 3) and 
            (modes[5][1] > 0.7 and modes[5][0] == 3)):
            pos_arr = [pos_arr[1], pos_arr[2], pos_arr[0]]
            return pos_arr, None, IS_MOVE
        # 1,2,3 -> 3,1,2
        if ((modes[3][1] > 0.7 and modes[3][0] == 4) and 
            (modes[4][1] > 0.7 and modes[4][0] == 4) and 
            (modes[5][1] > 0.7 and modes[5][0] == 3)):
            pos_arr = [pos_arr[2], pos_arr[0], pos_arr[1]]
            return pos_arr, None, IS_MOVE
        # 1,2,3 -> 1,3,2
        if ((modes[4][1] > 0.7 and modes[3][0] == 4) and 
            (modes[5][1] > 0.7 and modes[4][0] == 3)):
            pos_arr = [pos_arr[0], pos_arr[2], pos_arr[1]]
            return pos_arr, None, IS_MOVE
        # 1,2,3 -> 2,1,3
        if ((modes[3][1] > 0.7 and modes[3][0] == 4) and 
            (modes[4][1] > 0.7 and modes[4][0] == 3)):
            pos_arr = [pos_arr[1], pos_arr[0], pos_arr[2]]
            return pos_arr, None, IS_MOVE
        # 1,2,3 -> 3,2,1
        if ((modes[3][1] > 0.7 and modes[3][0] == 4) and 
            (modes[5][1] > 0.7 and modes[5][0] == 3)):
            pos_arr = [pos_arr[2], pos_arr[1], pos_arr[0]]
            return pos_arr, None, IS_MOVE

    # 1,2,3 -> 2,3,1
    if ((modes[0][1] > 0.6 and modes[0][0] == 4) and 
        (modes[1][1] > 0.6 and modes[1][0] == 3) and 
        (modes[2][1] > 0.6 and modes[2][0] == 3)):
        pos_arr = [pos_arr[1], pos_arr[2], pos_arr[0]]
        return pos_arr, None, IS_MOVE
    # 1,2,3 -> 3,1,2
    if ((modes[0][1] > 0.6 and modes[0][0] == 4) and 
        (modes[1][1] > 0.6 and modes[1][0] == 4) and 
        (modes[2][1] > 0.6 and modes[2][0] == 3)):
        pos_arr = [pos_arr[2], pos_arr[0], pos_arr[1]]
        return pos_arr, None, IS_MOVE
    # 1,2,3 -> 1,3,2
    if ((modes[1][1] > 0.6 and modes[1][0] == 4) and 
        (modes[2][1] > 0.6 and modes[2][0] == 3)):
        pos_arr = [pos_arr[0], pos_arr[2], pos_arr[1]]
        return pos_arr, None, IS_MOVE
    # 1,2,3 -> 2,1,3
    if ((modes[0][1] > 0.6 and modes[0][0] == 4) and 
        (modes[1][1] > 0.6 and modes[1][0] == 3)):
        pos_arr = [pos_arr[1], pos_arr[0], pos_arr[2]]
        return pos_arr, None, IS_MOVE
    # 1,2,3 -> 3,2,1
    if ((modes[0][1] > 0.6 and modes[0][0] == 4) and 
        (modes[2][1] > 0.6 and modes[2][0] == 3)):
        pos_arr = [pos_arr[2], pos_arr[1], pos_arr[0]]
        return pos_arr, None, IS_MOVE

    return pos_arr, None, IS_NEITHER

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
    if full_count / len(full_arr) >= 0.2: # and full_mode not in is_moving:
        return pos_arr, int(full_mode), IS_DANCE

    return pos_arr, None, IS_NEITHER

def knn_process(data):
    knn_result = knn_test(data)
    '''
    proba_dict = {}

    dances = ['dab', 'elbowkick', 'gun', 'hair', 'listen', 'pointhigh', 'sidepump', 'wipetable']
    # dances = ['gun', 'hair', 'sidepump']
    for x in knn_result:
        x = dances[int(x)-1]
        if x not in proba_dict:
            proba_dict[x] = 1
        else:
            proba_dict[x] += 1
    for k in proba_dict.keys():
        proba_dict[k] /= len(knn_result) / 100
        proba_dict[k] = float("%.2f" % proba_dict[k])
    sorted_dict = dict(sorted(proba_dict.items(), key=lambda item: -item[1]))
    # print("Predicting KNN")
    print("Percentage per prediction:", sorted_dict)
    '''
    return knn_result

def cnn_process(data):
    cnn_model = CNN(72, 70, 3)
    cnn_model.load('CNN_Model_moves_3_ws50_ol48_epoch15')
    cnn_model.eval()
    df_target = torch.from_numpy(np.array(data))
    cnn_result = cnn_model.predict(df_target)
    proba_dict = {}

    dances = ['hair', 'sidepump', 'gun']
    for x in cnn_result:
        x = dances[int(x)-1]
        if x not in proba_dict:
            proba_dict[x] = 1
        else:
            proba_dict[x] += 1
    for k in proba_dict.keys():
        proba_dict[k] /= len(cnn_result) / 100
        proba_dict[k] = float("%.2f" % proba_dict[k])
    sorted_dict = dict(sorted(proba_dict.items(), key=lambda item: -item[1]))
    print("Percentage per prediction:", sorted_dict)
    return cnn_result

def remap(c):
    for i in range(len(c)):
        if c[i] == 3:
            c[i] = 1
        elif c[i] == 4:
            c[i] = 2
        elif c[i] == 7:
            c[i] = 3
        else:
            c[i] = None
    return c

def knn_predict(moves, *arrays):
    collections = list(np.array([]))
    for x in arrays:
        x = feature_extraction(x)
        collection = knn_process(x)
        collections.append(collection)

    pos_arr, dance_move, return_type = check_confidence_level(
        [1,2,3], 
        *collections,
    )
    return pos_arr, dance_move, return_type 

def cnn_predict(moves, *arrays):
    collections = list(np.array([]))
    for x in arrays:
        x = feature_extraction(x)
        collection = cnn_process(x)
        print(collection)
        collections.append(collection)
    pos_arr, dance_move, return_type = check_confidence_level_2(
        [1,2,3], 
        *collections,
    )
    return pos_arr, dance_move, return_type 

# arr1 = [1,1,1,1,1,1,3]
# arr2 = [1,3,3,1,2,3,3]
# arr3 = [1,1,1,1,2,1,3]
# move_left = [3,3,3,4,3]
# move_right = [4,4,4,3,4]
# stationary = [5,5,5,5,5,5]

# print(check_confidence_level([1,2,3],arr1))
# print(check_confidence_level("1 2 3",arr1))
# print(check_confidence_level([1,2,3],arr1, arr2, arr3))
# print(check_confidence_level([1,2,3],arr3, arr1, arr3))

# print(check_confidence_level([1,2,3],stationary))
# print(check_confidence_level([1,2,3],move_right,move_left,move_left))
# print(check_confidence_level([1,2,3],move_right,move_left,arr2))
# print(check_confidence_level([1,2,3],move_left,move_left,move_left))
# print(check_confidence_level([1,2,3],move_right,stationary,move_left,move_right,stationary,move_left))
# print(check_confidence_level([1,2,3],move_right,move_right,move_left,move_right,stationary,move_left))
