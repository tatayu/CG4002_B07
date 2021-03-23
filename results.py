import numpy as np
import pandas as pd
import random
import time
import joblib

from statistics import mode
from scipy import stats
from scipy.signal import savgol_filter
import torch

import brevitas.nn as nn

from sklearn import preprocessing

from feature_extraction import *

from queue import Queue

window_size = 50

clustering = False
deployed = True

class CNN(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super(CNN, self).__init__()
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5)
        self.conv2 = torch.nn.Conv1d(in_channels=64,out_channels=64, kernel_size=5)
        self.conv3 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.conv4 = torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.lstm1 = torch.nn.LSTM(
            input_size=14,
            hidden_size=32,
            num_layers=2,
            batch_first=False,
        )
        self.fc2 = torch.nn.Linear(42, d_out)
        
        self.dropout = torch.nn.Dropout(p=0.15)
        
    def forward(self,x):
        x = x.float().unsqueeze(dim=1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.relu(self.conv3(x))
#         x = self.relu(self.conv4(x))
#         x,_ = self.lstm1(x)
        x = x[:, -1]
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
        self.linear2 = nn.QuantLinear(d_hidden, d_hidden//4, bias=True)
        self.linear3 = nn.QuantLinear(d_hidden//4, d_out, bias=False)

        self.dropout = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.ReLU()
        
    def forward(self, X):
        X = X.view(-1, self.d_in)
        X = self.relu(self.linear1(X.float()))
        # X = self.dropout(X)
        X = self.relu(self.linear2(X))
        X = self.linear3(X)
        # X = self.dropout(X)
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
        data = data.to_numpy()
    knn = joblib.load(open('knn_model.pkl', 'rb'))
    try:
        result = knn.predict(data)
    except ValueError:
        result = []
    return result

def rf_test(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    rfc = joblib.load(open('rf_model.pkl', 'rb'))
    try:
        result = rfc.predict(data)
    except ValueError:
        result = []
    return result

def _savgol_filter(dataset):
    return savgol_filter(dataset, 5, 3, mode='nearest')

def smoothing(dataset):
    dataset = dataset[dataset.columns.difference(['dance'])]
    return _savgol_filter(dataset)

def run(data):
    if isinstance(data, Queue) and data.qsize() % 6 != 0:
        return
    arr = np.array(data.queue).reshape(-1,6)
    df = pd.DataFrame(arr, columns=['accel1', 'accel2', 'accel3', 'gyro1', 'gyro2', 'gyro3'])
    # df['dance'] = True
    # print(df.head())

    # df = pd.read_csv('gun.csv')
    # df.columns = ['accel1', 'accel2', 'accel3', 'gyro1', 'gyro2', 'gyro3']
    # df = df[100:200]
    # print(df.head())

    df = df.apply(pd.to_numeric).interpolate(method='polynomial', order=2)
    x = df.values
    col = df.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=col)
    df.reset_index(drop=True, inplace=True)

    smoothed_dataset = _savgol_filter(df)
    test_x = feature_extract(smoothed_dataset, window_size=50).reset_index(drop=True)

    knn_result = knn_test(test_x)
    if len(knn_result) > 0:
        print("Predicted tag for KNN:", stats.mode(knn_result).mode[0], end='  ')

    # rf_result = rf_test(test_x)
    # if len(rf_result) > 0:
    #     print("Predicted tag for RF:", stats.mode(rf_result).mode[0], end='  ')

    mlp_model = MLP(54, 50, 3)
    mlp_model.load('MLP_Model')
    mlp_model.eval()
    test_torch = torch.from_numpy(np.array(test_x))
    mlp_result = np.array(mlp_model.predict(test_torch))
    mlp_result += 1
    print("Predicted tag for MLP:", stats.mode(mlp_result).mode[0], end='  ')

    cnn_model = CNN(54, 50, 3)
    cnn_model.load('CNN_Model')
    cnn_model.eval()
    test_torch = torch.from_numpy(np.array(test_x))
    cnn_result = np.array(cnn_model.predict(test_torch))
    cnn_result += 1
    print("Predicted tag for CNN:", stats.mode(cnn_result).mode[0])

    full_result = np.array([])
    full_result = np.append(full_result, knn_result)
    # full_result = np.append(full_result, rf_result)
    full_result = np.append(full_result, mlp_result)
    full_result = np.append(full_result, cnn_result)

    proba_dict = {}

    for x in full_result:
        x = dances[int(x)-1]
        if x not in proba_dict:
            proba_dict[x] = 1
        else:
            proba_dict[x] += 1
    for k in proba_dict.keys():
        proba_dict[k] /= len(full_result) / 100
        proba_dict[k] = float("%.2f" % proba_dict[k])

    print("Percentage per prediction:", dict(sorted(proba_dict.items(), key=lambda item: -item[1])))

    return ''.join(dances[stats.mode(full_result).mode[0]-1].rsplit('_')[0].split('_'))

f = open('gun_test.json',) 
data = json.load(f)
data = Queue()
for i in range(480):
    data.put(i*5)
print("Dance move:", run(data))