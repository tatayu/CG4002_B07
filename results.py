import numpy as np
import pandas as pd
import random
import time
import joblib

from statistics import mode
from scipy import stats
from scipy.signal import savgol_filter
import torch

from sklearn import preprocessing

from feature_extraction import *

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
        self.fc2 = torch.nn.Linear(46, d_out)
        
    def forward(self,x):
        x = x.float().unsqueeze(dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
#         x = self.relu(self.conv3(x))
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

        self.linear1 = torch.nn.Linear(d_in, d_hidden)
        self.linear2 = torch.nn.Linear(d_hidden, d_hidden)
        self.linear3 = torch.nn.Linear(d_hidden, d_out)

    def forward(self, X):
        X = X.view(-1, self.d_in)
        X = self.linear1(X.float())
        X = self.linear2(X)
        X = self.linear3(X)
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
    result = knn.predict(data)
    return result

def rf_test(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    rfc = joblib.load(open('rf_model.pkl', 'rb'))
    result = rfc.predict(data)
    return result

def _savgol_filter(dataset):
    return savgol_filter(dataset, 5, 3, mode='nearest')

def smoothing(dataset):
    dataset = dataset[dataset.columns.difference(['dance'])]
    return _savgol_filter(dataset)

def run():
    f = open('test_gun.json',) 
    json_file = json.load(f)
    df = pd.DataFrame.from_dict(json_file)

    df = df.apply(pd.to_numeric).dropna()
    x = df.values #returns a numpy array
    col = df.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(df)
    df = pd.DataFrame(df_scaled, columns=col)
    df.reset_index(drop=True, inplace=True)

    smoothed_dataset = smoothing(df)
    test_x = feature_extract(smoothed_dataset, window_size=50).reset_index(drop=True)
    knn_result = knn_test(test_x)
    print("Predicted tag for KNN:", stats.mode(knn_result).mode[0], end='  ')

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
    print("Predicted tag for CNN:", stats.mode(cnn_result).mode[0], end='  ')

    knn_result = np.append(knn_result, mlp_result)
    knn_result = np.append(knn_result, cnn_result)

    return stats.mode(knn_result).mode[0]

run()