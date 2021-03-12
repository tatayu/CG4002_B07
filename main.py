import numpy as np
import pandas as pd
import random
import time
import json

from config import *
from classic_models import *
from data_preprocessing import *
from feature_extraction import *
from helpers import *

from statistics import mode
from scipy import stats
import torch

from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv1d, MaxPool1d, Module, Softmax, BatchNorm1d, Dropout, Flatten

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

def main():
    train, test = consolidate_data()
    f = open('features.json',) 
    json_file = json.load(f)
    print(json_file)
    if deployed:
        for i in range(testing_count):
            testset, tag = generate_test_data(clustering=clustering, df=test)
            t0 = time.time()
            smoothed_dataset = smoothing(testset, deployed)
            test_x = feature_extract(smoothed_dataset, window_size=window_size).reset_index(drop=True)
            # test_x = test_x[json_file[:-1]]
            print("############# Actual tag:", tag, end='  ')
            knn_result = knn_test(test_x)
            print("Predicted tag for KNN:", stats.mode(knn_result).mode[0], end='  ')

            # rf_result = rf_test(test_x)
            # print("Predicted tag for RF:", stats.mode(rf_result).mode[0], end='  ')

            cnn_model = CNN(54, 50, 3)
            cnn_model.load('CNN_Model')
            cnn_model.eval()
            test_torch = torch.from_numpy(np.array(test_x))
            cnn_result = np.array(cnn_model.predict(test_torch))
            print("Predicted tag for CNN:", stats.mode(cnn_result).mode[0]+1, end='  ')

            mlp_model = MLP(54, 50, 3)
            mlp_model.load('MLP_Model')
            mlp_model.eval()
            test_torch = torch.from_numpy(np.array(test_x))
            mlp_result = np.array(mlp_model.predict(test_torch))
            print("Predicted tag for MLP:", stats.mode(mlp_result).mode[0]+1, end='  ')

            t1 = time.time()

            print("Time taken:", t1-t0)

    else:
        smoothed_dataset = smoothing(train, deployed)
        train = feature_extract(smoothed_dataset, window_size=window_size).reset_index(drop=True)
        
        feature_importance = random_forest(train, dim=2, save=True)
        # feature_list = feature_importance[:15].reset_index()['index'].to_list()
        # feature_list.append('tag')
        # train = train[feature_list]
        train.to_csv('out_2.csv', index=False)
        # print(train.head())

        # with open('features.json', 'w') as f:
        #     json.dump(feature_list, f)

        # _ = random_forest(train, dim=2, save=True)
        svm_accuracy = svm(train, size=window_size, save=True)
        print("Support Vector Machine Accuracy:", svm_accuracy)
        print()
        if clustering:
            knn_accuracy = knn(train, dim=1, save=True)
        else:
            knn_accuracy = knn(train, dim=1, save=True)
        print("K-Nearest Neighbour Accuracy:", knn_accuracy)

        print(feature_importance)
    return

if __name__ == "__main__":
    main()