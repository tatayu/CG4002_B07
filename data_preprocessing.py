import numpy as np
import pandas as pd
import random

from config import *
from data_preprocessing import *
from helpers import *

from scipy.signal import savgol_filter
from sklearn import preprocessing

def _consolidate_data():
    print("Consolidating Data")
    train_df = None
    test_df = None
    i = -1
    for j, entry in enumerate(entries):
        df = pd.read_csv(directory + '/' + entry) 
        df = initialize(df)

        if entry.rsplit('_')[0] not in dances:
            dances[entry] = 1
            i += 1
        tagged_df = data_tagging(df, i)
        
        if train_df is None:
            train_df = tagged_df[:int(len(tagged_df) * 1/40)]
        else:
            train_df = pd.concat([train_df, tagged_df[:int(len(tagged_df) * 1/40)]])
        if test_df is None:
            test_df = tagged_df[int(len(tagged_df) * 39/40):]
        else:
            test_df = pd.concat([test_df, tagged_df[int(len(tagged_df) * 39/40):]])
    train_temp, test_temp = train_df['tag'], test_df['tag']
    del train_temp['tag'], test_temp['tag']

    col = train_df.columns
    # min_max_scaler = preprocessing.MinMaxScaler()
    # train_scaled = min_max_scaler.fit_transform(train_df.values)
    # train_scaled = min_max_scaler.fit_transform(test_df.values)

    train_scaled = train_df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    test_scaled = test_df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

    train_df = pd.DataFrame(train_scaled, columns=col)
    test_df = pd.DataFrame(test_scaled, columns=col)
    train_df['tag'], test_df['tag'] = train_temp, test_temp

    return train_df.astype(float), test_df.astype(float)

def consolidate_data():
    print("Consolidating Data")
    train_df = None
    test_df = None
    i = -1
    dances_dict = {}

    temp = 0
    for j, entry in enumerate(entries):
        f = open(directory + '/' + entry,) 
        json_file = json.load(f)
        df = pd.DataFrame.from_dict(json_file)
        if entry.rsplit('_')[0][:-1] not in dances_dict:
            dances_dict[entry.rsplit('_')[0][:-1]] = 1
            i += 1
        tagged_df = data_tagging(df, i)
        
        if train_df is None:
            train_df = tagged_df[int(tagged_df.shape[0] * 4/30):int(tagged_df.shape[0] * 23/30)]
        else:
            train_df = pd.concat([train_df, tagged_df[int(tagged_df.shape[0] * 4/30):int(tagged_df.shape[0] * 23/30)]])
        if test_df is None:
            test_df = tagged_df[int(tagged_df.shape[0] * 23/30):int(tagged_df.shape[0] * 27/30)]
        else:
            test_df = pd.concat([test_df, tagged_df[int(tagged_df.shape[0] * 23/30):int(tagged_df.shape[0] * 27/30)]])
    
    if 'dance' in train_df:
        del train_df['dance']
    if 'dance' in test_df:
        del test_df['dance']

    # train_df = train_df.apply(pd.to_numeric).dropna()
    # test_df = test_df.apply(pd.to_numeric).dropna()

    train_temp = train_df[['tag']]
    test_temp = test_df[['tag']]
    train_df = train_df.drop(['tag'], axis=1)
    test_df = test_df.drop(['tag'], axis=1)

    train_df = train_df.apply(pd.to_numeric).interpolate(method='polynomial', order=2)
    test_df = test_df.apply(pd.to_numeric).interpolate(method='polynomial', order=2)

    x = train_df.values #returns a numpy array
    col = train_df.columns
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    train_scaled = train_df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    train_df = pd.DataFrame(train_scaled, columns=col)
    # robust_scaler = preprocessing.RobustScaler()
    # train_scaled = robust_scaler.fit_transform(x)

    train_df.reset_index(drop=True, inplace=True)
    train_temp.reset_index(drop=True, inplace=True)
    train_df = pd.concat([train_df,train_temp], axis=1)

    x = test_df.values #returns a numpy array
    col = test_df.columns
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    test_scaled = test_df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    test_df = pd.DataFrame(test_scaled, columns=col)

    test_df.reset_index(drop=True, inplace=True)
    test_temp.reset_index(drop=True, inplace=True)
    test_df = pd.concat([test_df,test_temp], axis=1)

    return train_df, test_df

def test_consolidate_data():
    print("Consolidating Data")
    train_df = None
    test_df = None
    i = -1
    dances_dict = {}

    temp = 0
    import os
    directory = 'new_data_collection'
    entries = sorted(os.listdir(directory))

    for j, entry in enumerate(entries):
        f = open(directory + '/' + entry,) 
        df = pd.read_csv(f)
        df.columns = ['accel1', 'accel2', 'accel3', 'gyro1', 'gyro2', 'gyro3', 'timestamp']
        df = df[df.columns[:-1]]
        print(entry.split('_')[-1][:-4])
        if entry.split('_')[-1][:-4] == 'hand' and entry.split('_')[-2] not in dances_dict:
            print('a')
            dances_dict[entry.split('_')[-2]] = 1
            i += 1
        tagged_df = data_tagging(df, i)
        
        if train_df is None:
            train_df = tagged_df[int(tagged_df.shape[0] * 7/30):int(tagged_df.shape[0] * 23/30)]
        else:
            train_df = pd.concat([train_df, tagged_df[int(tagged_df.shape[0] * 7/30):int(tagged_df.shape[0] * 23/30)]])
        if test_df is None:
            test_df = tagged_df[int(tagged_df.shape[0] * 23/30):int(tagged_df.shape[0] * 27/30)]
        else:
            test_df = pd.concat([test_df, tagged_df[int(tagged_df.shape[0] * 23/30):int(tagged_df.shape[0] * 27/30)]])
    
    if 'dance' in train_df:
        del train_df['dance']
    if 'dance' in test_df:
        del test_df['dance']

    # train_df = train_df.apply(pd.to_numeric).dropna()
    # test_df = test_df.apply(pd.to_numeric).dropna()

    print(train_df)
    train_temp = train_df[['tag']]
    test_temp = test_df[['tag']]
    train_df = train_df.drop(['tag'], axis=1)
    test_df = test_df.drop(['tag'], axis=1)

    train_df = train_df.apply(pd.to_numeric).interpolate(method='polynomial', order=2)
    test_df = test_df.apply(pd.to_numeric).interpolate(method='polynomial', order=2)

    x = train_df.values #returns a numpy array
    col = train_df.columns
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    train_scaled = train_df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    train_df = pd.DataFrame(train_scaled, columns=col)
    # robust_scaler = preprocessing.RobustScaler()
    # train_scaled = robust_scaler.fit_transform(x)

    train_df.reset_index(drop=True, inplace=True)
    train_temp.reset_index(drop=True, inplace=True)
    train_df = pd.concat([train_df,train_temp], axis=1)

    x = test_df.values #returns a numpy array
    col = test_df.columns
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    test_scaled = test_df.apply(lambda x: (x - min(x)) / (max(x) - min(x)))
    test_df = pd.DataFrame(test_scaled, columns=col)

    test_df.reset_index(drop=True, inplace=True)
    test_temp.reset_index(drop=True, inplace=True)
    test_df = pd.concat([test_df,test_temp], axis=1)

    print(train_df, test_df)
    return train_df, test_df

def data_tagging(df,j):
    df_x = df.copy()
    df_x['tag'] = j+1
    return df_x

def initialize(df):
    df.columns = ['init'] # dummy
    df = df.init.str.split(expand=True)
    df.columns = ['X', 'Y', 'Z']
    df = df.apply(pd.to_numeric)
    return df 

def clustering_test(df):
    df['tag'] = df['tag'].apply(lambda x: 2 if x < 8 else 1)
    return df

def data_split(df):
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    return train, test

def generate_test_data(clustering, df):
    if clustering:
        rand = random.randint(1,2)
    else:
        rand = random.randint(1,len(dances))
    selected = df[df['tag']==rand][0:75]

    del selected['tag']
    mu, sigma = 0, 0.03
    noise = np.random.normal(mu, sigma, [75,7]) 
    selected = selected + noise
    return selected, rand

# https://ieeexplore.ieee.org/document/8713728
def smoothing(dataset, deployed):
    if deployed:
        dataset = dataset[dataset.columns.difference(['dance'])]
        return dataset #_savgol_filter(dataset)
    train, test = dataset[dataset.columns.difference(['tag', 'dance'])], dataset['tag']
    dataset = _savgol_filter(train)
    train['tag'] = test
    return train

def _savgol_filter(dataset):
    return savgol_filter(dataset, 5, 3, mode='nearest')

import json

def save_as_json(df):
    df['X_'], df['Y_'], df['Z_'] = df['X'], df['Y'], df['Z']
    df['State'] = True
    return df.to_json(orient="records")