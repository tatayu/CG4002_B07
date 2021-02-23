import numpy as np
import pandas as pd
import random

from config import *
from data_preprocessing import *
from helpers import *

def consolidate_data(clustering):
    print("Consolidating Data")
    full_df = None
    for j, entry in enumerate(entries):
        df = pd.read_csv(directory + '/' + entry) 
        df = initialize(df)
        tagged_df = data_tagging(df, j)

        if full_df is None:
            full_df = tagged_df
        else:
            full_df = pd.concat([full_df, tagged_df])
    if clustering:
        full_df = clustering_test(full_df)
    return full_df

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

def data_split(features_set):
    msk = np.random.rand(len(features_set)) < 0.8
    train = features_set[msk]
    test = features_set[~msk]
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    return train, test

def generate_test_data(clustering, df):
    if clustering:
        rand = random.randint(1,2)
    else:
        rand = random.randint(1,len(dances))

    unselected, selected = df[df['tag']!=rand], df[df['tag']==rand]
    if selected.shape[0] < unselected.shape[0]:
        unselected = unselected.sample(frac=selected.shape[0]/unselected.shape[0] * 0.2)

    print("Noise count:", unselected.shape)
    print("Normal Data count:", selected.shape)
    return pd.concat([unselected, selected]).sample(frac=1).iloc[:, :-1], rand