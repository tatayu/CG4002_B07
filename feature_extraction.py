import numpy as np
import pandas as pd
import math

from config import *
from data_preprocessing import *
from feature_extraction import *
from helpers import *

from scipy.stats import skew 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def feature_extract(df, window_size):
    full_features = np.array([])
    axis = ['x', 'y', 'z']
    titles = np.ravel(np.array([i+'_'+j for i in feature_list for j in axis]))
    titles = np.append(titles,["tag"])

    print("Begin Feature Extraction")
    windows = set_sliding_windows(df, window_size//5, window_size)
    # windows = set_windows(df, window_size)
    
    for window in windows:
        for i,ax in enumerate(window.T):
            if i == 3:
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
                full_features = np.append(full_features, add_skew(ax))
                full_features = np.append(full_features, add_zero_crossing_count(ax))
                full_features = np.append(full_features, add_dominant_frequency(ax))

    full_features = full_features.reshape(
        -1,
        len(titles)
    )
    full_features_df = pd.DataFrame(full_features)
    full_features_df.columns = titles
    print(full_features_df)
    return full_features_df

def set_windows(df, window_size):
    max_rows = window_size * math.floor(len(df)/window_size)
    df = df[:max_rows]
    return np.array(df).reshape(-1, window_size, df.shape[1])

def set_sliding_windows(df, shift, window_size):
    window_count = math.floor((len(df)-window_size)/shift)
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
