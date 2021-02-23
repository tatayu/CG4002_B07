import numpy as np
import pandas as pd
import random

from config import *
from classic_models import *
from data_preprocessing import *
from feature_extraction import *
from helpers import *

from statistics import mode

def main():
    if deployed:
        dataset = collect_data(data)
        # features_set = feature_extract(dataset, window_size=10)
    else:
        dataset = consolidate_data(clustering=clustering)
        features_set = feature_extract(dataset, window_size=50).reset_index(drop=True)
        # features_set.to_csv('out.csv', index=False)
        train, test = data_split(features_set)

        if testing:
            knn_acc, rf_acc = 0, 0
            for i in range(testing_count):
                print("############# Test round", i+1)
                testset, tag = generate_test_data(clustering=clustering, df=test)
                print("############# Actual tag:", tag, end='  ')
                knn_result = knn_test(testset)
                print("Predicted tag for KNN:", mode(knn_result), end='  ')
                rf_result = rf_test(testset)
                print("Predicted tag for RF:", mode(rf_result))
                if mode(knn_result) == tag: knn_acc += 1
                if mode(rf_result) == tag: rf_acc += 1
            print("KNN Accuracy:", knn_acc/testing_count, end='  ')
            print("RF Accuracy:", rf_acc/testing_count)

        else:
            feature_importance = random_forest(train, save=True)
            print(feature_importance)
            svm_accuracy = svm(train, size=window_size, save=False)
            print("Max svm accuracy:", svm_accuracy)
            if clustering:
                knn_accuracy = knn(train, size=window_size, dim=1, save=True)
            else:
                knn_accuracy = knn(train, size=window_size, dim=6, save=True)
            print("Max knn accuracy:", knn_accuracy)
    return

if __name__ == "__main__":
    main()