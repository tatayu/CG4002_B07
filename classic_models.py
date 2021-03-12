import numpy as np
import pandas as pd
import joblib

from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.svm import TimeSeriesSVC

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, KFold

def svm(df, size, save):
    if isinstance(df, pd.DataFrame):
        df = df.to_numpy()
    X, y = df[:,:-1], df[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    X_train, X_test, y_train, y_test = X_train[:size,:], X_test[:size,:], y_train[:size], y_test[:size]

    X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = TimeSeriesScalerMinMax().fit_transform(X_test)
    X_test = X_test.reshape(X_test.shape[0], -1)

    clf = TimeSeriesSVC(kernel="gak", gamma=.3)
    kf = KFold()

    clf.fit(X_train, y_train)
    print('Support Vector Machine Validation Score:', 
        np.mean(np.array(cross_val_score(clf, X_train, y_train, cv=kf, scoring='f1_weighted')))
    )
    score = clf.score(X_test, y_test)

    # save the model to disk
    if save:
        filename = './svm_model.pkl'
        joblib.dump(clf, filename)

    return score

def knn(df, dim, save):
    if isinstance(df, pd.DataFrame):
        df = df.to_numpy()
    X, y = df[:,:-1], df[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
    # X_train = X_train.reshape(X_train.shape[0], -1)
    # X_test = TimeSeriesScalerMinMax().fit_transform(X_test)
    # X_test = X_test.reshape(X_test.shape[0], -1)
    
    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=dim, random_state=42))
    lda = make_pipeline(StandardScaler(),
                        LinearDiscriminantAnalysis(n_components=dim))
    nca = make_pipeline(StandardScaler(),
                        NeighborhoodComponentsAnalysis(n_components=dim,
                                                       random_state=42))

    knn = KNeighborsClassifier(n_neighbors=7)
    kf = KFold()

    dim_reduction_methods = [
        ('Principal Component Analysis',pca), 
        ('Latent Dirichlet Allocation', lda),
        ('Neighborhood Components Analysis', nca)
    ]
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    print("K-Nearest Neighbour Score:", score)
    print()

    if save:
        filename = './knn_model.pkl'
        joblib.dump(knn, filename)

    return score
    acc_knn = []
    for (name,model) in dim_reduction_methods:
        model.fit(X_train, y_train)
        if save:
            filename = './knn_model.pkl'
            tuple_obj = (knn, model)
            joblib.dump(tuple_obj, filename)
        knn.fit(model.transform(X_train), y_train)
        if name == 'Latent Dirichlet Allocation':
            print('K-Nearest Neighbour Validation Score:', 
                np.mean(np.array(cross_val_score(model, X_train, y_train, cv=kf, scoring='f1_weighted'
            ))))
        score = knn.score(model.transform(X_test), y_test)
        acc_knn.append(score)

    return np.max(np.array(acc_knn))

def random_forest(df, dim, save):
    if isinstance(df, pd.DataFrame):
        df_np = df.copy()
        df_np = df_np.to_numpy()
    X, y = df_np[:,:-1], df_np[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=dim, random_state=42))
    lda = make_pipeline(StandardScaler(),
                        LinearDiscriminantAnalysis(n_components=dim))
    nca = make_pipeline(StandardScaler(),
                        NeighborhoodComponentsAnalysis(n_components=dim,
                                                       random_state=42))

    dim_reduction_methods = [
        ('Principal Component Analysis',pca), 
        ('Latent Dirichlet Allocation', lda),
        ('Neighborhood Components Analysis', nca)
    ]

    # fit model
    rfc = RandomForestClassifier(n_estimators=1000)
    kf = KFold()
    print(X_train.shape, X_test.shape)
    print('Random Forest Validation Score:', np.mean(np.array(cross_val_score(rfc, X_train, y_train, cv=kf))))

    acc_knn = []

    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy:", accuracy)
    print()

    if save:
        filename = './rf_model.pkl'
        joblib.dump(rfc, filename)

    return pd.DataFrame(
            {
                'col_name': rfc.feature_importances_
            },  index=df.columns[:-1]
        ).sort_values(
            by='col_name', ascending=False
        )

    for (name,model) in dim_reduction_methods:
        model.fit(X_train, y_train)
        rfc.fit(model.transform(X_train), y_train)
        y_pred = rfc.predict(model.transform(X_test))
        accuracy = accuracy_score(y_test, y_pred)
        print("Random Forest Accuracy:", accuracy)
        print()
        acc_knn.append(accuracy)

    # save the model to disk
    if save:
        filename = './rf_model.pkl'
        tuple_obj = (rfc, model)
        joblib.dump(tuple_obj, filename)

    return None
    # pd.DataFrame(
    #         {
    #             'col_name': rfc.feature_importances_
    #         },  index=df.columns[:-1]
    #     ).sort_values(
    #         by='col_name', ascending=False
    #     )

def knn_test(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    knn = joblib.load(open('knn_model.pkl', 'rb'))
    result = knn.predict(data)
    return result

def svm_test(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    clf = joblib.load(open('svm_model.pkl', 'rb'))
    result = clf.predict(data)
    return result

def rf_test(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    rfc = joblib.load(open('rf_model.pkl', 'rb'))
    result = rfc.predict(data)
    return result
