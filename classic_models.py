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


def svm(df, size, save):
    if isinstance(df, pd.DataFrame):
        df = df.to_numpy()
    X, y = df[:,:-1], df[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    X_train, X_test, y_train, y_test = X_train[:size,:], X_test[:size,:], y_train[:size], y_test[:size]

    X_train = TimeSeriesScalerMinMax().fit_transform(X_train)
    X_test = TimeSeriesScalerMinMax().fit_transform(X_test)

    clf = TimeSeriesSVC(kernel="gak", gamma=.3)
    print("<SVM> Fitting Data")
    clf.fit(X_train, y_train)
    print("<SVM> Scoring Data")
    score = clf.score(X_test, y_test)
    print("<SVM> Correct classification rate:", score)

    # save the model to disk
    if save:
        filename = './svm_model.pkl'
        joblib.dump(clf, filename)

    return score

def knn(df,size,dim,save):
    if isinstance(df, pd.DataFrame):
        df = df.to_numpy()
    X, y = df[:,:-1], df[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = X_train[:size,:], X_test[:size,:], y_train[:size], y_test[:size]

    pca = make_pipeline(StandardScaler(),
                        PCA(n_components=dim, random_state=42))
    lda = make_pipeline(StandardScaler(),
                        LinearDiscriminantAnalysis(n_components=dim))
    nca = make_pipeline(StandardScaler(),
                        NeighborhoodComponentsAnalysis(n_components=dim,
                                                       random_state=42))

    knn = KNeighborsClassifier()

    dim_reduction_methods = [
        ('Principal Component Analysis',pca), 
        ('Latent Dirichlet Allocation', lda),
        ('Neighborhood Components Analysis', nca)
    ]
    acc_knn = []

    for (name,model) in dim_reduction_methods:
        print("<KNN> Fitting Data for", name)
        model.fit(X_train, y_train)
        knn.fit(model.transform(X_train), y_train)
        print("<KNN> Scoring Data for", name)
        score = knn.score(model.transform(X_test), y_test)
        print("<KNN> Correct classification rate for", name, ":", score)
        acc_knn.append(score)

    print(np.argmax(acc_knn))
    # save the model to disk
    if save:
        filename = './knn_model.pkl'
        tuple_obj = (knn, model)
        joblib.dump(tuple_obj, filename)

    return np.max(np.array(acc_knn))

def random_forest(df, save):
    if isinstance(df, pd.DataFrame):
        df_np = df.copy()
        df_np = df_np.to_numpy()
    X, y = df_np[:,:-1], df_np[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # fit model
    model = RandomForestClassifier(n_estimators=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy:", accuracy)

    # save the model to disk
    if save:
        filename = './rf_model.pkl'
        joblib.dump(model, filename)

    return pd.DataFrame(
            {
                'col_name': model.feature_importances_
            },  index=df.columns[:-1]
        ).sort_values(
            by='col_name', ascending=False
        )

def knn_test(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    knn, model = joblib.load(open('knn_model.pkl', 'rb'))
    result = knn.predict(model.transform(data))
    return result

def rf_test(data):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    model = joblib.load(open('rf_model.pkl', 'rb'))
    result = model.predict(data)
    return result