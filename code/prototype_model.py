import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pywt
from custom_layers import FourierConvLayer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import logging
from scipy import stats

# Make warnings be quiet
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import warnings
from numpy import ComplexWarning

warnings.filterwarnings(action='ignore', category=ComplexWarning)

# Configuration options
BATCH_SIZE = 100
EPOCHS = 20

# Data locations
REQUEST_REPLY_CSV_LOCATION = "./data/requestreply.csv"
REPLY_REPLY_CSV_LOCATION = "./data/replyreply.csv"


def df_to_dataset(X, y, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(X), y)).repeat(5)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size)
    return ds


def fix_data(dataset, shuffle=True, batch_size=32):
    dataset = dataset.copy()
    labels = dataset.pop("Malicious")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=.2, stratify=labels)

    train = df_to_dataset(X_train, y_train, shuffle=shuffle, batch_size=batch_size)
    test = df_to_dataset(X_test, y_test, shuffle=shuffle, batch_size=batch_size)

    return train, test


def build_fc_model(features):
    feature_layer = layers.DenseFeatures(features)

    model = Sequential()
    model.add(feature_layer)
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='SGD',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_summary_model(features):
    feature_layer = layers.DenseFeatures(features)

    model = Sequential()
    model.add(feature_layer)
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='SGD',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_conv_model(features):
    feature_layer = layers.DenseFeatures(features)

    model = Sequential()
    model.add(feature_layer)
    model.add(layers.Reshape((100, 1)))
    model.add(layers.Conv1D(256, input_shape=(None, 100), kernel_size=5, strides=1, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(layers.Conv1D(256, input_shape=(None, 100), kernel_size=3, strides=2, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2, strides=None, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='SGD',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def fourier_model(features):
    feature_layer = layers.DenseFeatures(features)

    model = Sequential()
    model.add(feature_layer)
    model.add(layers.Reshape((100, 1)))
    model.add(FourierConvLayer(256, autocast=False))
    model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(FourierConvLayer(256, autocast=False))
    model.add(layers.MaxPooling1D(pool_size=2, strides=None, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='SGD',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, data, train_epoch=EPOCHS):
    callback = EarlyStopping(monitor='loss', patience=2)
    train, test = fix_data(data, shuffle=True, batch_size=BATCH_SIZE)
    history = model.fit(train, epochs=train_epoch, steps_per_epoch=50, verbose=0, callbacks=[callback])

    results = model.evaluate(test, steps=50, verbose=0)

    return history, results


def generate_features(dataset):
    dataset.columns = [str(col) for col in dataset.columns]
    features = [tf.feature_column.numeric_column(col) for col in dataset.columns if col not in ["Malicious"]]
    return features


def create_fourier_dataset(dataset):
    columns = dataset.columns.tolist()
    columns.remove("Malicious")
    fourier_data = dataset[columns].apply(np.fft.fft)
    fourier_data["Malicious"] = dataset["Malicious"]
    return fourier_data


def create_continuous_wavelet_dataset(dataset):
    scales = [1]
    wavelet = "morl"
    columns = dataset.columns.tolist()
    columns.remove("Malicious")
    wavelet_data = dataset[columns].apply(pywt.cwt, args=(scales, wavelet), axis=1).to_frame()
    wavelet_array = []
    for i in range(wavelet_data.shape[0]):
        wavelet_array.append(wavelet_data.loc[i][0][0][0])
    wavelet_data = pd.DataFrame(data=wavelet_array)
    wavelet_data["Malicious"] = dataset["Malicious"]
    return wavelet_data


def create_summary_dataset(dataset):
    dataset = dataset.copy()
    mal = dataset.pop("Malicious")
    amean = dataset.copy().apply(np.mean, axis=1)
    sd = dataset.copy().apply(np.std, axis=1)
    var = dataset.copy().apply(np.var, axis=1)
    maximum = dataset.copy().apply(np.max, axis=1)
    minimum = dataset.copy().apply(np.min, axis=1)
    geo_mean = dataset.copy().apply(np.abs).apply(stats.gmean, axis=1)
    har_mean = dataset.copy().apply(np.abs).apply(stats.hmean, axis=1)
    summary_data = pd.DataFrame()
    summary_data["Malicious"] = mal.values
    summary_data["Arithmetic_Mean"] = amean.values
    summary_data["Standard_Deviation"] = sd.values
    summary_data["Variance"] = var.values
    summary_data["Max"] = maximum.values
    summary_data["Min"] = minimum.values
    summary_data["Geometric_Mean"] = geo_mean.values
    summary_data["Harmonic_Mean"] = har_mean.values
    return summary_data

if __name__ == "__main__":
    request_reply_df = pd.read_csv(REQUEST_REPLY_CSV_LOCATION, header=None)
    request_reply_df.columns = [str(col) for col in request_reply_df.columns]
    request_reply_df.drop("1", axis=1, inplace=True)
    request_reply_df.rename(columns={"0": "Malicious"}, inplace=True)
    request_reply_df.replace({"Malicious": {'legit': 0, 'malware': 1}}, inplace=True)

    request_reply_fourier = create_fourier_dataset(request_reply_df)
    request_reply_wavelet = create_continuous_wavelet_dataset(request_reply_df)
    request_reply_summary = create_summary_dataset(request_reply_df)

    print("Experiment 1 running 25 times.")
    # Raw data
    features = generate_features(request_reply_df)
    # Fully Connected Neural Network
    for _ in range(25):
        score = list()
        model = build_fc_model(features)
        fc_history, fc_results = train_model(model, request_reply_df)
        score.append(fc_results[1])
    print("Average accuracy of fully-connected model on raw data: {}".format(np.average(score)))
    # Convolutional Neural Network
    for _ in range(25):
        score = list()
        model = build_conv_model(features)
        conv_history, conv_results = train_model(model, request_reply_df)
        score.append(conv_results[1])
    print("Average accuracy of convolutional model on raw data: {}".format(np.average(score)))
    dataset = request_reply_df.copy()
    labels = dataset.pop("Malicious")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=.2, stratify=labels)
    for _ in range(25):
        score = list()
        clf = SVC()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        svm_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(svm_accuracy)
    print("Average accuracy of SVC on raw data: {}".format(np.average(score)))
    for _ in range(25):
        score = list()
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        rf_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(rf_accuracy)
    print("Average accuracy of Random Forest on raw data: {}".format(np.average(score)))

    # Fourier data
    features = generate_features(request_reply_fourier)
    # Fully Connected Neural Network
    for _ in range(25):
        score = list()
        model = build_fc_model(features)
        fc_history, fc_results = train_model(model, request_reply_fourier)
        score.append(1 - fc_results[1])
    print("Average accuracy of fully-connected model on Fourier data: {}".format(np.average(score)))
    # Convolutional Neural Network
    for _ in range(25):
        score = list()
        model = build_conv_model(features)
        conv_history, conv_results = train_model(model, request_reply_fourier)
        score.append(1 - conv_results[1])
    print("Average accuracy of convolutional model on Fourier data: {}".format(np.average(score)))
    dataset = request_reply_fourier.copy()
    labels = dataset.pop("Malicious")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=.2, stratify=labels)
    for _ in range(25):
        score = list()
        clf = SVC()
        clf.fit(X_train.astype('float32'), y_train)
        predictions = clf.predict(X_test.astype('float32'))
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        svm_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(1 - svm_accuracy)
    print("Average accuracy of SVC on Fourier data: {}".format(np.average(score)))
    for _ in range(25):
        score = list()
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train.astype('float32'), y_train)
        predictions = clf.predict(X_test.astype('float32'))
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        rf_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(1 - rf_accuracy)
    print("Average accuracy of Random Forest on Fourier data: {}".format(np.average(score)))

    # Wavelet data
    features = generate_features(request_reply_wavelet)
    # Fully Connected Neural Network
    for _ in range(25):
        score = list()
        model = build_fc_model(features)
        fc_history, fc_results = train_model(model, request_reply_wavelet)
        score.append(fc_results[1])
    print("Average accuracy of fully-connected model on Wavelet data: {}".format(np.average(score)))
    # Convolutional Neural Network
    for _ in range(25):
        score = list()
        model = build_conv_model(features)
        conv_history, conv_results = train_model(model, request_reply_wavelet)
        score.append(conv_results[1])
    print("Average accuracy of convolutional model on Wavelet data: {}".format(np.average(score)))
    dataset = request_reply_wavelet.copy()
    labels = dataset.pop("Malicious")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=.2, stratify=labels)
    for _ in range(25):
        score = list()
        clf = SVC()
        clf.fit(X_train.astype('float32'), y_train)
        predictions = clf.predict(X_test.astype('float32'))
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        svm_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(svm_accuracy)
    print("Average accuracy of SVC on Wavelet data: {}".format(np.average(score)))
    for _ in range(25):
        score = list()
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train.astype('float32'), y_train)
        predictions = clf.predict(X_test.astype('float32'))
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        rf_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(rf_accuracy)
    print("Average accuracy of Random Forest on Wavelet data: {}".format(np.average(score)))

    # Summary data
    features = generate_features(request_reply_summary)
    # Fully Connected Neural Network
    for _ in range(25):
        score = list()
        model = build_summary_model(features)
        summary_history, summary_results = train_model(model, request_reply_summary)
        score.append(summary_results[1])
    print("Average accuracy of fully-connected model on Summary data: {}".format(np.average(score)))
    dataset = request_reply_summary.copy()
    labels = dataset.pop("Malicious")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=.2, stratify=labels)
    for _ in range(25):
        score = list()
        clf = SVC()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        svm_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(svm_accuracy)
    print("Average accuracy of SVC on Summary data: {}".format(np.average(score)))
    for _ in range(25):
        score = list()
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        rf_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(rf_accuracy)
    print("Average accuracy of Random Forest on Summary data: {}".format(np.average(score)))
