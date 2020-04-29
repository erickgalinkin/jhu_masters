import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pywt
from custom_layers import FourierConvLayer

tf.keras.backend.set_floatx('float32')

# Configuration options
BATCH_SIZE = 100
EPOCHS = 25

# Data locations
REQUEST_REPLY_CSV_LOCATION = "./data/requestreply.csv"
REPLY_REPLY_CSV_LOCATION = "./data/replyreply.csv"


def df_to_dataset(X, y, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
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
    train, test = fix_data(data, shuffle=True, batch_size=BATCH_SIZE)
    history = model.fit(train, epochs=train_epoch)

    print("History: {}".format(history.history))

    results = model.evaluate(test)
    print('test loss, test acc:', results)

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


if __name__ == "__main__":
    request_reply_df = pd.read_csv(REQUEST_REPLY_CSV_LOCATION, header=None)
    request_reply_df.columns = [str(col) for col in request_reply_df.columns]
    request_reply_df.drop("1", axis=1, inplace=True)
    request_reply_df.rename(columns={"0": "Malicious"}, inplace=True)
    request_reply_df.replace({"Malicious": {'legit': 0, 'malware': 1}}, inplace=True)

    reply_reply_df = pd.read_csv(REPLY_REPLY_CSV_LOCATION, header=None)
    reply_reply_df.columns = [str(col) for col in reply_reply_df.columns]
    reply_reply_df.drop("1", axis=1, inplace=True)
    reply_reply_df.rename(columns={"0": "Malicious"}, inplace=True)
    reply_reply_df.replace({"Malicious": {'legit': 0, 'malware': 1}}, inplace=True)

    request_reply_fourier = create_fourier_dataset(request_reply_df)
    reply_reply_fourier = create_fourier_dataset(reply_reply_df)
    request_reply_wavelet = create_continuous_wavelet_dataset(request_reply_df)
    reply_reply_wavelet = create_continuous_wavelet_dataset(reply_reply_df)


