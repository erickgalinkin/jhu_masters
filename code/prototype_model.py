import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# Configuration options
BATCH_SIZE = 50
EPOCHS = 15

# Data locations
REQUEST_REPLY_CSV_LOCATION = "./data/requestreply.csv"
REPLY_REPLY_CSV_LOCATION = "./data/replyreply.csv"

request_reply_df = pd.read_csv(REQUEST_REPLY_CSV_LOCATION, header=None)
reply_reply_df = pd.read_csv(REPLY_REPLY_CSV_LOCATION, header=None)


def df_to_dataset(X, y, shuffle=True, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices((dict(X), y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    ds = ds.batch(batch_size)
    return ds


def fix_data(dataset, shuffle=True, batch_size=32):
    dataset = dataset.copy()
    dataset.rename(columns={'0': 'Malicious', '1': "appname"}, inplace=True)
    dataset["Malicious"] = dataset["Malicious"].map({'legit': 0, 'malware': 1}).values
    labels = dataset.pop("Malicious")
    dataset.drop("appname", axis=1, inplace=True)
    X_trainval, X_test, y_trainval, y_test = train_test_split(dataset, labels, test_size=.15)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=.15)

    train = df_to_dataset(X_train, y_train, shuffle=shuffle, batch_size=batch_size)
    val = df_to_dataset(X_val, y_val, shuffle=shuffle, batch_size=batch_size)
    test = df_to_dataset(X_test, y_test, shuffle=shuffle, batch_size=batch_size)

    return train, val, test


def build_fc_model(features):
    feature_layer = layers.DenseFeatures(features)

    model = Sequential()
    model.add(feature_layer)
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_conv_model(features):
    feature_layer = layers.DenseFeatures(features)

    model = Sequential()
    model.add(feature_layer)
    model.add(layers.Reshape((100, 1)))
    model.add(layers.Conv1D(64, input_shape=(None, 100), kernel_size=3, strides=1, activation='relu'))
    model.add(layers.Conv1D(64, input_shape=(None, 100), kernel_size=3, strides=1, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model, data):
    train, val, test = fix_data(data, shuffle=True, batch_size=BATCH_SIZE)
    history = model.fit(train,
                        epochs=EPOCHS,
                        validation_data=val)

    print("History: {}".format(history.history))

    results = model.evaluate(test)
    print('test loss, test acc:', results)

    return history, results


def generate_features(dataset):
    dataset.columns = [str(col) for col in dataset.columns]
    features = [tf.feature_column.numeric_column(col) for col in dataset.columns if col not in ["0", "1"]]
    return features


if __name__ == "__main__":
    print("Training fully connected model on request reply data")
    features = generate_features(request_reply_df)
    model = build_fc_model(features)
    fc_qr_history, fc_qr_results = train_model(model, request_reply_df)
    print("Training fully connected model on reply reply data")
    features = generate_features(reply_reply_df)
    model = build_fc_model(features)
    fc_rr_history, fc_rr_results = train_model(model, reply_reply_df)
    print("Training 1D convolutional model on request reply data")
    features = generate_features(request_reply_df)
    model = build_conv_model(features)
    conv_qr_history, conv_qr_results = train_model(model, request_reply_df)
