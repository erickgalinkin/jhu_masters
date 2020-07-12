import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from loggingreporter import LoggingReporter
import pandas as pd
import numpy as np
import pywt
from custom_layers import FourierConvLayer, WaveletNN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import logging
from scipy import stats
import torch
from torch import nn, optim
import torch.utils.data as torchdata
from collections import namedtuple
import pickle
from collections import defaultdict, OrderedDict
import kde
import simplebinmi
import matplotlib.pyplot as plt
import seaborn as sns

tf.compat.v1.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Make warnings be quiet
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
import warnings
from numpy import ComplexWarning

warnings.filterwarnings(action='ignore', category=ComplexWarning)

# Configuration options
cfg = dict()
cfg["EPOCHS"] = 1000
cfg["BASE_DIR"] = "./rawdata/"
cfg["SGD_LEARNINGRATE"] = 0.001
cfg["SGD_BATCHSIZE"] = 128
cfg['ACTIVATION'] = "ReLU"

# Data locations
REQUEST_REPLY_CSV_LOCATION = "./data/requestreply.csv"
REPLY_REPLY_CSV_LOCATION = "./data/replyreply.csv"


def fix_data(dataset):
    nb_classes = 2
    dataset = dataset.copy()
    labels = dataset.pop("Malicious")
    X_train, X_test, y_train, y_test = train_test_split(dataset.values, labels.values, test_size=.2, stratify=labels)

    X_train = np.reshape(X_train, [X_train.shape[0], -1]).astype('float32')
    X_test = np.reshape(X_test, [X_test.shape[0], -1]).astype('float32')

    Y_train = y_train
    Y_test = y_test

    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    train = Dataset(X_train, Y_train, y_train, nb_classes)
    test = Dataset(X_test , Y_test, y_test, nb_classes)

    return train, test


def build_fc_model(input_shape=(100, )):
    model = Sequential()
    model.add(layers.Input(input_shape))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.SGD(learning_rate=cfg["SGD_LEARNINGRATE"])

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_conv_model(input_shape=(100, )):

    model = Sequential()
    model.add(layers.Input(input_shape))
    model.add(layers.Reshape((100, 1)))
    model.add(layers.Conv1D(256, input_shape=(None, 100), kernel_size=5, strides=1, activation='relu'))
    # model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(layers.Conv1D(256, input_shape=(None, 100), kernel_size=3, strides=2, activation='relu'))
    # model.add(layers.MaxPooling1D(pool_size=2, strides=None, padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.SGD(learning_rate=cfg["SGD_LEARNINGRATE"])

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def fourier_model(input_shape=(100, )):

    model = Sequential()
    model.add(layers.Input(input_shape))
    # model.add(layers.Reshape((100, 1)))
    model.add(FourierConvLayer(256, autocast=False))
    # model.add(layers.MaxPooling1D(pool_size=2, strides=2, padding='valid'))
    model.add(FourierConvLayer(256, autocast=False))
    # model.add(layers.MaxPooling1D(pool_size=2, strides=None, padding='valid'))
    # model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.SGD(learning_rate=cfg["SGD_LEARNINGRATE"])

    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_results(tst, cur_dir, model_name):
    infoplane_measure = 'bin'

    DO_SAVE = True  # Whether to save plots or just show them
    DO_LOWER = (infoplane_measure == 'lower')  # Whether to compute lower bounds also
    DO_BINNED = (infoplane_measure == 'bin')  # Whether to compute MI estimates based on binning

    MAX_EPOCHS = 1000  # Max number of epoch for which to compute mutual information measure
    COLORBAR_MAX_EPOCHS = 1000

    noise_variance = 1e-1  # Added Gaussian noise variance
    Klayer_activity = K.placeholder(ndim=2)  # Keras placeholder
    entropy_func_upper = K.function([Klayer_activity, ], [kde.entropy_estimator_kl(Klayer_activity, noise_variance),])
    entropy_func_lower = K.function([Klayer_activity, ], [kde.entropy_estimator_bd(Klayer_activity, noise_variance),])

    # nats to bits conversion factor
    nats2bits = 1.0 / np.log(2)

    # Save indexes of tests data for each of the output classes
    saved_labelixs = {}
    for i in range(150):
        saved_labelixs[i] = tst.y == i

    labelprobs = np.mean(tst.Y, axis=0)

    PLOT_LAYERS = None  # Which layers to plot.  If None, all saved layers are plotted

    # Data structure used to store results
    measures = OrderedDict()
    measures['relu'] = {}

    for epochfile in sorted(os.listdir(cur_dir)):
        if not epochfile.startswith('epoch'):
            break

        fname = cur_dir + "/" + epochfile
        with open(fname, 'rb') as f:
            d = pickle.load(f)

        epoch = d['epoch']

        num_layers = len(d['data']['activity_tst'])
        if PLOT_LAYERS is None:
            PLOT_LAYERS = []
            for lndx in range(num_layers):
                PLOT_LAYERS.append(lndx)

        cepochdata = defaultdict(list)
        for lndx in range(num_layers):
            activity = d['data']['activity_tst'][lndx]
            if len(activity.shape) == 3:
                activity = activity[:,1]

            # Compute marginal entropies
            h_upper = entropy_func_upper([activity, ])[0]
            if DO_LOWER:
                h_lower = entropy_func_lower([activity, ])[0]

            # Layer activity given input. This is simply the entropy of the Gaussian noise
            hM_given_X = kde.kde_condentropy(activity, noise_variance)

            # Compute conditional entropies of layer activity given output
            hM_given_Y_upper = 0.
            hcond_upper = entropy_func_upper([activity[saved_labelixs[i], :], ])[0]
            hM_given_Y_upper += labelprobs * hcond_upper

            if DO_LOWER:
                hM_given_Y_lower = 0.
                hcond_lower = entropy_func_lower([activity[saved_labelixs[i], :], ])[0]
                hM_given_Y_lower += labelprobs * hcond_lower

            cepochdata['MI_XM_upper'].append(nats2bits * (h_upper - hM_given_X))
            cepochdata['MI_YM_upper'].append(nats2bits * (h_upper - hM_given_Y_upper))
            cepochdata['H_M_upper'].append(nats2bits * h_upper)

            pstr = 'upper: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
            cepochdata['MI_XM_upper'][-1], cepochdata['MI_YM_upper'][-1])
            if DO_LOWER:  # Compute lower bounds
                cepochdata['MI_XM_lower'].append(nats2bits * (h_lower - hM_given_X))
                cepochdata['MI_YM_lower'].append(nats2bits * (h_lower - hM_given_Y_lower))
                cepochdata['H_M_lower'].append(nats2bits * h_lower)
                pstr += ' | lower: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                cepochdata['MI_XM_lower'][-1], cepochdata['MI_YM_lower'][-1])

            if DO_BINNED:  # Compute binner estimates
                binxm, binym = simplebinmi.bin_calc_information2(saved_labelixs, activity, 0.5)
                cepochdata['MI_XM_bin'].append(nats2bits * binxm)
                cepochdata['MI_YM_bin'].append(nats2bits * binym)
                pstr += ' | bin: MI(X;M)=%0.3f, MI(Y;M)=%0.3f' % (
                cepochdata['MI_XM_bin'][-1], cepochdata['MI_YM_bin'][-1])

            measures['relu'][epoch] = cepochdata

    sns.set_style('darkgrid')

    max_epoch = max((max(vals.keys()) if len(vals) else 0) for vals in measures.values())
    sm = plt.cm.ScalarMappable(cmap='gnuplot', norm=plt.Normalize(vmin=0, vmax=COLORBAR_MAX_EPOCHS))
    sm._A = []

    fig = plt.figure(figsize=(10, 5))
    for actndx, (activation, vals) in enumerate(measures.items()):
        epochs = sorted(vals.keys())
        if not len(epochs):
            continue
        plt.subplot(1, 2, actndx + 1)
        for epoch in epochs:
            c = sm.to_rgba(epoch)
            xmvals = np.array(vals[epoch]['MI_XM_' + infoplane_measure])[PLOT_LAYERS]
            ymvals = np.array(vals[epoch]['MI_YM_' + infoplane_measure])[PLOT_LAYERS]

            plt.plot(xmvals, ymvals, c=c, alpha=0.1, zorder=1)
            plt.scatter(xmvals, ymvals, s=20, facecolors=[c for _ in PLOT_LAYERS], edgecolor='none', zorder=2)

        plt.ylim([0, 3.5])
        plt.xlim([0, 14])
        plt.xlabel('I(X;M)')
        plt.ylabel('I(Y;M)')
        plt.title(activation)

    cbaxes = fig.add_axes([1.0, 0.125, 0.03, 0.8])
    plt.colorbar(sm, label='Epoch', cax=cbaxes)
    plt.tight_layout()

    if DO_SAVE:
        plt.savefig('plots/' + model_name + "_infoplane.png", bbox_inches='tight')


def train_model(model, data, model_name, iteration, train_epoch=cfg["EPOCHS"]):
    # stopping = EarlyStopping(monitor='loss', patience=2)
    train, test = fix_data(data)
    cfg["SAVE_DIR"] = cfg["BASE_DIR"] + model_name + "/iteration_" + iteration
    reporter = LoggingReporter(cfg=cfg, trn=train, tst=test, do_save_func=do_report)
    history = model.fit(x=train.X, y=train.Y, epochs=train_epoch, batch_size=cfg['SGD_BATCHSIZE'],
                        verbose=0, callbacks=[reporter])

    results = model.evaluate(x=test.X, y=test.Y, verbose=0)

    # plot_results(test, cfg["SAVE_DIR"], model_name)

    return history, results


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


class WaveletDataset(torchdata.Dataset):
    def __init__(self, df):
        self.labels = df["Malicious"]
        self.data = df.copy().drop("Malicious", axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row


def do_report(epoch):
    # Only log activity for some epochs.  Mainly this is to make things run faster.
    if epoch < 200:       # Log for all first 20 epochs
        return True
    elif epoch < 500:    # Then every 10th
        return (epoch % 10 == 0)
    else:                # Then every 100th
        return (epoch % 100 == 0)


if __name__ == "__main__":
    request_reply_df = pd.read_csv(REQUEST_REPLY_CSV_LOCATION, header=None)
    request_reply_df.columns = [str(col) for col in request_reply_df.columns]
    request_reply_df.drop("1", axis=1, inplace=True)
    request_reply_df.rename(columns={"0": "Malicious"}, inplace=True)
    request_reply_df.replace({"Malicious": {'legit': 0, 'malware': 1}}, inplace=True)

    request_reply_fourier = create_fourier_dataset(request_reply_df)
    request_reply_wavelet = create_continuous_wavelet_dataset(request_reply_df)
    request_reply_summary = create_summary_dataset(request_reply_df)

    num_experiments = 100

    print("Experiment 1 running {} times.".format(num_experiments))
    # Raw data
    # Fully Connected Neural Network
    score = list()
    for i in range(num_experiments):
        model = build_fc_model()
        fc_history, fc_results = train_model(model, request_reply_df, "Fully-Connected", str(i))
        score.append(fc_results[1])
    print("Average accuracy of fully-connected model on raw data: {}".format(np.average(score)))
    # Convolutional Neural Network
    score = list()
    for i in range(num_experiments):
        model = build_conv_model()
        conv_history, conv_results = train_model(model, request_reply_df, "Convolutional", str(i))
        score.append(conv_results[1])
    print("Average accuracy of convolutional model on raw data: {}".format(np.average(score)))
    dataset = request_reply_df.copy()
    labels = dataset.pop("Malicious")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=.2, stratify=labels)
    score = list()
    for _ in range(num_experiments):
        clf = SVC()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        svm_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(svm_accuracy)
    print("Average accuracy of SVC on raw data: {}".format(np.average(score)))
    score = list()
    for _ in range(num_experiments):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        rf_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(rf_accuracy)
    print("Average accuracy of Random Forest on raw data: {}".format(np.average(score)))

    # Fourier data
    # Fully Connected Neural Network
    score = list()
    for i in range(num_experiments):
        model = build_fc_model()
        fc_history, fc_results = train_model(model, request_reply_fourier, "Fully-Connected_Fourier", str(i))
        score.append(1 - fc_results[1])
    print("Average accuracy of fully-connected model on Fourier data: {}".format(np.average(score)))
    # Convolutional Neural Network
    score = list()
    for i in range(num_experiments):
        model = build_conv_model()
        conv_history, conv_results = train_model(model, request_reply_fourier, "Convolutional_Fourier", str(i))
        score.append(1 - conv_results[1])
    print("Average accuracy of convolutional model on Fourier data: {}".format(np.average(score)))
    dataset = request_reply_fourier.copy()
    labels = dataset.pop("Malicious")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=.2, stratify=labels)
    score = list()
    for _ in range(num_experiments):
        clf = SVC()
        clf.fit(X_train.astype('float32'), y_train)
        predictions = clf.predict(X_test.astype('float32'))
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        svm_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(1 - svm_accuracy)
    print("Average accuracy of SVC on Fourier data: {}".format(np.average(score)))
    score = list()
    for _ in range(num_experiments):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train.astype('float32'), y_train)
        predictions = clf.predict(X_test.astype('float32'))
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        rf_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(1 - rf_accuracy)
    print("Average accuracy of Random Forest on Fourier data: {}".format(np.average(score)))

    # Wavelet data
    # Fully Connected Neural Network
    score = list()
    for i in range(num_experiments):
        model = build_fc_model()
        fc_history, fc_results = train_model(model, request_reply_wavelet, "Fully-Connected_Wavelet", str(i))
        score.append(fc_results[1])
    print("Average accuracy of fully-connected model on Wavelet data: {}".format(np.average(score)))
    # Convolutional Neural Network
    score = list()
    for i in range(num_experiments):
        model = build_conv_model()
        conv_history, conv_results = train_model(model, request_reply_wavelet, "Convolutional_Wavelet", str(i))
        score.append(conv_results[1])
    print("Average accuracy of convolutional model on Wavelet data: {}".format(np.average(score)))
    dataset = request_reply_wavelet.copy()
    labels = dataset.pop("Malicious")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=.2, stratify=labels)
    score = list()
    for _ in range(num_experiments):
        clf = SVC()
        clf.fit(X_train.astype('float32'), y_train)
        predictions = clf.predict(X_test.astype('float32'))
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        svm_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(svm_accuracy)
    print("Average accuracy of SVC on Wavelet data: {}".format(np.average(score)))
    score = list()
    for _ in range(num_experiments):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train.astype('float32'), y_train)
        predictions = clf.predict(X_test.astype('float32'))
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        rf_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(rf_accuracy)
    print("Average accuracy of Random Forest on Wavelet data: {}".format(np.average(score)))

    # Summary data
    # Fully Connected Neural Network
    score = list()
    for i in range(num_experiments):
        model = build_fc_model(input_shape=(7, ))
        summary_history, summary_results = train_model(model, request_reply_summary, "Fully-Connected_Summary", str(i))
        score.append(summary_results[1])
    print("Average accuracy of fully-connected model on Summary data: {}".format(np.average(score)))
    dataset = request_reply_summary.copy()
    labels = dataset.pop("Malicious")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=.2, stratify=labels)
    score = list()
    for _ in range(num_experiments):
        clf = SVC()
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        svm_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(svm_accuracy)
    print("Average accuracy of SVC on Summary data: {}".format(np.average(score)))
    score = list()
    for _ in range(num_experiments):
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        errors = [0 if p == label else 1 for p, label in zip(predictions, y_test)]
        rf_accuracy = 1 - (np.sum(errors) / len(errors))
        score.append(rf_accuracy)
    print("Average accuracy of Random Forest on Summary data: {}".format(np.average(score)))

    # Experiment 2
    print("Experiment 2 running 100 times")

    # Wavelet Model
    score = list()
    dataset = request_reply_df.copy()
    labels = dataset.pop("Malicious")
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=.2, stratify=labels)
    for _ in range(num_experiments):
        model = WaveletNN(100)
        opt = optim.SGD(model.parameters(), lr=0.005)
        bce = nn.BCELoss()
        for epoch in range(20):
            for i in range(len(X_train)):
                output = model(X_train.iloc[i])
                y = torch.tensor([y_train.iloc[i]]).float()
                loss = bce(output, y.unsqueeze(0))
                opt.zero_grad()
                loss.backward()
                opt.step()
        preds = list()
        for i in range(len(X_test)):
            prediction = model(X_test.iloc[i])
            preds.append(prediction)
        errors = [0 if p == label else 1 for p, label in zip(preds, y_test)]
        accuracy = np.sum(errors) / len(errors)
        score.append(accuracy)
    print("Average accuracy of Wavelet model on raw data: {}".format(np.average(score)))

    # # Fourier Model
    score = list()
    for i in range(num_experiments):
        model = fourier_model()
        fourier_model_history, fourier_model_results = train_model(model, request_reply_df, "Fourier_NN", str(i))
        score.append(fourier_model_results[1])
    print("Average accuracy of Fourier model on raw data: {}".format(np.average(score)))

