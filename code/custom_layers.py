from tensorflow.keras.layers import Layer
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pywt

tf.keras.backend.set_floatx('float32')
cuda = torch.cuda.is_available()


class FourierConvLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        super(FourierConvLayer, self).__init__()

        self.output_dim = output_dim
        self.fft = tf.signal.fft
        self.ifft = tf.signal.ifft
        self.kernel = None  # We initialize the kernel below - there is no way to initialize without an input shape.

        self.relu = tf.nn.relu

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(FourierConvLayer, self).build(input_shape)

    def call(self, x):
        x = tf.cast(x, dtype=tf.dtypes.complex64)
        y = tf.cast(self.kernel, dtype=tf.dtypes.complex64)
        fourier_x = self.fft(x)
        fourier_x = tf.matmul(fourier_x, y)
        x = self.ifft(fourier_x)
        x = tf.convert_to_tensor(x)
        x = tf.cast(x, dtype=tf.float32)
        return self.relu(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class WaveletLayer(nn.Module):
    def __init__(self, d_in, bias=False):
        super(WaveletLayer, self).__init__()
        self.d_in = d_in
        self.weight = torch.nn.Parameter(torch.randn((2, (int(d_in/2) + 1))))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn((int(d_in/2) + 1)))

    def forward(self, x):
        if isinstance(x, pd.Series):
            x = x.to_numpy()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        shape = x.size()
        cA, cD = pywt.dwt(x.data.numpy(), 'db2')
        output = np.multiply(self.weight.data.numpy()[0], cD)
        try:
            z = torch.from_numpy(pywt.idwt(cA, output, 'db2')).float()
        except RuntimeError as e:  # We get runtime errors when output is a zero vector.
            print("Blah")
            z = x.float()
        a = z.view(shape)
        out = F.leaky_relu(a)
        return out


class WaveletNN(nn.Module):
    def __init__(self, d_in):
        super(WaveletNN, self).__init__()

        self.model = nn.Sequential(
            WaveletLayer(d_in),
            WaveletLayer(d_in),
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.model(x)
        return y


if __name__ == "__main__":
    x = torch.Tensor(1, 1, 1, 100)
    wl = WaveletLayer(100)
    y = wl(x)
    print(y.shape)
    wavelet = WaveletNN(100)
    output = wavelet(x)
    print(output)