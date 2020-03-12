from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

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
    def __init__(self, d_in, bias=True):
        super(WaveletLayer, self).__init__()
        self.d_in = d_in
        self.xfm = DWTForward(J=1, mode='symmetric', wave='db2')
        self.ifm = DWTInverse(mode='symmetric', wave='db2')
        # if cuda:
        #     self.xfm.cuda()
        #     self.ifm.cuda()
        self.weight = torch.nn.Parameter(torch.randn((2, (int(d_in/2) + 1))))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn((int(d_in/2) + 1)))

    def forward(self, x):
        dim_1, dim_2, dim_3, y = x.shape
        if y != self.d_in:
            sys.exit('Incorrect Input Features. Please use a tensor with {} Input Features'.format(self.d_in))
        Yl, Yh = self.xfm(x)
        output = torch.mm(self.weight.T, Yl[0][0]) + self.bias
        output = output[:2]
        Yl_o = output.view(1, 1, 2, (1 + int(self.d_in/2)))
        z = self.ifm((Yl_o, Yh))
        a = z.view(dim_1, dim_2, 2*dim_3, y)
        return F.leaky_relu(a)


class WaveletNN(nn.Module):
    def __init__(self, d_in):
        super(WaveletNN, self).__init__()

        self.model = nn.Sequential(
            WaveletLayer(d_in),
            WaveletLayer(d_in),
            nn.MaxPool1d(2, 2),
            nn.BatchNorm1d(1),
            nn.Flatten(),
            nn.Linear(400, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        y = self.model(x)
        return y


if __name__ == "__main__":
    x = torch.Tensor(1, 1, 1, 100)
    wl = WaveletLayer(100)
    y = wl(x)
    print(y.shape)