from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import pywt
import torch
import pytorch_wavelets as ptw

tf.keras.backend.set_floatx('float32')


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


class WaveletLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        super(WaveletLayer, self).__init__()

        self.output_dim = output_dim
        self.dwt = pywt.dwt
        self.idwt = pywt.idwt
        self.kernel = None  # We initialize the kernel below - there is no way to initialize without an input shape.

        self.conv1d = tf.nn.conv1d
        self.relu = tf.nn.relu

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(WaveletLayer, self).build(input_shape)

    def call(self, x):
        a = np.array(x)
        cA, cD = self.dwt(a, 'db2')
        cA = tf.multiply(cA, self.kernel)
        cD = tf.multiply(cD, self.kernel)
        x = self.idwt(cA, cD, 'db2')
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return self.relu(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


if __name__ == "__main__":
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input

    print("Running test...")
    inp = Input(shape=(100))
    out = WaveletLayer(64)(inp)
    model = Model(inp, out)

    model.predict(tf.ones((1, 100)))
