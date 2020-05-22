import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

import pickle
import os


class LoggingReporter(tf.keras.callbacks.Callback):
    def __init__(self, cfg, trn, tst, do_save_func=None, *kargs, **kwargs):
        super(LoggingReporter, self).__init__(*kargs, **kwargs)
        self.cfg = cfg  # Configuration options dictionary
        self.trn = trn  # Train data
        self.tst = tst  # Test data

        # do_save_func(epoch) should return True if we should save on that epoch
        self.do_save_func = do_save_func

    def on_train_begin(self, logs={}):
        if not os.path.exists(self.cfg['SAVE_DIR']):
            os.makedirs(self.cfg['SAVE_DIR'])

        # Indexes of the layers which we keep track of. Basically, this will be any layer
        # which has a 'kernel' attribute, which is essentially the "Dense" or "Dense"-like layers
        self.layerixs = []

        # Functions return activity of each layer
        self.layerfuncs = []

        # Functions return weights of each layer
        self.layerweights = []
        for lndx, l in enumerate(self.model.layers):
            if hasattr(l, 'kernel'):
                self.layerixs.append(lndx)
                self.layerfuncs.append(K.function(self.model.inputs, [l.output, ]))
                self.layerweights.append(l.kernel)

        grads = K.gradients(self.model.total_loss, self.model.trainable_weights)
        input_tensors = self.model._feed_inputs + self.model._feed_targets + self.model._feed_sample_weights
        self.get_gradients = K.function(input_tensors, grads)

        # Get cross-entropy loss
        self.get_loss = K.function(inputs=input_tensors, outputs=[self.model.total_loss, ])

    def on_epoch_begin(self, epoch, logs={}):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            self._log_gradients = False
        else:
            # We will log this epoch.  For each batch in this epoch, we will save the gradients (in on_batch_begin)
            # We will then compute means and vars of these gradients

            self._log_gradients = True
            self._batch_weightnorm = []

            self._batch_gradients = [[] for _ in self.model.layers[1:]]

            # Indexes of all the training data samples. These are shuffled and read-in in chunks of SGD_BATCHSIZE
            ixs = list(range(len(self.trn.X)))
            np.random.shuffle(ixs)
            self._batch_todo_ixs = ixs

    def on_batch_begin(self, batch, logs={}):
        if not self._log_gradients:
            # We are not keeping track of batch gradients, so do nothing
            return

        # Sample a batch
        batchsize = self.cfg['SGD_BATCHSIZE']
        cur_ixs = self._batch_todo_ixs[:batchsize]
        # Advance the indexing, so next on_batch_begin samples a different batch
        self._batch_todo_ixs = self._batch_todo_ixs[batchsize:]

        # Get gradients for this batch
        inputs = [self.trn.X[cur_ixs, :],  # Inputs
                  self.trn.Y[cur_ixs],  # Outputs
                  np.ones(len(self.trn.Y))
                  ]
        for lndx, g in enumerate(self.get_gradients(inputs)):
            # g is gradients for weights of lndx's layer
            oneDgrad = np.reshape(g, -1, 1)  # Flatten to one dimensional vector
            try:
                self._batch_gradients[lndx].append(oneDgrad)
            except IndexError as e:
                pass

    def on_epoch_end(self, epoch, logs={}):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            return

        # Get overall performance
        loss = {}
        for cdata, cdataname, istrain in ((self.trn, 'trn', 1), (self.tst, 'tst', 0)):
            loss[cdataname] = self.get_loss([cdata.X, [1, ] * len(cdata.X), cdata.Y, istrain])[0].flat[0]

        data = {
            'weights_norm': [],  # L2 norm of weights
            'gradmean': [],  # Mean of gradients
            'gradstd': [],  # Std of gradients
            'activity_tst': []  # Activity in each layer for test set
        }

        for lndx, layerix in enumerate(self.layerixs):
            clayer = self.model.layers[layerix]

            data['weights_norm'].append(np.linalg.norm(K.get_value(clayer.kernel)))
            try:
                stackedgrads = np.stack(self._batch_gradients[lndx], axis=1)
            except IndexError as e:
                pass
            data['gradmean'].append(np.linalg.norm(stackedgrads.mean(axis=1)))
            data['gradstd'].append(np.linalg.norm(stackedgrads.std(axis=1)))

            data['activity_tst'].append(self.layerfuncs[lndx]([self.tst.X, ])[0])

        fname = self.cfg['SAVE_DIR'] + "/epoch%08d" % epoch
        with open(fname, 'wb') as f:
            pickle.dump({'ACTIVATION': self.cfg['ACTIVATION'], 'epoch': epoch, 'data': data, 'loss': loss}, f,
                         pickle.HIGHEST_PROTOCOL)

