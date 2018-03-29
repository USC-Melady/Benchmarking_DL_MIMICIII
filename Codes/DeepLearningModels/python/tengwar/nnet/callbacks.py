# -*- coding: utf-8 -*-

import os
import warnings
import sys
from keras.callbacks import Callback
import numpy as np


class EarlyStoppingRestoringWeights(Callback):
    def __init__(self, monitor='val_loss', patience=0, verbose=0, 
                 weight_path = None):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.weight_path = weight_path
        self.best = np.Inf
        self.wait = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" \
                % (self.monitor), RuntimeWarning)
        sys.stdout.flush()
        if current < self.best:
            self.best = current
            self.model.save_weights(self.weight_path, overwrite=True)
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping" % (epoch))
                self.model.stop_training = True
            self.wait += 1

    def on_train_end(self, epoch, logs={}):
        try:
            self.model.load_weights(self.weight_path)
            os.remove(self.weight_path)
        except Exception as exc:
            print 'error while loading and removing weights', str(exc)
