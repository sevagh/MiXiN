import sys
import os
import json
import numpy
from keras.models import load_model
import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense
import numpy
from nsgt import NSGT, LogScale, LinScale, MelScale, OctScale, BarkScale
import scipy.io.wavfile
import librosa
from librosa.decompose import hpss
from librosa.core import stft, istft
from librosa.util import fix_length
from scipy.signal import medfilt
import os
import matplotlib.pyplot as plt
import tensorflow
import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import keras.backend as K

mypath = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(mypath, "../model")
checkpoint_dir = os.path.join(mypath, "../logdir")

physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)

model_file = os.path.join(model_dir, "model.h5")
checkpoint_file = os.path.join(checkpoint_dir, "model.ckpt")

with open(os.path.join(mypath, '../params.json')) as f:
    params = json.load(f)
    nn_time_win = params["stft_window_size"]
    chunk_size = params["chunk_size"]


def _create_model():
    model = Sequential()

    model.add(Conv2D(12, kernel_size=3, activation='relu',padding='same')) #13*1023*12
    model.add(MaxPooling2D(pool_size=(3, 5), strides=None, padding='same', data_format=None))#4*204*12
    model.add(Conv2D(20, kernel_size=3, activation='relu',padding='same'))#2*202*20
    model.add(MaxPooling2D(pool_size=(1, 5), strides=None, padding='same', data_format=None))#2*40*20
    model.add(Conv2D(30, kernel_size=3, activation='relu',padding='same'))#2*40*20 --------------------------
    model.add(Conv2D(40, kernel_size=3, activation='relu',padding='same'))
    model.add(Conv2D(30, kernel_size=3, activation='relu',padding='same'))
    model.add(Conv2D(20, kernel_size=3, activation='relu',padding='same'))
    model.add(UpSampling2D(size=(1, 5), data_format=None, interpolation='nearest'))
    model.add(Conv2D(12, kernel_size=3, activation='relu',padding='same'))
    model.add(UpSampling2D(size=(3, 5), data_format=None, interpolation='nearest'))
    model.add(Conv2D(1, kernel_size=3, activation='relu',padding='same'))
    model.compile(loss='mae', optimizer='adam')

    return model


class Model:
    def __init__(self):
        try:
            self.model = load_model(model_file)
            self.model.summary()
        except IOError:
            create_dirs = [model_dir, checkpoint_dir]
            print('checking for or creating directories: {0}'.format(create_dirs))

            for d in create_dirs:
                if not os.path.isdir(d):
                    os.mkdir(d)
            self.model = _create_model()

        self.monitor = EarlyStopping(monitor="loss", patience=5)

        self.checkpoint = ModelCheckpoint(
            checkpoint_file,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="auto",
        )

    def train(self, X, Y, batch_size=10, epochs=100, validation_split=0.1, plot=False):
        history = self.model.fit(
            X,
            Y,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[self.monitor, self.checkpoint],
            validation_split=validation_split,
            verbose=1,
        )
        if plot:
            #  "Accuracy"
            plt.plot(history.history['mae'])
            plt.plot(history.history['val_mae'])
            plt.title('model mae')
            plt.ylabel('mae')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
            # "Loss"
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()

    def evalute_scores(self, X, Y, name):
        train_scores = self.model.evaluate(X, Y)
        print(
            "%s scores: %s: %.2f%%" % (name, self.model.metrics_names[1], train_scores[1] * 100)
        )

        test_scores = self.model.evaluate(X_test, Y_test)
        print(
            "test scores: %s: %.2f%%" % (self.model.metrics_names[1], test_scores[1] * 100)
        )

    def save(self):
        self.model.save(model_file)
        self.model.summary()



def xtract_primal(x, fs):
    model = Model().model

    n_samples = x.shape[0]
    n_chunks = int(numpy.ceil(n_samples/chunk_size))
    n_pad = n_chunks*chunk_size - x.shape[0]

    x = numpy.concatenate((x, numpy.zeros(n_pad)))
    x_out = numpy.zeros_like(x)

    # 96 bins per octave on the bark scale
    # bark chosen as it might better represent low-frequency sounds (drums)
    scl = BarkScale(0, 22050, 96)

    # calculate transform parameters
    L = chunk_size
    nsgt = NSGT(scl, fs, L, real=True, matrixform=True)

    for chunk in range(n_chunks-1):
        s = x[chunk*chunk_size:(chunk+1)*chunk_size]

        # forward transform 
        c = nsgt.forward(s)
        C = numpy.asarray(c)

        Cmag = numpy.abs(C)

        # two iterations of CQT median filtering
        for i in range(2):
            _, Mp = hpss(Cmag, power=2.0, margin=1.0, kernel_size=(17, 7), mask=True)
            Cp = numpy.multiply(Mp, C)
            Cmag = Cp

        # inverse transform 
        s_r = nsgt.backward(Cp)

        perc_time_win = 2048

        S = stft(s_r, n_fft=2*perc_time_win, win_length=perc_time_win, hop_length=int(0.5*perc_time_win))
        Smag = numpy.abs(S)

        # two iterations of STFT median filtering
        for i in range(2):
            _, Mp = hpss(Smag, power=2.0, margin=1.0, kernel_size=(17, 17), mask=True)
            Sp = numpy.multiply(Mp, S)
            Smag = Sp

        s_p = fix_length(istft(Sp, win_length=perc_time_win, hop_length=int(0.5*perc_time_win)), len(s_r))

        # last step, apply learned network
        Sp = stft(s_p, n_fft=2*nn_time_win, win_length=nn_time_win, hop_length=int(0.5*nn_time_win))
        Spmag = numpy.abs(Sp)
        print(Spmag.shape)

        Spmag_for_nn = numpy.reshape(Spmag, (1, 1, 1025, 22))

        # inference from model
        Spmag_from_nn = model.predict(Spmag_for_nn)
        print(Spmag_from_nn.shape)
        Spmag_nn = numpy.reshape(Spmag_from_nn, (1025, 22))

        Mp = numpy.divide(Spmag_nn, Spmag)
        Sp_nn = numpy.multiply(Mp, Sp)

        s_p_nn = fix_length(istft(Sp_nn, win_length=nn_time_win, hop_length=int(0.5*nn_time_win)), len(s_p))
        x_out[chunk*chunk_size:(chunk+1)*chunk_size] = s_p_nn

    # strip off padding
    if n_pad > 0:
        x_out = x_out[:-n_pad]
    return x_out
