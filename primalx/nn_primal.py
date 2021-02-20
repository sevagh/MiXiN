import sys
import os
import numpy
from keras.models import load_model
import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,UpSampling2D
import numpy
from nsgt import NSGT, LogScale, LinScale, MelScale, OctScale, BarkScale
import scipy.io.wavfile
import librosa
from librosa.decompose import hpss
from librosa.core import stft, istft
from librosa.util import fix_length
from scipy.signal import medfilt
import os


def create_model():
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
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])

    return model


def _load_model():
    physical_devices = tensorflow.config.list_physical_devices("GPU")
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    mypath = os.path.dirname(os.path.abspath(__file__))

    # the trained model
    model_file = os.path.join(mypath, "../model/model.h5")
    model = load_model(model_file)
    model.summary()
    return model


def xtract_primal(x, fs, chunk_size):
    model = _load_model()

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

        nn_time_win = 512

        # last step, apply learned network
        Sp = stft(s_p, n_fft=2*nn_time_win, win_length=nn_time_win, hop_length=int(0.5*nn_time_win))
        Spmag = numpy.abs(Sp)
        Spmag_for_nn = numpy.reshape(Spmag, (1, 1025, 173))

        # inference from model
        Spmag_from_nn = model.predict(Spmag_for_nn)
        Spmag_nn = numpy.reshape(Spmag_from_nn, (1025, 173))

        Mp = numpy.divide(Spmag_nn, Spmag)
        Sp_nn = numpy.multiply(Mp, Sp)

        s_p_nn = fix_length(istft(Sp_nn, win_length=nn_time_win, hop_length=int(0.5*nn_time_win)), len(s_p))
        x_out[chunk*chunk_size:(chunk+1)*chunk_size] = s_p_nn

    # strip off padding
    if n_pad > 0:
        x_out = x_out[:-n_pad]
    return x_out
