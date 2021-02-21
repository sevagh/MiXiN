import numpy
from nsgt import NSGT, LogScale, LinScale, MelScale, OctScale, BarkScale
import scipy.io.wavfile
import librosa
from librosa.decompose import hpss
from librosa.core import stft, istft
from librosa.util import fix_length
from scipy.signal import medfilt
import os
from .params import chunk_size, sample_rate


def xtract_primitive(x):
    n_samples = x.shape[0]
    n_chunks = int(numpy.ceil(n_samples / chunk_size))
    n_pad = n_chunks * chunk_size - x.shape[0]

    x = numpy.concatenate((x, numpy.zeros(n_pad)))
    x_out = numpy.zeros_like(x)

    # 96 bins per octave on the bark scale
    # bark chosen as it might better represent low-frequency sounds (drums)
    scl = BarkScale(0, 22050, 96)

    # calculate transform parameters
    L = chunk_size
    nsgt = NSGT(scl, sample_rate, L, real=True, matrixform=True)

    for chunk in range(n_chunks - 1):
        s = x[chunk * chunk_size : (chunk + 1) * chunk_size]

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

        S = stft(
            s_r,
            n_fft=2 * perc_time_win,
            win_length=perc_time_win,
            hop_length=int(0.5 * perc_time_win),
        )
        Smag = numpy.abs(S)

        # two iterations of STFT median filtering
        for i in range(2):
            _, Mp = hpss(Smag, power=2.0, margin=1.0, kernel_size=(17, 17), mask=True)
            Sp = numpy.multiply(Mp, S)
            Smag = Sp

        s_p = fix_length(
            istft(Sp, win_length=perc_time_win, hop_length=int(0.5 * perc_time_win)),
            len(s_r),
        )
        x_out[chunk * chunk_size : (chunk + 1) * chunk_size] = s_p

    # strip off padding
    if n_pad > 0:
        x_out = x_out[:-n_pad]
    return x_out
