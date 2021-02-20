import h5py
import json
import sys
import os
import numpy
import multiprocessing
import itertools
import scipy
import librosa
from librosa.core import stft
from .. import xtract_primitive


mypath = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(mypath, "data")
with open(os.path.join(mypath, "../../params.json")) as f:
    params = json.load(f)
chunk_size = params["chunk_size"]
perc_time_win = params["stft_window_size"]


def compute_hdf5_row(tup):
    spec_in = []
    spec_out = []
    all_ndarray_rows = []

    (mix, ref) = tup

    x_mix, fs_mix = librosa.load(mix, sr=None, mono=True)
    x_ref, fs_ref = librosa.load(ref, sr=None, mono=True)
    assert fs_mix == fs_ref
    assert x_mix.shape == x_ref.shape

    fs = fs_mix

    all_ndarray_rows = []

    n_samples = x_mix.shape[0]
    n_chunks = int(numpy.ceil(n_samples/chunk_size))
    n_pad = n_chunks*chunk_size - x_mix.shape[0]

    x_mix = numpy.concatenate((x_mix, numpy.zeros(n_pad)))
    x_ref = numpy.concatenate((x_ref, numpy.zeros(n_pad)))

    print('Applying primitive drum extraction')
    x_sep = xtract_primitive(x_mix, fs)

    for chunk in range(n_chunks-1):
        x_sep_chunk = x_sep[chunk*chunk_size:(chunk+1)*chunk_size]
        x_ref_chunk = x_ref[chunk*chunk_size:(chunk+1)*chunk_size]

        Xsep = stft(x_sep_chunk, n_fft=2*perc_time_win, win_length=perc_time_win, hop_length=int(0.5*perc_time_win))
        Xsepmag = numpy.abs(Xsep)

        Xref = stft(x_ref_chunk, n_fft=2*perc_time_win, win_length=perc_time_win, hop_length=int(0.5*perc_time_win))
        Xrefmag = numpy.abs(Xref)

        spec_in.append(Xsepmag)
        print(Xsepmag.shape)
        spec_out.append(Xrefmag)
        print(Xrefmag.shape)

    for spec_pairs in zip(spec_in, spec_out):
        all_ndarray_rows.append(numpy.concatenate((spec_pairs[0], spec_pairs[1])))

    return all_ndarray_rows
