#!/usr/bin/env python3

import h5py
import sys
import argparse
import os
import numpy
import multiprocessing
import itertools
import scipy
import librosa
from librosa.core import stft
from primalx import xtract_primitive


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_file", help="path to write hdf5 data file")
    parser.add_argument(
        "--n-pool",
        type=int,
        default=multiprocessing.cpu_count(),
        help="size of python multiprocessing pool (default: %(default)s)",
    )
    parser.add_argument("--chunk", type=int, default=44032, help="chunk size in samples, roughly 1s but divisible by 1024 (default=%(default)s")

    return parser.parse_args()


data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
perc_time_win = 512


def compute_single_testcase(mixref_tup, chunk_size):
    spec_in = []
    spec_out = []
    all_ndarray_rows = []

    (mix, ref) = mixref_tup

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
    x_sep = xtract_primitive(x_mix, fs, chunk_size)

    for chunk in range(n_chunks-1):
        x_sep_chunk = x_sep[chunk*chunk_size:(chunk+1)*chunk_size]
        x_ref_chunk = x_ref[chunk*chunk_size:(chunk+1)*chunk_size]

        Xsep = stft(x_sep_chunk, n_fft=2*perc_time_win, win_length=perc_time_win, hop_length=int(0.5*perc_time_win))
        Xsepmag = numpy.abs(Xsep)

        Xref = stft(x_ref_chunk, n_fft=2*perc_time_win, win_length=perc_time_win, hop_length=int(0.5*perc_time_win))
        Xrefmag = numpy.abs(Xref)

        spec_in.append(Xsepmag)
        spec_out.append(Xrefmag)

    for spec_pairs in zip(spec_in, spec_out):
        all_ndarray_rows.append(numpy.concatenate((spec_pairs[0], spec_pairs[1])))

    return all_ndarray_rows


def main():
    args = parse_args()

    pool = multiprocessing.Pool(args.n_pool)
    #testcases = sorted([f for f in os.listdir(data_dir) if f.endswith(".wav")])

    testcases = []

    for track in os.scandir(data_dir):
        mix = None
        ref = None
        for fname in os.listdir(track):
            if fname == 'drum.wav':
                ref = fname
            elif fname == 'mix.wav':
                mix = fname
        mix = os.path.join(data_dir, track, mix)
        ref = os.path.join(data_dir, track, ref)
        testcases.append((mix, ref))

    outputs = list(
        itertools.chain.from_iterable(
            pool.starmap(
                compute_single_testcase,
                zip(
                    testcases,
                    itertools.repeat(args.chunk)
                )
            )
        )
    )

    with h5py.File(args.out_file, "w") as hf:
        hf.create_dataset("data", data=outputs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
