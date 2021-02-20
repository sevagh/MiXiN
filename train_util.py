#!/usr/bin/env python3

import h5py
import sys
import multiprocessing
import argparse
import os
import numpy
import subprocess
import shutil
import itertools
from essentia.standard import MonoLoader
import soundfile
from primalx import Model
from primalx.dataprep import prepare_stems, compute_hdf5_row
from sklearn.model_selection import train_test_split

mypath = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(mypath, "data")
model_dir = os.path.join(mypath, "model")
checkpoint_dir = os.path.join(mypath, "logdir")
data_hdf5_file = os.path.join(data_dir, "data.hdf5")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stem-dirs", nargs="+", help="directories containing instrument stems"
    )
    parser.add_argument("--track-limit", type=int, default=-1, help="limit to n tracks")
    parser.add_argument(
        "--segment-limit",
        type=int,
        default=sys.maxsize,
        help="limit to n segments per track",
    )
    parser.add_argument(
        "--segment-offset",
        type=int,
        default=0,
        help="offset of segment to start from (useful to skip intros)",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=30.0,
        help="segment duration in seconds",
    )
    parser.add_argument("--hdf5-in", help="path to input hdf5 file for training")
    parser.add_argument(
        "--plot-training", action="store_true", help="generate training plots"
    )
    parser.add_argument(
        "--n-pool",
        type=int,
        default=multiprocessing.cpu_count(),
        help="size of python multiprocessing pool (default: %(default)s)",
    )
    parser.add_argument(
        "--data-clean", action="store_true", help="delete data before re-prepping"
    )
    parser.add_argument(
        "--train-clean",
        action="store_true",
        help="delete model and checkpoints before re-prepping",
    )

    return parser.parse_args()


def prepare_data(args):
    if args.data_clean:
        shutil.rmtree(data_dir)

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    prepare_stems(
        args.stem_dirs,
        data_dir,
        args.track_limit,
        args.segment_duration,
        args.segment_limit,
        args.segment_offset,
    )

    pool = multiprocessing.Pool(args.n_pool)

    testcases = []

    for track in os.scandir(data_dir):
        mix = None
        ref = None
        for fname in os.listdir(track):
            if fname == "drum.wav":
                ref = fname
            elif fname == "mix.wav":
                mix = fname
        mix = os.path.join(data_dir, track, mix)
        ref = os.path.join(data_dir, track, ref)
        testcases.append((mix, ref))

    outputs = list(
        itertools.chain.from_iterable(
            pool.starmap(
                compute_hdf5_row,
                zip(
                    testcases,
                ),
            )
        )
    )

    with h5py.File(data_hdf5_file, "w") as hf:
        hf.create_dataset("data", data=outputs)


def train_network(args):
    if args.train_clean:
        shutil.rmtree(model_dir)
        shutil.rmtree(checkpoint_dir)

    model = Model()

    with h5py.File(data_hdf5_file, "r") as hf:
        data = hf["data"][:]

        X = numpy.copy(data[:, :1025, :])
        Y = numpy.copy(data[:, 1025:, :])
        print(X.shape)
        print(Y.shape)

        # split into 90/10. then pass validation_split to keras fit
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, train_size=0.9, test_size=0.1, random_state=42
        )

        X_train = numpy.reshape(
            X_train, (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
        )
        Y_train = numpy.reshape(
            Y_train, (Y_train.shape[0], 1, Y_train.shape[1], Y_train.shape[2])
        )
        # Y_train = numpy.reshape(Y_train, (Y_train.shape[0], 1, 1025*87))

        X_test = numpy.reshape(
            X_test, (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
        )
        Y_test = numpy.reshape(
            Y_test, (Y_test.shape[0], 1, Y_test.shape[1], Y_test.shape[2])
        )
        # Y_test = numpy.reshape(Y_test, (Y_test.shape[0], 1, 1025*87))

        print(X_train.shape)
        print(Y_train.shape)

        print(X_test.shape)
        print(Y_test.shape)

        model.train(X_train, Y_train, args.plot_training)
        model.evaluate_scores(X_train, Y_train, "train")
        model.evaluate_scores(X_test, Y_test, "test")

        print("saving model")
        model.save()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.stem_dirs is not None:
        prepare_data(args)
    else:
        train_network(args)

    sys.exit(0)
