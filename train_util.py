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
import tensorflow_io as tfio
import tensorflow as tf
import soundfile
from primalx import Model
from primalx.params import (
    data_dir,
    model_dir,
    checkpoint_dir,
    data_hdf5_file,
    stft_nfft,
    n_frames,
)
from primalx.dataprep import prepare_stems, compute_hdf5_row

TRAIN = 0.8
VALIDATION = 0.1
TEST = 0.1


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
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="don't train, only data prep",
    )
    parser.add_argument(
        "--no-stems",
        action="store_true",
        help="don't prepare stems",
    )
    parser.add_argument(
        "--no-hdf5",
        action="store_true",
        help="don't prepare hdf5 file",
    )

    return parser.parse_args()


def prepare_data(args):
    if args.data_clean:
        try:
            shutil.rmtree(data_dir)
        except FileNotFoundError:
            pass

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    if not args.no_stems:
        prepare_stems(
            args.stem_dirs,
            data_dir,
            args.track_limit,
            args.segment_duration,
            args.segment_limit,
            args.segment_offset,
        )

    if not args.no_stems:
        pool = multiprocessing.Pool(args.n_pool)

        testcases = []

        for track in os.scandir(data_dir):
            mix = None
            ref = None
            if os.path.isfile(track):
                # skip the hdf5 file
                continue
            for fname in os.listdir(track):
                if fname == "drum.wav":
                    ref = fname
                elif fname == "mix.wav":
                    mix = fname
            mix = os.path.join(data_dir, track, mix)
            ref = os.path.join(data_dir, track, ref)
            testcases.append((mix, ref))

        with h5py.File(data_hdf5_file, "w") as hf:
            x_train_dataset = hf.create_dataset(
                "data-x-train",
                (1, n_frames, stft_nfft, 1),
                maxshape=(None, n_frames, stft_nfft, 1),
            )
            y_train_dataset = hf.create_dataset(
                "data-y-train",
                (1, n_frames, stft_nfft, 1),
                maxshape=(None, n_frames, stft_nfft, 1),
            )

            x_test_dataset = hf.create_dataset(
                "data-x-test",
                (1, n_frames, stft_nfft, 1),
                maxshape=(None, n_frames, stft_nfft, 1),
            )
            y_test_dataset = hf.create_dataset(
                "data-y-test",
                (1, n_frames, stft_nfft, 1),
                maxshape=(None, n_frames, stft_nfft, 1),
            )

            x_validation_dataset = hf.create_dataset(
                "data-x-validation",
                (1, n_frames, stft_nfft, 1),
                maxshape=(None, n_frames, stft_nfft, 1),
            )
            y_validation_dataset = hf.create_dataset(
                "data-y-validation",
                (1, n_frames, stft_nfft, 1),
                maxshape=(None, n_frames, stft_nfft, 1),
            )

            for i in range(0, len(testcases) - 1, args.n_pool):
                limited_testcases = testcases[i : i + args.n_pool]

                outputs = list(
                    itertools.chain.from_iterable(
                        pool.starmap(
                            compute_hdf5_row,
                            zip(
                                limited_testcases,
                            ),
                        )
                    )
                )
                to_add = numpy.asarray(outputs)

                train_idx = int(TRAIN * to_add.shape[0])
                test_idx = int(VALIDATION * to_add.shape[0])

                to_add_train = to_add[:train_idx, :, :]
                to_add_test = to_add[train_idx : train_idx + test_idx, :, :]
                to_add_validation = to_add[train_idx + test_idx :, :, :]

                to_add_x_train = to_add_train[:, :, :stft_nfft]
                to_add_y_train = to_add_train[:, :, stft_nfft:]

                to_add_x_test = to_add_test[:, :, :stft_nfft]
                to_add_y_test = to_add_test[:, :, stft_nfft:]

                to_add_x_validation = to_add_validation[:, :, :stft_nfft]
                to_add_y_validation = to_add_validation[:, :, stft_nfft:]

                print(
                    "TRAIN/TEST/VALIDATION SPLIT:\n\tall: {0}\n\ttrain: {1}\n\ttest: {2}\n\tvalidation: {3}".format(
                        to_add.shape,
                        to_add_train.shape,
                        to_add_test.shape,
                        to_add_validation.shape,
                    )
                )

                x_train_dataset.resize(
                    (x_train_dataset.shape[0] + to_add_x_train.shape[0]), axis=0
                )
                x_train_dataset[
                    -to_add_x_train.shape[0] :, :, :
                ] = to_add_x_train.reshape(
                    to_add_x_train.shape[0],
                    to_add_x_train.shape[1],
                    to_add_x_train.shape[2],
                    1,
                )

                y_train_dataset.resize(
                    (y_train_dataset.shape[0] + to_add_y_train.shape[0]), axis=0
                )
                y_train_dataset[
                    -to_add_y_train.shape[0] :, :, :
                ] = to_add_y_train.reshape(
                    to_add_y_train.shape[0],
                    to_add_y_train.shape[1],
                    to_add_y_train.shape[2],
                    1,
                )

                x_test_dataset.resize(
                    (x_test_dataset.shape[0] + to_add_x_test.shape[0]), axis=0
                )
                x_test_dataset[-to_add_x_test.shape[0] :, :, :] = to_add_x_test.reshape(
                    to_add_x_test.shape[0],
                    to_add_x_test.shape[1],
                    to_add_x_test.shape[2],
                    1,
                )

                y_test_dataset.resize(
                    (y_test_dataset.shape[0] + to_add_y_test.shape[0]), axis=0
                )
                y_test_dataset[-to_add_y_test.shape[0] :, :, :] = to_add_y_test.reshape(
                    to_add_y_test.shape[0],
                    to_add_y_test.shape[1],
                    to_add_y_test.shape[2],
                    1,
                )

                x_validation_dataset.resize(
                    (x_validation_dataset.shape[0] + to_add_x_validation.shape[0]),
                    axis=0,
                )
                x_validation_dataset[
                    -to_add_x_validation.shape[0] :, :, :
                ] = to_add_x_validation.reshape(
                    to_add_x_validation.shape[0],
                    to_add_x_validation.shape[1],
                    to_add_x_validation.shape[2],
                    1,
                )

                y_validation_dataset.resize(
                    (y_validation_dataset.shape[0] + to_add_y_validation.shape[0]),
                    axis=0,
                )
                y_validation_dataset[
                    -to_add_y_validation.shape[0] :, :, :
                ] = to_add_y_validation.reshape(
                    to_add_y_validation.shape[0],
                    to_add_y_validation.shape[1],
                    to_add_y_validation.shape[2],
                    1,
                )


def train_network(args):
    if args.train_clean:
        try:
            shutil.rmtree(model_dir)
        except FileNotFoundError:
            pass

        try:
            shutil.rmtree(checkpoint_dir)
        except FileNotFoundError:
            pass

    model = Model()
    model.build_and_summary()

    # input spectrogram
    X_train = tfio.IODataset.from_hdf5(data_hdf5_file, dataset="/data-x-train")
    Y_train = tfio.IODataset.from_hdf5(data_hdf5_file, dataset="/data-y-train")

    X_test = tfio.IODataset.from_hdf5(data_hdf5_file, dataset="/data-x-test")
    Y_test = tfio.IODataset.from_hdf5(data_hdf5_file, dataset="/data-y-test")

    X_validation = tfio.IODataset.from_hdf5(
        data_hdf5_file, dataset="/data-x-validation"
    )
    Y_validation = tfio.IODataset.from_hdf5(
        data_hdf5_file, dataset="/data-y-validation"
    )

    train_data_set = tf.data.Dataset.zip((X_train.batch(64), Y_train.batch(64)))
    test_data_set = tf.data.Dataset.zip((X_test.batch(64), Y_test.batch(64)))
    validation_data_set = tf.data.Dataset.zip(
        (X_validation.batch(64), Y_validation.batch(64))
    )

    model.train(train_data_set, validation_data_set, plot=args.plot_training)
    model.evaluate_scores(train_data_set, "train")
    model.evaluate_scores(validation_data_set, "validation")
    model.evaluate_scores(test_data_set, "test")

    print("saving model")
    model.save()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.stem_dirs is not None:
        prepare_data(args)

    if not args.no_train:
        train_network(args)

    sys.exit(0)
