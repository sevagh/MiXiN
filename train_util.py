#!/usr/bin/env python3

import h5py
import sys
import multiprocessing
import argparse
import os
import numpy
import itertools
import tensorflow_io as tfio
import tensorflow as tf
from mixin import Model
from mixin.params import (
    data_dir,
    model_dir,
    checkpoint_dir,
    stft_nfft,
    n_frames,
    batch_size,
    components,
)
from mixin.dataprep import prepare_stems, create_hdf5_from_dir

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
        "--train",
        action="store_true",
        help="train the models",
    )
    parser.add_argument(
        "--prepare-stems",
        action="store_true",
        help=" prepare stems",
    )
    parser.add_argument(
        "--create-hdf5",
        action="store_true",
        help="prepare hdf5 file",
    )

    return parser.parse_args()


def train_network(args):
    for component, component_files in components.items():
        print("Training model for {0}".format(component))
        model = Model(component_files["model_file"], component_files["checkpoint_file"])
        model.build_and_summary()

        # input spectrogram
        X_train = tfio.IODataset.from_hdf5(
            component_files["data_hdf5_file"], dataset="/data-x-train"
        )
        Y_train = tfio.IODataset.from_hdf5(
            component_files["data_hdf5_file"], dataset="/data-y-train"
        )

        X_test = tfio.IODataset.from_hdf5(
            component_files["data_hdf5_file"], dataset="/data-x-test"
        )
        Y_test = tfio.IODataset.from_hdf5(
            component_files["data_hdf5_file"], dataset="/data-y-test"
        )

        X_validation = tfio.IODataset.from_hdf5(
            component_files["data_hdf5_file"], dataset="/data-x-validation"
        )
        Y_validation = tfio.IODataset.from_hdf5(
            component_files["data_hdf5_file"], dataset="/data-y-validation"
        )

        train_data_set = tf.data.Dataset.zip(
            (X_train.batch(batch_size), Y_train.batch(batch_size))
        )
        test_data_set = tf.data.Dataset.zip(
            (X_test.batch(batch_size), Y_test.batch(batch_size))
        )
        validation_data_set = tf.data.Dataset.zip(
            (X_validation.batch(batch_size), Y_validation.batch(batch_size))
        )

        model.train(train_data_set, validation_data_set, plot=args.plot_training)
        model.evaluate_scores(validation_data_set, "validation")
        model.evaluate_scores(test_data_set, "test")

        print("saving model")
        model.save()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.prepare_stems:
        if args.stem_dirs is None:
            raise ValueError("must specify --stem-dirs with --prepare-stems option")

        prepare_stems(
            args.stem_dirs,
            data_dir,
            args.track_limit,
            args.segment_duration,
            args.segment_limit,
            args.segment_offset,
        )
    else:
        print("--prepare-stems not specified, skipping...")

    if args.create_hdf5:
        create_hdf5_from_dir(data_dir, args.n_pool)
    else:
        print("--create-hdf5 not specified, skipping...")

    if args.train:
        train_network(args)
    else:
        print("--train not specified, skipping...")

    sys.exit(0)
