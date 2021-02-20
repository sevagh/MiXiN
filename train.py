#!/usr/bin/env python3

import h5py
import sys
import argparse
import os
import numpy
import keras
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from primalx import create_model
import keras.backend as K
from sklearn.model_selection import train_test_split
import tensorflow
import matplotlib.pyplot as plt

mypath = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(mypath, "model")
checkpoint_dir = os.path.join(mypath, "logdir")
model_file = os.path.join(model_dir, "model.h5")

physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)

checkpoint_file = os.path.join(checkpoint_dir, "model.ckpt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", help="hdf5 file with training data")
    parser.add_argument("plot", action="store_true", help="generate training plots")
    return parser.parse_args()


def main():
    create_dirs = [model_dir, checkpoint_dir]
    print('checking for or creating directories: {0}'.format(create_dirs))

    for d in create_dirs:
        if not os.path.isdir(d):
            os.mkdir(d)

    args = parse_args()

    model = None
    try:
        model = load_model(model_file)
        model.summary()
    except IOError:
        model = create_model()

    monitor = EarlyStopping(monitor="loss", patience=5)

    checkpoint = ModelCheckpoint(
        checkpoint_file,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="auto",
        #save_freq="epoch",
    )

    with h5py.File(args.data_file, "r") as hf:
        data = hf["data"][:]

        X = numpy.copy(data[:, :1025, :])
        Y = numpy.copy(data[:, 1025:, :])

        print(X.shape)
        print(Y.shape)

        # split into 90/10. then pass validation_split to keras fit
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, train_size=0.9, test_size=0.1, random_state=42
        )

        X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
        Y_train = numpy.reshape(Y_train, (Y_train.shape[0], 1, Y_train.shape[1], Y_train.shape[2]))

        X_test = numpy.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
        Y_test = numpy.reshape(Y_test, (Y_test.shape[0], 1, Y_test.shape[1], Y_test.shape[2]))

        history = model.fit(
            X_train,
            Y_train,
            batch_size=10,
            epochs=100,
            callbacks=[monitor, checkpoint],
            validation_split=0.1,  # reserve 10% of the 90% of train data for external validation
            verbose=1,
        )

        train_scores = model.evaluate(X_train, Y_train)
        print(
            "train scores: %s: %.2f%%" % (model.metrics_names[1], train_scores[1] * 100)
        )

        test_scores = model.evaluate(X_test, Y_test)
        print(
            "test scores: %s: %.2f%%" % (model.metrics_names[1], test_scores[1] * 100)
        )

        print("saving model")
        model.save(model_file)

        if args.plot:
            print([k for k in history.history.keys()])
            #  "Accuracy"
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
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


if __name__ == "__main__":
    sys.exit(main())
