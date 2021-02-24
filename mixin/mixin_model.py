import os
import numpy
import tensorflow
import librosa
from keras.models import Sequential, load_model
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Cropping2D,
    Input,
)
from nsgt import NSGT
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from .params import (
    chunk_size,
    sample_rate,
    dim_1,
    dim_2,
    model_dir,
    checkpoint_dir,
    conv_kernel_crop_dim_1,
    conv_kernel_crop_dim_2,
    components,
    nsgt_scale,
)

physical_devices = tensorflow.config.list_physical_devices("GPU")
tensorflow.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def _pol2cart(rho, phi):
    real = rho * numpy.cos(phi)
    imag = rho * numpy.sin(phi)
    return real + 1j * imag


def create_model():
    model = Sequential()
    model.add(Input(shape=(dim_1, dim_2, 1)))
    model.add(Conv2D(12, kernel_size=3, activation="relu", padding="same"))
    model.add(
        MaxPooling2D(pool_size=(3, 5), strides=None, padding="same", data_format=None)
    )
    model.add(Conv2D(20, kernel_size=3, activation="relu", padding="same"))
    model.add(
        MaxPooling2D(pool_size=(1, 5), strides=None, padding="same", data_format=None)
    )
    model.add(Conv2D(30, kernel_size=3, activation="relu", padding="same"))
    model.add(Conv2D(40, kernel_size=3, activation="relu", padding="same"))
    model.add(Conv2D(30, kernel_size=3, activation="relu", padding="same"))
    model.add(Conv2D(20, kernel_size=3, activation="relu", padding="same"))
    model.add(UpSampling2D(size=(1, 5), data_format=None, interpolation="nearest"))
    model.add(Conv2D(12, kernel_size=3, activation="relu", padding="same"))
    model.add(UpSampling2D(size=(3, 5), data_format=None, interpolation="nearest"))

    model.add(Conv2D(1, kernel_size=1, activation="relu", padding="same"))
    model.add(
        Cropping2D(cropping=((0, conv_kernel_crop_dim_1), (0, conv_kernel_crop_dim_2)))
    )

    model.compile(loss="mae", optimizer="adam", metrics=["mae"])

    return model


class Model:
    def __init__(self, model_file, checkpoint_file=None):
        self.model_file = model_file
        try:
            self.model = load_model(self.model_file)
            self.model.summary()
        except IOError:
            create_dirs = [model_dir, checkpoint_dir]
            print("checking for or creating directories: {0}".format(create_dirs))

            for d in create_dirs:
                if not os.path.isdir(d):
                    os.mkdir(d)
            self.model = create_model()

        self.monitor = EarlyStopping(
            monitor="val_loss", patience=2, min_delta=0, mode="auto", verbose=1
        )

        if checkpoint_file:
            self.checkpoint = ModelCheckpoint(
                checkpoint_file,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode="auto",
            )

    def train(self, XY_train, XY_val, epochs=100, plot=False):
        history = self.model.fit(
            XY_train,
            epochs=epochs,
            callbacks=[self.monitor, self.checkpoint],
            validation_data=XY_val,
            verbose=1,
        )
        if plot:
            plt.plot(history.history["mae"])
            plt.plot(history.history["val_mae"])
            plt.title("model mae")
            plt.ylabel("mae")
            plt.xlabel("epoch")
            plt.legend(["train", "validation"], loc="upper left")
            plt.show()

            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "validation"], loc="upper left")
            plt.show()

    def evaluate_scores(self, XY, name):
        scores = self.model.evaluate(XY)
        print(
            "%s scores: %s: %.2f%%"
            % (name, self.model.metrics_names[1], scores[1] * 100)
        )

    def save(self):
        self.model.save(self.model_file)
        self.model.summary()

    def build_and_summary(self):
        self.model.summary()


def xtract_mixin(x, instrumental=False, single_model=False, pretrained_model_dir=None):
    if pretrained_model_dir is None:
        p_model = components["percussive"]["model_file"]
        h_model = components["harmonic"]["model_file"]
        v_model = components["vocal"]["model_file"]
    else:
        p_model = os.path.join(pretrained_model_dir, "model_percussive.h5")
        h_model = os.path.join(pretrained_model_dir, "model_harmonic.h5")
        v_model = os.path.join(pretrained_model_dir, "model_vocal.h5")

    print("Loading models from:\n\t{0}\n\t{1}\n\t{2}".format(h_model, p_model, v_model))
    percussive_model = Model(p_model).model
    harmonic_model = Model(h_model).model
    vocal_model = Model(v_model).model

    n_samples = x.shape[0]
    n_chunks = int(numpy.ceil(n_samples / chunk_size))
    n_pad = n_chunks * chunk_size - x.shape[0]

    x = numpy.concatenate((x, numpy.zeros(n_pad)))
    x_out_h = numpy.zeros_like(x)
    x_out_p = numpy.zeros_like(x)
    x_out_v = numpy.zeros_like(x)

    # calculate transform parameters
    L = chunk_size
    nsgt = NSGT(nsgt_scale, sample_rate, L, real=True, matrixform=True)

    for chunk in range(n_chunks - 1):
        s = x[chunk * chunk_size : (chunk + 1) * chunk_size]

        # forward transform
        c = nsgt.forward(s)
        C = numpy.asarray(c)

        Cmag_orig, Cphase_orig = librosa.magphase(C)
        Cmag_for_nn = numpy.reshape(Cmag_orig, (1, dim_1, dim_2, 1))

        # inference from model
        Cmag_p = percussive_model.predict(Cmag_for_nn)
        Cmag_p = numpy.reshape(Cmag_p, (dim_1, dim_2))

        Cmag_h = harmonic_model.predict(Cmag_for_nn)
        Cmag_h = numpy.reshape(Cmag_h, (dim_1, dim_2))

        Cmag_v = numpy.zeros_like(Cmag_h)
        if not instrumental:
            Cmag_v = vocal_model.predict(Cmag_for_nn)
            Cmag_v = numpy.reshape(Cmag_v, (dim_1, dim_2))

        if single_model:
            Ch_desired = _pol2cart(Cmag_h, Cphase_orig)
            Cp_desired = _pol2cart(Cmag_p, Cphase_orig)

            if not instrumental:
                Cv_desired = _pol2cart(Cmag_v, Cphase_orig)
        else:
            # soft mask first
            Mp = numpy.ones_like(Cmag_orig)
            Mh = numpy.ones_like(Cmag_orig)
            Mv = numpy.ones_like(Cmag_orig)

            tot = (
                numpy.power(Cmag_p, 2.0)
                + numpy.power(Cmag_h, 2.0)
                + numpy.power(Cmag_v, 2.0)
                + K.epsilon()
            )
            Mp = numpy.divide(numpy.power(Cmag_p, 2.0), tot)
            Mh = numpy.divide(numpy.power(Cmag_h, 2.0), tot)
            Mv = numpy.divide(numpy.power(Cmag_v, 2.0), tot)

            Cp_desired = numpy.multiply(Mp, C)
            Ch_desired = numpy.multiply(Mh, C)
            Cv_desired = numpy.multiply(Mv, C)

        # inverse transform
        s_p = nsgt.backward(Cp_desired)
        s_h = nsgt.backward(Ch_desired)

        s_v = numpy.zeros_like(s_h)
        if not instrumental:
            s_v = nsgt.backward(Cv_desired)

        x_out_p[chunk * chunk_size : (chunk + 1) * chunk_size] = s_p
        x_out_v[chunk * chunk_size : (chunk + 1) * chunk_size] = s_v
        x_out_h[chunk * chunk_size : (chunk + 1) * chunk_size] = s_h

    # strip off padding
    if n_pad > 0:
        x_out_p = x_out_p[:-n_pad]
        x_out_h = x_out_h[:-n_pad]
        x_out_v = x_out_v[:-n_pad]

    x_out_h = x_out_h.astype(numpy.float32)
    x_out_p = x_out_p.astype(numpy.float32)
    x_out_v = x_out_v.astype(numpy.float32)

    return x_out_h, x_out_p, x_out_v
