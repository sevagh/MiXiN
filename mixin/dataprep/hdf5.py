import numpy
import h5py
import itertools
import os
import librosa
import multiprocessing
from nsgt import NSGT, BarkScale
from ..params import (
    chunk_size,
    sample_rate,
    stft_nfft,
    n_frames,
    components,
)

TRAIN = 0.8
VALIDATION = 0.1
TEST = 0.1


def _compute_hdf5_row(tup):
    spec_in = []
    spec_out = []
    all_ndarray_rows = []

    (mix, ref) = tup

    x_mix, _ = librosa.load(mix, sr=sample_rate, mono=True)
    x_ref, _ = librosa.load(ref, sr=sample_rate, mono=True)
    assert x_mix.shape == x_ref.shape

    all_ndarray_rows = []

    n_samples = x_mix.shape[0]
    n_chunks = int(numpy.ceil(n_samples / chunk_size))
    n_pad = n_chunks * chunk_size - x_mix.shape[0]

    x_mix = numpy.concatenate((x_mix, numpy.zeros(n_pad)))
    x_ref = numpy.concatenate((x_ref, numpy.zeros(n_pad)))

    scl = BarkScale(0, 22050, 96)

    # calculate transform parameters
    L = chunk_size
    nsgt = NSGT(scl, sample_rate, L, real=True, matrixform=True)

    for chunk in range(n_chunks - 1):
        x_mix_chunk = x_mix[chunk * chunk_size : (chunk + 1) * chunk_size]
        x_ref_chunk = x_ref[chunk * chunk_size : (chunk + 1) * chunk_size]

        # forward transform
        cmix = nsgt.forward(x_mix_chunk)
        Cmix = numpy.asarray(cmix)

        Cmagmix = numpy.abs(Cmix)

        cref = nsgt.forward(x_ref_chunk)
        Cref = numpy.asarray(cref)

        Cmagref = numpy.abs(Cref)

        spec_in.append(Cmagmix)
        spec_out.append(Cmagref)

    for spec_pairs in zip(spec_in, spec_out):
        all_ndarray_rows.append(
            numpy.concatenate((spec_pairs[0], spec_pairs[1]), axis=1)
        )

    return all_ndarray_rows


def create_hdf5_from_dir(data_dir, n_pool):
    pool = multiprocessing.Pool(n_pool)
    testcases = {"harmonic": [], "percussive": [], "vocal": []}

    for track in os.scandir(data_dir):
        mix = None
        ref_percussive = None
        ref_harmonic = None
        ref_vocal = None

        if os.path.isfile(track):
            # skip the hdf5 file
            continue
        for fname in os.listdir(track):
            if fname == "mix.wav":
                mix = fname
            elif fname == "percussive.wav":
                ref_percussive = fname
            elif fname == "harmonic.wav":
                ref_harmonic = fname
            elif fname == "vocal.wav":
                if "nov" not in track.name:
                    ref_vocal = fname

        mix = os.path.join(data_dir, track, mix)
        ref_percussive = os.path.join(data_dir, track, ref_percussive)
        ref_harmonic = os.path.join(data_dir, track, ref_harmonic)

        testcases["percussive"].append((mix, ref_percussive))
        testcases["harmonic"].append((mix, ref_harmonic))

        if ref_vocal:
            ref_vocal = os.path.join(data_dir, track, ref_vocal)
            testcases["vocal"].append((mix, ref_vocal))

    for component, component_files in components.items():
        with h5py.File(component_files["data_hdf5_file"], "w") as hf:
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

            for i in range(0, len(testcases[component]) - 1, n_pool):
                limited_testcases = testcases[component][i : i + n_pool]

                outputs = list(
                    itertools.chain.from_iterable(
                        pool.starmap(
                            _compute_hdf5_row,
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
                    "{0} chunk {1} TRAIN/TEST/VALIDATION SPLIT:\n\tall data: {2}\n\ttrain: {3}\n\ttest: {4}\n\tvalidation: {5}".format(
                        component,
                        i,
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
