import numpy
import librosa
from nsgt import NSGT, BarkScale
from ..params import chunk_size, sample_rate


def compute_hdf5_row(tup):
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
