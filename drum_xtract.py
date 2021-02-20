#! /usr/bin/env python

import librosa
import numpy
import os
from primalx import xtract_primitive, xtract_primal
from argparse import ArgumentParser
import scipy


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--chunk", type=int, default=44032, help="chunk size in samples, roughly 1s but divisible by 1024, (default=%(default)s")
    parser.add_argument("--use-nn", action="store_true" , help="use trained neural network vs plain/primitive separation")
    parser.add_argument("input", type=str, help="input audio file")
    parser.add_argument('output', type=str, help="output audio file")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error("Input file '%s' not found"%args.input)

    # Read audio data
    x, fs = librosa.load(args.input, sr=None, mono=True)

    x_out = numpy.zeros_like(x)
    if args.use_nn:
        x_out = xtract_primal(x, fs, args.chunk)
    else:
        x_out = xtract_primitive(x, fs, args.chunk)

    print("Writing audio file '%s'"%args.output)
    scipy.io.wavfile.write(args.output, fs, x_out)
    print("Done")
