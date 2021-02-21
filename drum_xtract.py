#! /usr/bin/env python

import librosa
import numpy
import json
import os
from primalx import xtract_primitive, xtract_primal
from primalx.params import sample_rate
from argparse import ArgumentParser
import scipy


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--use-nn",
        action="store_true",
        help="use trained neural network vs plain DSP separation",
    )
    parser.add_argument("input", type=str, help="input audio file")
    parser.add_argument("output", type=str, help="output audio file")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error("Input file '%s' not found" % args.input)

    # Read audio data
    x, _ = librosa.load(args.input, sr=sample_rate, mono=True)

    x_out = numpy.zeros_like(x)
    if args.use_nn:
        x_out = xtract_primal(x)
    else:
        x_out = xtract_primitive(x)

    print("Writing audio file '%s'" % args.output)
    scipy.io.wavfile.write(args.output, sample_rate, x_out)
    print("Done")
