#! /usr/bin/env python

"""
origin is https://github.com/grrr/nsgt

Thomas Grill, 2011-2020
http://grrrr.org/nsgt
"""

import librosa
import os
from primalx import xtract_primitive
from argparse import ArgumentParser
import scipy


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--chunk", type=int, default=44032, help="chunk size in samples, roughly 1s but divisible by 1024, (default=%(default)s")
    parser.add_argument("input", type=str, help="input audio file")
    parser.add_argument('output', type=str, help="output audio file")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error("Input file '%s' not found"%args.input)

    # Read audio data
    x, fs = librosa.load(args.input, sr=None, mono=True)

    x_out = xtract_primitive(x, fs, args.chunk)

    print("Writing audio file '%s'"%args.output)
    scipy.io.wavfile.write(args.output, fs, x_out)
    print("Done")
