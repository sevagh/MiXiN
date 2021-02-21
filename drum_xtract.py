#! /usr/bin/env python

import librosa
import os
from primalx import xtract_primal
from primalx.params import sample_rate
from argparse import ArgumentParser
import scipy


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--beta",
        default=None,
        type=float,
        help="hard mask separation factor, default %(default)s",
    )
    parser.add_argument(
        "--power", default=None, type=float, help="soft mask power, default %(default)s"
    )
    parser.add_argument(
        "--instrumental",
        action="store_true",
        help="use for instrumental songs (excludes the vocal model, which may improve quality of results)",
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="use the single model approach (invert network output magnitude + original phase)",
    )
    parser.add_argument("input", type=str, help="input audio file")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error("Input file '%s' not found" % args.input)

    # Read audio data
    x, _ = librosa.load(args.input, sr=sample_rate, mono=True)

    x_h_out, x_p_out, x_v_out = xtract_primal(
        x, instrumental=args.instrumental, power=args.power, beta=args.beta
    )

    print("Writing harmonic and percussive audio files")
    scipy.io.wavfile.write("harmonic.wav", sample_rate, x_h_out)
    scipy.io.wavfile.write("percussive.wav", sample_rate, x_p_out)

    if not args.instrumental:
        print("Writing vocal audio files")
        scipy.io.wavfile.write("vocal.wav", sample_rate, x_v_out)
    print("Done")
