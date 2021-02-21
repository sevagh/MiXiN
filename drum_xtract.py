#! /usr/bin/env python

import librosa
import os
from mixin import xtract_mixin
from mixin.params import sample_rate
from argparse import ArgumentParser
import scipy
import scipy.io.wavfile


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--power", default=2.0, type=float, help="soft mask power, default %(default)s"
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
    parser.add_argument(
        "--pretrained-model-dir",
        type=str,
        default=None,
        help="path to pretrained model directory (default of None uses ./model)",
    )
    parser.add_argument("input", type=str, help="input audio file")

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error("Input file '%s' not found" % args.input)

    # Read audio data
    x, _ = librosa.load(args.input, sr=sample_rate, mono=True)

    x_h_out, x_p_out, x_v_out = xtract_mixin(
        x,
        instrumental=args.instrumental,
        power=args.power,
        single_model=args.single_model,
        pretrained_model_dir=args.pretrained_model_dir,
    )

    song_name = os.path.splitext(os.path.basename(args.input))[0]

    print("Writing harmonic and percussive audio files")
    scipy.io.wavfile.write("{0}_harmonic.wav".format(song_name), sample_rate, x_h_out)
    scipy.io.wavfile.write("{0}_percussive.wav".format(song_name), sample_rate, x_p_out)

    if not args.instrumental:
        print("Writing vocal audio files")
        scipy.io.wavfile.write("{0}_vocal.wav".format(song_name), sample_rate, x_v_out)
    print("Done")
