#! /usr/bin/env python

import librosa
import os
from mixin import xtract_mixin
from mixin.params import sample_rate
from argparse import ArgumentParser
import scipy
import scipy.io.wavfile

mypath = os.path.dirname(os.path.abspath(__file__))
default_pretrained = os.path.join(mypath, "./pretrained-models")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--instrumental",
        action="store_true",
        help="use for instrumental songs (excludes the vocal model, which may improve quality of results)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./",
        help="output directory",
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="use the single model approach (invert network output magnitude + original phase)",
    )
    parser.add_argument(
        "--first-prefix",
        action="store_true",
        help="split on underscore in file name, use first prefix only",
    )
    parser.add_argument(
        "--pretrained-model-dir",
        type=str,
        default=default_pretrained,
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
        single_model=args.single_model,
        pretrained_model_dir=args.pretrained_model_dir,
    )

    song_name = os.path.splitext(os.path.basename(args.input))[0]
    if args.first_prefix:
        song_name = song_name.split("_")[0]

    harm_dest = os.path.join(args.outdir, "{0}_harmonic.wav".format(song_name))
    perc_dest = os.path.join(args.outdir, "{0}_percussive.wav".format(song_name))
    vocal_dest = os.path.join(args.outdir, "{0}_vocal.wav".format(song_name))

    print("Writing harmonic and percussive audio files")
    scipy.io.wavfile.write(harm_dest, sample_rate, x_h_out)
    scipy.io.wavfile.write(perc_dest, sample_rate, x_p_out)

    if not args.instrumental:
        print("Writing vocal audio files")
        scipy.io.wavfile.write(vocal_dest, sample_rate, x_v_out)
    print("Done")
