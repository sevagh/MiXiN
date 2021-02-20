#!/usr/bin/env python3

import sys
import argparse
import os
import numpy
import subprocess
from essentia.standard import MonoLoader
import soundfile

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="prepare_data",
        description="Prepare evaluation datasets for HPSS from instrument stems",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="sample rate (default: 44100 Hz)",
    )
    parser.add_argument(
        "stem_dirs", nargs="+", help="directories containing instrument stems"
    )
    parser.add_argument("--track-limit", type=int, default=-1, help="limit to n tracks")
    parser.add_argument(
        "--segment-limit",
        type=int,
        default=sys.maxsize,
        help="limit to n segments per track",
    )
    parser.add_argument(
        "--segment-offset",
        type=int,
        default=0,
        help="offset of segment to start from (useful to skip intros)",
    )
    parser.add_argument(
        "--segment-duration", type=float, default=30.0, help="segment duration in seconds"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    seq = 0
    for sd in args.stem_dirs:
        for song in os.scandir(sd):
            for dir_name, _, file_list in os.walk(song):
                instruments = [
                    os.path.join(dir_name, f) for f in file_list if f.endswith(".wav")
                ]
                if instruments:
                    print("Found directory containing wav files: %d" % seq)
                    print(os.path.basename(dir_name).replace(" ", "_"))
                    loaded_wavs = [None] * len(instruments)
                    drum_track_index = -1
                    vocal_track_index = -1
                    mix_track_index = -1
                    for i, instrument in enumerate(instruments):
                        if "drum" in instrument.lower():
                            drum_track_index = i
                        elif "vocal" in instrument.lower():
                            vocal_track_index = i
                        elif "mix" in instrument.lower():
                            mix_track_index = i

                        # automatically resamples for us
                        loaded_wavs[i] = MonoLoader(
                            filename=instrument, sampleRate=args.sample_rate
                        )()
                    track_len = len(loaded_wavs[0])

                    # ensure all stems have the same length
                    assert (
                        len(loaded_wavs[i]) == track_len
                        for i in range(1, len(loaded_wavs))
                    )

                    interf_mix_novocal = sum(
                        [
                            l
                            for i, l in enumerate(loaded_wavs)
                            if i
                            not in [
                                drum_track_index,
                                vocal_track_index,
                                mix_track_index,
                            ]
                        ]
                    )
                    interf_mix_vocal = interf_mix_novocal + loaded_wavs[vocal_track_index] 

                    full_mix_novocal = interf_mix_novocal + loaded_wavs[drum_track_index]
                    full_mix_vocal = interf_mix_vocal + loaded_wavs[drum_track_index]

                    seg_samples = int(numpy.floor(args.segment_duration * args.sample_rate))
                    total_segs = int(numpy.floor(track_len / seg_samples))

                    seg_limit = min(total_segs - 1, args.segment_limit)

                    for seg in range(seg_limit):
                        if seg < args.segment_offset:
                            continue
                        seqstr = "%03d%04d" % (seq, seg)

                        seqdirv = os.path.join(data_dir, "{0}v".format(seqstr))
                        seqdirnov = os.path.join(data_dir, "{0}nov".format(seqstr))

                        if not os.path.isdir(seqdirv):
                            os.mkdir(seqdirv)

                        if not os.path.isdir(seqdirnov):
                            os.mkdir(seqdirnov)

                        left = seg * seg_samples
                        right = (seg + 1) * seg_samples

                        harm_path_nov = os.path.join(
                            seqdirnov, "interf.wav"
                        )
                        mix_path_nov = os.path.join(seqdirnov, "mix.wav")
                        perc_path_nov = os.path.join(
                            seqdirnov, "drum.wav"
                        )

                        soundfile.write(
                            harm_path_nov, interf_mix_novocal[left:right], args.sample_rate
                        )
                        soundfile.write(
                            mix_path_nov, full_mix_novocal[left:right], args.sample_rate
                        )

                        # write the drum track
                        soundfile.write(
                            perc_path_nov,
                            loaded_wavs[drum_track_index][left:right],
                            args.sample_rate,
                        )

                        harm_path_v = os.path.join(
                            seqdirv, "interf.wav"
                        )
                        mix_path_v = os.path.join(seqdirv, "mix.wav")
                        perc_path_v = os.path.join(
                            seqdirv, "drum.wav"
                        )

                        soundfile.write(
                            harm_path_v, interf_mix_vocal[left:right], args.sample_rate
                        )
                        soundfile.write(
                            mix_path_v, full_mix_vocal[left:right], args.sample_rate
                        )

                        # write the drum track
                        soundfile.write(
                            perc_path_v,
                            loaded_wavs[drum_track_index][left:right],
                            args.sample_rate,
                        )

                    seq += 1

                    if args.track_limit > -1:
                        if seq == args.track_limit:
                            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
