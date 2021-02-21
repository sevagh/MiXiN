import os
import numpy
from essentia.standard import MonoLoader
import soundfile
from ..params import sample_rate


"""
take path to instrument stems
prepare vocal and non-vocal mixes
"""


def prepare_stems(
    stem_dirs, data_dir, track_limit, segment_duration, segment_limit, segment_offset
):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    seq = 0
    for sd in stem_dirs:
        for song in os.scandir(sd):
            for dir_name, _, file_list in os.walk(song):
                instruments = [
                    os.path.join(dir_name, f) for f in file_list if f.endswith(".wav")
                ]
                if instruments:
                    print("Found directory containing wav files: %d" % seq)
                    print(os.path.basename(dir_name).replace(" ", "_"))
                    loaded_wavs = [None] * len(instruments)
                    percussive_track_index = -1
                    vocal_track_index = -1
                    mix_track_index = -1
                    for i, instrument in enumerate(instruments):
                        if "drum" in instrument.lower():
                            percussive_track_index = i
                        elif "vocal" in instrument.lower():
                            vocal_track_index = i
                        elif "mix" in instrument.lower():
                            mix_track_index = i

                        # automatically resamples for us
                        loaded_wavs[i] = MonoLoader(
                            filename=instrument, sampleRate=sample_rate
                        )()
                    track_len = len(loaded_wavs[0])

                    # ensure all stems have the same length
                    assert (
                        len(loaded_wavs[i]) == track_len
                        for i in range(1, len(loaded_wavs))
                    )

                    harmonic_mix = sum(
                        [
                            l
                            for i, l in enumerate(loaded_wavs)
                            if i
                            not in [
                                percussive_track_index,
                                vocal_track_index,
                                mix_track_index,  # skip auto-mix, make our own
                            ]
                        ]
                    )
                    full_mix_vocal = (
                        harmonic_mix
                        + loaded_wavs[vocal_track_index]
                        + loaded_wavs[percussive_track_index]
                    )

                    full_mix_novocal = (
                        harmonic_mix + loaded_wavs[percussive_track_index]
                    )

                    seg_samples = int(numpy.floor(segment_duration * sample_rate))
                    total_segs = int(numpy.floor(track_len / seg_samples))

                    seg_limit = min(total_segs - 1, segment_limit)

                    for seg in range(seg_limit):
                        if seg < segment_offset:
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

                        harm_path_nov = os.path.join(seqdirnov, "harmonic.wav")
                        mix_path_nov = os.path.join(seqdirnov, "mix.wav")
                        perc_path_nov = os.path.join(seqdirnov, "percussive.wav")

                        soundfile.write(
                            harm_path_nov, harmonic_mix[left:right], sample_rate
                        )
                        soundfile.write(
                            mix_path_nov, full_mix_novocal[left:right], sample_rate
                        )

                        # write the percussive track
                        soundfile.write(
                            perc_path_nov,
                            loaded_wavs[percussive_track_index][left:right],
                            sample_rate,
                        )

                        harm_path_v = os.path.join(seqdirv, "harmonic.wav")
                        vocal_path_v = os.path.join(seqdirv, "vocal.wav")
                        mix_path_v = os.path.join(seqdirv, "mix.wav")
                        perc_path_v = os.path.join(seqdirv, "percussive.wav")

                        soundfile.write(
                            harm_path_v, harmonic_mix[left:right], sample_rate
                        )
                        soundfile.write(
                            mix_path_v, full_mix_vocal[left:right], sample_rate
                        )
                        soundfile.write(
                            vocal_path_v,
                            loaded_wavs[vocal_track_index][left:right],
                            sample_rate,
                        )

                        # write the percussive track
                        soundfile.write(
                            perc_path_v,
                            loaded_wavs[percussive_track_index][left:right],
                            sample_rate,
                        )

                    seq += 1
                    print("wrote seq {0}".format(seq))

                    if track_limit > -1:
                        if seq == track_limit:
                            return 0

    return 0
