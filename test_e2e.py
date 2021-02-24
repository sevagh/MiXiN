#!/usr/bin/env python

import subprocess
import os
import h5py
import sys
import librosa
import shutil
from mixin.params import (
        data_dir,
        model_dir,
        checkpoint_dir,
        sample_rate,
        stft_nfft = 1948,
        n_frames = 96,
)


# print colored outputs
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def errexit(message):
    print(f"{bcolors.FAIL}[MiXiN e2e test] {message}{bcolors.ENDC}", file=sys.stderr)
    sys.exit(1)

def nextstep(message):
    print(f"{bcolors.OKBLUE}[MiXiN e2e test] {message}{bcolors.ENDC}", file=sys.stderr)

def success(message):
    print(f"{bcolors.OKGREEN}[MiXiN e2e test] {message}{bcolors.ENDC}", file=sys.stderr)


if __name__ == '__main__':
    #nextstep('Checking if dirs exist... dont want to overwrite user training')
    #if os.path.isdir(data_dir) or os.path.isdir(model_dir) or os.path.isdir(checkpoint_dir):
    #    errexit('delete data, model, and logdir before running tests')

    #nextstep('Checking if STEM_DIR is defined')
    #stem_dir = os.environ.get('STEM_DIR', None)
    #if not stem_dir:
    #    errexit('specify STEM_DIR, e.g. STEM_DIR=/path/to/musdb18hq/test')

    #nextstep('Preparing stems with train util')
    #try:
    #    stem_out = subprocess.check_output(
    #        './train_util.py --prepare-stems --stem-dirs {0} --segment-offset 1 --segment-limit 3 --track-limit 2 --segment-duration 10'.format(stem_dir), shell=True)
    #except subprocess.CalledProcessError as e:
    #    errexit('error when running train util: {0}'.format(str(e)))

    #nextstep('Verifying prepared stems')
    #segment_dirs = os.listdir(data_dir)
    #track_1_count = len([s for s in segment_dirs if s.startswith('001')])
    #track_2_count = len([s for s in segment_dirs if s.startswith('000')])
    #novocal_count = len([s for s in segment_dirs if s.endswith('nov')])
    #total = len(segment_dirs)
    #print(track_1_count)
    #print(track_2_count)
    #print(novocal_count)
    #print(total)
    #if track_1_count != track_2_count or track_1_count != 4 or novocal_count != 4 or total != 8:
    #    errexit('--prepare-stems did not do what was expected...')

    nextstep('Verifying prepared segments for audio properties')
    try:
        n_tested = 0
        for vocaldir in ['0000001v', '0000002v', '0010001v', '0010002v']:
            for track in ['mix.wav', 'vocal.wav', 'harmonic.wav', 'percussive.wav']:
                audio_segment = os.path.join(data_dir, vocaldir, track)
                x, fs = librosa.load(audio_segment, sr=None)
                n_tested += 1
                if len(x) != 10*fs or fs != 44100:
                    errexit('tested track: {0} - wrong duration {1} or sample rate {2}'.format(
                        audio_segment, float(len(x))/fs, fs))
        for novocaldir in ['0000001nov', '0000002nov', '0010001nov', '0010002nov']:
            for track in ['mix.wav', 'harmonic.wav', 'percussive.wav']:
                audio_segment = os.path.join(data_dir, novocaldir, track)
                x, fs = librosa.load(audio_segment, sr=None)
                n_tested += 1
                if len(x) != 10*fs or fs != 44100:
                    errexit('tested track: {0} - wrong duration {1} or sample rate {2}'.format(
                        audio_segment, float(len(x))/fs, fs))
        if n_tested != 28:
            errexit('expected 28 segments, got {0}'.format(n_tested))
    except Exception as e:
        errexit('error when veriying prepared stem segments: {0}'.format(str(e)))
    success('good')

    nextstep('Creating hdf5 with train util')
    #try:
    #    subprocess.check_output(
    #    './train_util.py --create-hdf5', shell=True)
    #except subprocess.CalledProcessError as e:
    #    errexit('error when running train util: {0}'.format(str(e)))

    hdf5_files = ['data_harmonic.hdf5', 'data_percussive.hdf5', 'data_vocal.hdf5']

    data_files = [f for f in os.listdir(data_dir) if f.endswith('.hdf5')]
    if len(data_files) != 3 or not all([h in data_files for h in hdf5_files]):
        errexit('--create-hdf5 did not do what was expected...')

    success('good')

    nextstep('Verifying dimensionality of hdf5 files')
    for hdf5_file in hdf5_files:
        hdf5_file_p = os.path.join(data_dir, hdf5_file)
        try:
            with h5py.File(hdf5_file_p, "r") as hf:
                xtr = hf['data-x-train']
                ytr = hf['data-y-train']
                xte = hf['data-x-test']
                yte = hf['data-y-test']
                xv = hf['data-x-validation']
                yv = hf['data-y-validation']

                if hdf5_file != 'data_vocal.hdf5':
                    tr_len = 65
                    val_len = 9
                else:
                    tr_len = 33
                    val_len = 5

                    if xtr.shape != (tr_len, 96, 1948, 1) or \
                        ytr.shape != (tr_len, 96, 1948, 1) or \
                        xte.shape != (val_len, 96, 1948, 1) or \
                        yte.shape != (val_len, 96, 1948, 1) or \
                        xv.shape != (val_len, 96, 1948, 1) or \
                        yv.shape != (val_len, 96, 1948, 1):
                            errexit('hdf5 file has incorrect datasets or dimensions. expected data-{x,y}-{train,test,validation}')
        except Exception as e:
            errexit('error when loading datasets from hdf5: {0}'.format(str(e)))
    success('good')

    #print(subprocess.check_output(
    #    './train_util.py --train', shell=True))
