import os


mypath = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(mypath, "../data")
model_dir = os.path.join(mypath, "../model")
checkpoint_dir = os.path.join(mypath, "../logdir")

components = {
    "percussive": {
        "data_hdf5_file": os.path.join(data_dir, "data_percussive.hdf5"),
        "model_file": os.path.join(model_dir, "model_percussive.h5"),
        "checkpoint_file": os.path.join(model_dir, "model_percussive.ckpt"),
    },
    "harmonic": {
        "data_hdf5_file": os.path.join(data_dir, "data_harmonic.hdf5"),
        "model_file": os.path.join(model_dir, "model_harmonic.h5"),
        "checkpoint_file": os.path.join(model_dir, "model_harmonic.ckpt"),
    },
    "vocal": {
        "data_hdf5_file": os.path.join(data_dir, "data_vocal.hdf5"),
        "model_file": os.path.join(model_dir, "model_vocal.h5"),
        "checkpoint_file": os.path.join(model_dir, "model_vocal.ckpt"),
    },
}

chunk_size = 44032
sample_rate = 44100

n_frames = 96
stft_nfft = 1948
conv_kernel_crop = 2

batch_size = 64
