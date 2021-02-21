import os


mypath = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(mypath, "../data")
model_dir = os.path.join(mypath, "../model")
checkpoint_dir = os.path.join(mypath, "../logdir")
data_hdf5_file = os.path.join(data_dir, "data.hdf5")

model_file = os.path.join(model_dir, "model.h5")
checkpoint_file = os.path.join(checkpoint_dir, "model.ckpt")

nn_time_win = 1024
chunk_size = 44032
sample_rate = 44100

n_frames = 96
stft_nfft = 1948