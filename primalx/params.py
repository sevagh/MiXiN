import os
import json


mypath = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(mypath, "../data")
model_dir = os.path.join(mypath, "../model")
checkpoint_dir = os.path.join(mypath, "../logdir")
data_hdf5_file = os.path.join(data_dir, "data.hdf5")

model_file = os.path.join(model_dir, "model.h5")
checkpoint_file = os.path.join(checkpoint_dir, "model.ckpt")

with open(os.path.join(mypath, "../params.json")) as f:
    params = json.load(f)

    nn_time_win = params["stft_window_size"]
    chunk_size = params["chunk_size"]
    sample_rate = params["sample_rate"]

    # nfft = nfft/2 + 1
    stft_nfft = int(nn_time_win + 1)
    n_frames = int(chunk_size / (int(0.5 * nn_time_win)) + 1)
