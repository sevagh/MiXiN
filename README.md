# MiXiN

MiXiN, or **M**usic **X**traction with **N**onstationary Gabor Transforms, is a model for harmonic/percussive/vocal source separation based on [Convolutional Denoising Autoencoders](https://arxiv.org/abs/1703.08019). The pretrained models are trained on Periphery stems from the albums Juggernaut, Omega, Periphery III and Hail Stan (available for purchase [here](https://store.periphery.net/music/music)).

MiXiN takes the simple [median-filtering HPSS](http://dafx10.iem.at/papers/DerryFitzGerald_DAFx10_P15.pdf) algorithm (which applies soft masks computed from harmonic and percussive magnitude estimates), and replaces the simple (but not so impressive) median filtering estimation step with trained CDAEs.

## Demo clips

This demo page (sevag.xyz) contains some sound samples generated with the available pretrained models.

## Install, use, test, and train your own

## Architecture

MiXiN uses 3 CDAEs, each trained on separate harmonic, vocal, and percussive components.

From the paper on [CDAEs for source separation](https://arxiv.org/abs/1703.08019):
![!cdae](./.github/cdae_arch.png)

The CDAE paper uses an STFT of 15 frames with a window size of 1024 (1025 FFT points, representing 2048/2 + 1 non-redundant spectral coefficients) to perform source separation, by training a separate CDAE for each source. One CDAE is recommended for each desired source. In MiXiN, the sources considered are harmonic/percussive/vocal - this is influenced by the [median-filtering HPSS](http://dafx10.iem.at/papers/DerryFitzGerald_DAFx10_P15.pdf) and [HPSS vocal separation with CQT](https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1007&context=argart) papers. This gives us an architecture with 3 CDAEs.

In MiXiN, the Nonstationary Gabor Transform is used with 96 bins per ocatve and the Bark frequency scale from 0-22050 Hz, available in [my fork of the Python nsgt implementation](https://github.com/sevagh/nsgt). Think of the NSGT like an STFT with some useful time-frequency properties that might make it more amenable to musical applications - for further reading more musical time-frequency analyses, check out [Judith Brown's first paper on the CQT](https://www.ee.columbia.edu/~dpwe/papers/Brown91-cqt.pdf), [Monika Doerfler's dissertation](http://www.mathe.tu-freiberg.de/files/thesis/gamu_1.pdf) for a treatment of multiple Gabor dictionaries, and [paper 1](https://ltfat.github.io/notes/ltfatnote010.pdf), [paper 2](https://ltfat.github.io/notes/ltfatnote018.pdf) on NSGTs.

The signal is split into chunk sizes of 44032 samples (representing roughly 1s of audio, divisible by 1024), and NSGT is taken. The magnitude of the NSGT coefficients the input to all of the 3 CDAEs, and the outputs are the estimates of vocal, percussive, and harmonic magnitude NSGT coefficients.

The original CDAE paper uses the phase of the mixture and the magnitude of the CDAE output to invert and create the separated source. The approach in MiXiN is closer to that of the HPSS formulation, where the magnitude estimates of each source (harmonic, percussive, vocal) are used to compute soft masks using the Wiener filter formula:

<img src="./.github/mixin_arch.png" width="640px">

## Training

As mentioned, the training data used was 4 albums from Periphery, prepared in two sets:
* Instrumental mix, harmonic (rhythm guitar + lead guitar + bass + other stems), and percussive (drum stem)
* Full mix, harmonic (rhythm guitar + lead guitar + bass + other stems), and percussive (drum stem), vocal (vocal stem)

A consequence is that the vocal CDAE is trained on half the data of the percussive and harmonic ones.

The data is split into 80%/20%/20% train/validation/test. There are 3 models trained, with 37,000 parameters each. Ideas for the CDAE implementation was relatively clear in the paper, and helped by [this implementation](https://github.com/SahilJindal1/Sound-Separation). Here are the training plots for the 3 networks - the loss is mae.

Percussive:

<img src="./.github/percussive_train_loss.png" width=512px>

Harmonic:

<img src="./.github/harmonic_train_loss.png" width=512px>

Vocal:

<img src="./.github/vocal_train_loss.png" width=512px>

## Evaluation

A small evaluation was performed on some tracks from the [MUSDB18-HQ](https://zenodo.org/record/3338373) test set, using the testbench from my larger [Music-Separation-TF](https://gitlab.com/sevagh/Music-Separation-TF) project (survey of various DSP approaches to source separation).

The metric is SigSep's [BSSv4](https://github.com/sigsep/bsseval), and MiXiN is compared against [Open-Unmix](https://sigsep.github.io/open-unmix/):

![bssv4](./.github/bssv4_results_musdb18hq.png)

MiXiN scores poorly, but to me it still sounds pretty good - I like the percussive and vocal outputs more than harmonic.

