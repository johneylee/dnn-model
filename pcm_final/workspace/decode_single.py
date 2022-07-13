#!/usr/bin/python

"""
Enhance noisy speech from trained network.
System arguments:
    'filename [trained model name] [input wave] [output wave]'
"""
import os
import sys
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import se_decode as senet
import ssplib as ssp
import time


def main():

    if len(sys.argv) != 4:
        print('Invalid input arguments')
        exit()

    # Initialize setup.
    model_name, addr_in, addr_out = sys.argv[1:]

    # Initialize parameters
    shps = ssp.initialize_params(flag=1, ratio=0.25)

    # Initialize Tensorflow session.
    options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=options))

    # Create and load enhancement network.
    net = senet.SpeechEnhancementNetwork()
    model_str = 'models/%s' % model_name
    net.import_model(sess, model_str)

    # Perform enhancement.
    buf = []
    len_wav = []
    addr_noisy = addr_in
    fs, audio = wav.read(addr_noisy)

    wav_noisy = audio.astype(np.float32) / (2 ** 15)
    sig = np.std(wav_noisy)
    wav_noisy = wav_noisy / sig * 0.0167

    len_wav.append(len(audio))
    logmag_noisy, pha_noisy \
        = ssp.wav2logmagpha_half(wav_noisy, shps.fs,
                                 shps.window_analysis,
                                 shps.frm_size, shps.ratio)
    tmpin = np.concatenate([logmag_noisy, pha_noisy], axis=1)
    buf.append(np.float16(tmpin))
    enhanced = net.run_decode_batch_step(sess, buf,
                                         shps.window_analysis,
                                         shps.window_synthesis,
                                         len_wav, shps.ratio, sig)

    wav.write(addr_out, 16000, enhanced[0])

if __name__ == '__main__':
    main()