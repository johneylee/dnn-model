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
import jknet_decode as senet
import ssplib as ssp
import time

if len(sys.argv) != 5:
    print('Invalid input arguments')
    exit()

# Initialize setup.
ratio, batchsize, model_name, set_name = sys.argv[1:]

if os.path.isdir('outputs') is False:
    os.system('mkdir outputs')

ratio = float(ratio)
batchsize = int(batchsize)
addr = []
tmp = []
cnt = 0
dir_target = './data_wav/' + set_name
dir_dest = 'outputs/%s/%s' % (set_name, model_name)
os.system('mkdir outputs/%s' % set_name)
os.system('mkdir outputs/%s/%s' % (set_name, model_name))
for subdir, dirs, files in os.walk('%s/noisy' % dir_target):
    subdir_dest = subdir.replace(dir_target, dir_dest)
    print(subdir_dest)
    if os.path.isdir(subdir_dest) is False:
        os.system('mkdir %s' % subdir_dest)
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".wav"):
            cnt = cnt + 1
            tmp.append(filepath)
            if cnt == batchsize:
                cnt = 0
                addr.append(tmp)
                tmp = []
if cnt != 0:
    addr.append(tmp)
numbatches = len(addr)

if len(addr[-1]) == batchsize:
    print('decoding schedule: %d * %d' % (batchsize, numbatches))
else:
    print('decoding schedule: %d * %d + %d' % (batchsize, numbatches - 1, len(addr[-1])))

#
shps = ssp.initialize_params(flag=1, ratio=ratio)

# Initialize Tensorflow session.
options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=options))

# Create and load enhancement network.
net = senet.SpeechEnhancementNetwork()
model_str = 'models/%s' % model_name
net.import_model(sess, model_str)


# Perform enhancement.
for i in range(numbatches):
    t = time.time()
    buf = []
    len_wav = []
    num_samps = len(addr[i])
    for j in range(num_samps):
        addr_noisy = addr[i][j]
        fs, audio = wav.read(addr_noisy)
        wav_noisy = audio.astype(np.float32) / (2 ** 15)
        len_wav.append(len(audio))
        logmag_noisy, pha_noisy \
            = ssp.wav2logmagpha_half(wav_noisy, shps.fs,
                                     shps.window_analysis,
                                     shps.frm_size, shps.ratio)
        tmpin = np.concatenate([logmag_noisy, pha_noisy], axis=1)
        buf.append(np.float32(tmpin))
    enhanced = net.run_decode_batch_step(sess, buf,
                                         shps.window_analysis,
                                         shps.window_synthesis,
                                         len_wav, shps.ratio)
    for j in range(num_samps):
        addr_out = addr[i][j].replace(dir_target, dir_dest)
        wav.write(addr_out, 16000, enhanced[j])
    print("[%d/%d] %s    takes %.2f sec."
          % (i + 1, numbatches, model_name, time.time() - t))
