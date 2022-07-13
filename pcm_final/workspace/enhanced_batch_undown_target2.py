#!/usr/bin/python

"""
Enhance noisy speech from trained network.
System arguments:
    'filename [trained model path] [input wave] [output wave]'
"""
import os
import sys
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import se_decode as senet
import ssplib as ssp
from os import makedirs
import pdb
import soundfile
#import time

#start_time = time.time()

def main():

    # if len(sys.argv) != 5:
    #     print('Invalid input arguments')
    #     exit()
    #
    # # Initialize setup.
    # model_name, noisy_dir, mode = sys.argv[1:]

    model_name = 'apsipa_single_mic0/opt'
    noisy_dir = '../../pcm_multi/data/multi_tar2/TEST/noisy'
    mode = 'new'
    # Initialize parameters
    shps = ssp.initialize_params(flag=1, ratio=0.25)

    # Initialize Tensorflow session.
    options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=options))

    # Create and load enhancement network.
    net = senet.SpeechEnhancementNetwork()
    model_str = 'models/%s' % model_name
    net.import_model(sess, model_str)
    #net.import_model2(sess, model_str)
 
    #Create the 'enhanced path' from the noisy_dir
    for (path, dir, files) in os.walk(noisy_dir):
        for data in files:
            ext = os.path.splitext(data)[-1]
            if ext == '.wav':
                addr_input = "%s/%s" % (path, data)
                addr_new = path.replace('noisy', 'mic1')
                makedirs(addr_new, exist_ok=True)
                
            if os.path.isfile(addr_input) is False:
                print('[Error] There is no input file.')
                exit()
            # Perform enhancement.
            buf = []
            len_wav = []
            addr_noisy = addr_input
            fs, audio = wav.read(addr_noisy)
            wav_noisy_mic0_ = audio[:,0]
            wav_noisy_mic0 = wav_noisy_mic0_.astype(np.float32) / (2 ** 15)
            wav_noisy_mic1_ = audio[:,1]
            wav_noisy_mic1 = wav_noisy_mic1_.astype(np.float32) / (2 ** 15)

            #wav_concat = np.concatenate((data_mic0, data_mic1), axis = 0)
            #wav_noisy = wav_concat.astype(np.float32) / (2 ** 15)
            len_wav.append(len(wav_noisy_mic1))
            logmag_noisy0, pha_noisy0 \
                = ssp.wav2logmagpha_half(wav_noisy_mic0, shps.fs,
                                         shps.window_analysis,
                                         shps.frm_size, shps.ratio)
            bpd_noisy0 = ssp.pha2bpd(pha_noisy0, shps.fs, shps.frm_size, shps.ratio)
            bpd_noisy0 = bpd_noisy0[:, :128]
            
            # len_wav.append(len(wav_noisy_mic1))
            logmag_noisy1, pha_noisy1 \
                = ssp.wav2logmagpha_half(wav_noisy_mic1, shps.fs,
                                         shps.window_analysis,
                                         shps.frm_size, shps.ratio)
            bpd_noisy1 = ssp.pha2bpd(pha_noisy1, shps.fs, shps.frm_size, shps.ratio)
            bpd_noisy1 = bpd_noisy1[:, :128]

            logmag_noisy = np.concatenate((logmag_noisy0, logmag_noisy1), axis = 1)
            pha_noisy = np.concatenate((pha_noisy0, pha_noisy1), axis = 1)
            bpd_noisy = np.concatenate((bpd_noisy0, bpd_noisy1), axis = 1)

            tmpin = np.concatenate([logmag_noisy, pha_noisy, bpd_noisy], axis=1)

            buf.append(np.float16(tmpin))
            enhanced = net.run_decode_batch_step(sess, buf,
                                                 shps.window_analysis,
                                                 shps.window_synthesis,
                                                 len_wav, shps.ratio)

            #addr_out = addr_input.replace('.wav', '_' + mode.split('/')[-1] + '.wav')
            addr_out = addr_input
            enhanced_out = addr_out.replace('noisy', 'mic1')
            print('%s' % addr_input)
            print(' -> %s' % enhanced_out)
            #print("--- %s seconds ---" % (time.time() - start_time))

            # soundfile.write(enhanced_out, enhanced[0][0].T, fs)
            wav.write(enhanced_out, 16000, enhanced[0])

if __name__ == '__main__':
    main()
