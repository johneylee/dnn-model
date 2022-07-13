
"""
Enhance noisy speech from trained network.
System arguments:
    'filename [trained model path] [noisy dir] [mode name]'
"""
import os
import sys
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
#import jknet_decode as senet
import se_decode as senet
import ssplib as ssp
from os import makedirs


def main():

    if len(sys.argv) != 4:
        print('Invalid input arguments')
        exit()

    # Initialize setup.
    model_name, noisy_dir, mode = sys.argv[1:]

    # Initialize parameters
    shps = ssp.initialize_params(flag=1, ratio=0.25)

    # Initialize Tensorflow session.
    options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=options))

    # Create and load enhancement network.
    net = senet.SpeechEnhancementNetwork()
    model_str = 'models/%s' % model_name
    net.import_model(sess, model_str)

    noisy_folder_name = noisy_dir.split('/')[3]
    # Create the 'enhanced path' from the noisy_dir
    for (path, dir, files) in os.walk(noisy_dir):
        for data in files:
            ext = os.path.splitext(data)[-1]
            if ext == '.wav':
                addr_input = "%s/%s" % (path, data)
                addr_new = path.replace(noisy_folder_name, mode)
                #addr_new = path.replace('data_small', mode)
                makedirs(addr_new, exist_ok=True)

            if os.path.isfile(addr_input) is False:
                print('[Error] There is no input file.')
                exit()

            # Perform enhancement.
            buf = []
            len_wav = []
            addr_noisy = addr_input
            fs, audio = wav.read(addr_noisy)
            wav_noisy = audio.astype(np.float32) / (2 ** 15)
            len_wav.append(len(audio))
            logmag_noisy, pha_noisy \
                = ssp.wav2logmagpha_half(wav_noisy, shps.fs,
                                         shps.window_analysis,
                                         shps.frm_size, shps.ratio)
            bpd_noisy = ssp.pha2bpd(pha_noisy, shps.fs, shps.frm_size, shps.ratio)
            bpd_noisy = bpd_noisy[:, :128]
            tmpin = np.concatenate([logmag_noisy, pha_noisy, bpd_noisy], axis=1)
            buf.append(np.float32(tmpin))

            enhanced = net.run_decode_batch_step(sess, buf,
                                                 shps.window_analysis,
                                                 shps.window_synthesis,
                                                 len_wav, shps.ratio)

            addr_out = addr_input.replace('.wav', '_' + mode.split('/')[-1] + '.wav')
            enhanced_out = addr_out.replace(noisy_folder_name, mode)
            #enhanced_out = addr_out.replace('data_small', mode)
            print('%s' % addr_input)
            print(' -> %s' % enhanced_out)
            wav.write(enhanced_out, 16000, enhanced[0])

if __name__ == '__main__':
    main()
