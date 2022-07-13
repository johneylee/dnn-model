"""
example 
python enhance_speech.py job/model_irm irm XXXX.wav
"""
import os
import sys
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import ssp_library as ssp
import config as cfg
import senet_decode_library as senet
from tools_for_network import *

def main():
    if len(sys.argv) != 4:
        print('Invalid input arguments')
        print('python enhance_speech.py [trained model name] [mode] [address input wav]')
        exit()

    # Initialize setup.
    model_name, mode, addr_input = sys.argv[1:]
    if os.path.isfile(addr_input) is False:
        print('[Error] There is no input file.')
        exit()

    shps = ssp.initialize_params()

    # Define tensorflow session.
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % cfg.gpu_id
    options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_frac)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=options))

    # Create and load enhancement network.
    net = senet.SpeechEnhancementNetwork()
    net.import_model(sess, model_name)

    # Perform enhancement.
    perform_enhancement(sess, addr_input, model_name, mode, net, shps)

def perform_enhancement(sess, addr_in, model_name, mode, network, shps):

    addr_out = addr_in.replace('.wav', '_' + model_name.split('/')[-1] + '.wav')
    print('%s' % addr_in)
    print(' -> %s' % addr_out)

    fs, input_wav = wav.read(addr_in)
    noisy_wav = input_wav.astype(np.float32) / (2 ** 15)
    length = len(noisy_wav)
    if mode == 'direct':
        logmag_noisy, pha_noisy \
            = ssp.wav2logmagpha_half(noisy_wav, shps.fs, shps.window_analysis,
                                     shps.frm_size, shps.ratio)
        enhanced_logmag = network.run_decode_step(sess, logmag_noisy)
    elif mode in ['ibm', 'irm']:
        mag_noisy, pha_noisy \
            = ssp.wav2magpha_half(noisy_wav, shps.fs, shps.window_analysis,
                                     shps.frm_size, shps.ratio)
        logmag_noisy = np.log(mag_noisy + 1e-7)
        mask = network.run_decode_step(sess, logmag_noisy, mode)
        enhanced_logmag = np.log(mask * mag_noisy + 1e-7)
    enhanced = ssp.logmagpha_half2wav(enhanced_logmag, pha_noisy,
                                      shps.window_analysis, 
                                      shps.window_synthesis, 
                                      length, shps.ratio)
    wav.write(addr_out, shps.fs, enhanced) 

if __name__ == '__main__':
    main()