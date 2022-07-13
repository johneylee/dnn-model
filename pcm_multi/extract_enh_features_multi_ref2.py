#!/usr/bin/python

"""
Extract feature from database and save feature.
System arguments:
    'filename [feature type] [database address] [save address]'
"""
import os
import sys
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import ssplib as ssp
import librosa
import pdb

def main():
    #if len(sys.argv) != 6:
    #    print('Invalid input arguments')
    #    print('python extract_features.py [type_name] [type_name] [dir_name] [dir_name] [fea_name]')
    #    exit()

    featype_input = 'logmag_noisy+pha_noisy+bpd_noisy'
    featype_refer = 'sig_max+frm_rect_norm_clean+mag_norm_warp_clean+mag_norm_warp_noise+cos_xn+cos_xy+sin_xy'
    mode = 'train'
    wav_dir_name = '../data/multi_tar2/12/' + mode
    fea_dir_name = 'features_12'
    fea_name = mode # devel or train


    # Initialize setup.
    #featype_input, featype_refer, wav_dir_name, fea_dir_name, fea_name = sys.argv[1:]
    shps = ssp.initialize_params(ratio=0.25)

    # Grammar for features
    featype_input_list = featype_input.split('+')
    featype_refer_list = featype_refer.split('+')
    featype_input_list
    featype_refer_list

    # Save address of clean and noisy wave data.

    # addr_list = scan_directory('%s/noisy' % wav_dir_name)  # original
    addr_list = scan_directory_multi_stereo('%s/noisy' % wav_dir_name) # multi case

    # Save feature data.
    if os.path.isdir(fea_dir_name) is False:
        os.system('mkdir ' + fea_dir_name)

    # Get feature data from address of database.
    num_utts = len(addr_list)
    for idx_utt in range(num_utts):
        addr_tfrecord \
            = "%s/%s_%05d.tfrecord" % (fea_dir_name, fea_name, idx_utt)
        addr_noisy = addr_list[idx_utt][0]
        addr_clean = addr_list[idx_utt][1]
        inp, ref = extract_features_multi_upNdown_target2(addr_noisy, addr_clean,
                                    featype_input_list,
                                    featype_refer_list, shps)
        write_tfrecord_writer(inp, ref, addr_tfrecord)
        print("%s -> %s" % (addr_noisy, addr_tfrecord))

def scan_directory_multi_stereo(dir_name):
    """Scan directory and save address of clean/noisy wav data.

    Args:
        dir_name: directory name to scan

    Returns:
        addr_list: all address list of clean/noisy wave data in subdirectory
    """
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()
    else:
        print("Scanning a directory %s " % dir_name)

    addr_list = []
    for path, dirs, files in os.walk(dir_name):
        for data in files:
            ext = os.path.splitext(data)[-1]
            if ext == '.wav':
                addr_noisy = path + os.sep + data
                #dump = path.split('/')
                #clean_path = dumpi[0] + '/' + dump[1] + '/' + dump[2] + '/' + dump[3] + '/' + 'clean'
                addr_clean = addr_noisy.replace('noisy', 'clean')
                addr_list.append([addr_noisy, addr_clean])
                #pdb.set_trace()
    return addr_list

def write_tfrecord_writer(inp, ref, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)

    # Feature contains a map of string to feature proto objects
    feature = {
        'inp': tf.train.Feature(
            float_list=tf.train.FloatList(value=inp.flatten())),
        'ref': tf.train.Feature(
            float_list=tf.train.FloatList(value=ref.flatten()))}

    # Construct the Example proto object
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    # Serialize the example to a string
    serialized = example.SerializeToString()

    # write the serialized object to the disk
    writer.write(serialized)
    writer.close()


def extract_features_multi_upNdown_target2(addr_noisy, addr_clean,
                     featype_input_list, featype_refer_list, shps):
    """Extract features

    Args:

    Returns:
        data: output features
    """
    # Read noisy and clean wavefile
    fs, audio = wav.read(addr_noisy)
    mic0_audio = audio[:,0]
    mic1_audio = audio[:,1]
    mic0_noisy = mic0_audio.astype(np.float32) / (2 ** 15)
    mic1_noisy = mic1_audio.astype(np.float32) / (2 ** 15)

    fs, audio_c = wav.read(addr_clean)
    mic0_clean_c = audio_c[:,0]
    mic1_clean_c = audio_c[:,1]
    mic0_clean = mic0_clean_c.astype(np.float32) / (2 ** 15)
    mic1_clean = mic1_clean_c.astype(np.float32) / (2 ** 15)

    # noisy and clean
    mic0_noisy = mic0_noisy[:len(mic0_clean)]
    mic1_noisy = mic1_noisy[:len(mic1_clean)]
    mic0_clean = mic0_clean[:len(mic0_noisy)]
    mic1_clean = mic1_clean[:len(mic1_noisy)]

    # noise mic0, mic1
    wav_noise_mic0 = mic0_noisy - mic0_clean
    wav_noise_mic1 = mic1_noisy - mic1_clean

    # normalization
    vma_0 = np.max(np.abs(mic0_clean))
    vma_1 = np.max(np.abs(mic1_clean))
    wav_norm_clean_0 = mic0_clean / vma_0
    wav_norm_clean_1 = mic1_clean / vma_1
    wav_norm_noise_mic0 = wav_noise_mic0 / vma_0
    wav_norm_noise_mic1 = wav_noise_mic1 / vma_1

    # feature input shape
    logmag_noisy_mic0, pha_noisy_mic0 \
        = ssp.wav2logmagpha_half(mic0_noisy, fs, shps.window_analysis,
                                 shps.frm_size, shps.ratio)
    logmag_noisy_mic1, pha_noisy_mic1 \
        = ssp.wav2logmagpha_half(mic1_noisy, fs, shps.window_analysis,
                                 shps.frm_size, shps.ratio)
    logmag_noisy = np.concatenate((logmag_noisy_mic0, logmag_noisy_mic1), axis=1)
    pha_noisy_concat = np.concatenate((pha_noisy_mic0, pha_noisy_mic1), axis=1)

    # extract feature
    # input
    inp = np.zeros([logmag_noisy.shape[0], 0])
    for feature_type in featype_input_list:
        if feature_type == 'logmag_noisy':
            inp = np.concatenate([inp, logmag_noisy], axis=-1)
        elif feature_type == 'pha_noisy':
            inp = np.concatenate([inp, pha_noisy_concat], axis=-1)
        elif feature_type == 'bpd_noisy':
            bpd_noisy_mic0 = ssp.pha2bpd(pha_noisy_mic0, fs, shps.frm_size, shps.ratio)
            bpd_noisy_mic1 = ssp.pha2bpd(pha_noisy_mic1, fs, shps.frm_size, shps.ratio)
            bpd_noisy = np.concatenate((bpd_noisy_mic0[:,:128], bpd_noisy_mic1[:,:128]), axis=1)
            inp = np.concatenate([inp, bpd_noisy], axis=-1)

    # target
    ref = np.zeros([inp.shape[0], 0])
    for feature_type in featype_refer_list:

        # TDR target

        if feature_type == 'frm_rect_norm_clean':
            frm_rect_norm_clean_0 = ssp.enframe(wav_norm_clean_0, fs,
                                              np.ones(512),
                                              shps.frm_size, shps.ratio)
            frm_rect_norm_clean_0 = frm_rect_norm_clean_0[:, :128]
            frm_rect_norm_clean_1 = ssp.enframe(wav_norm_clean_1, fs,
                                              np.ones(512),
                                              shps.frm_size, shps.ratio)
            frm_rect_norm_clean_1 = frm_rect_norm_clean_1[:, :128]
            frm_rect_norm_clean = np.concatenate((frm_rect_norm_clean_0, frm_rect_norm_clean_1), axis=1)
            ref = np.concatenate([ref, frm_rect_norm_clean], axis=-1)

        elif feature_type == 'sig_max':
            num_frms = logmag_noisy.shape[0]
            sig_max = np.ones([num_frms, 1])
            sig_max[:] = vma_0
            ref = np.concatenate([ref, sig_max], axis=-1)

        # Freq-domain clean

        elif feature_type == 'mag_norm_warp_clean':
            mag_norm_clean_0, pha_clean \
                = ssp.wav2magpha_half(wav_norm_clean_0, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_norm_clean_1, pha_clean \
                = ssp.wav2magpha_half(wav_norm_clean_1, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_norm_clean = np.concatenate((mag_norm_clean_0, mag_norm_clean_1), axis=1)

            mag_norm_warp_clean = np.power(mag_norm_clean, 2.0 / 3.0)

            ref = np.concatenate([ref, mag_norm_warp_clean], axis=-1)

        # Freq-domain noise
        elif feature_type == 'mag_norm_warp_noise':
            mag_norm_noise0, pha_noise0 \
                = ssp.wav2magpha_half(wav_norm_noise_mic0, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_norm_noise1, pha_noise1 \
                = ssp.wav2magpha_half(wav_norm_noise_mic1, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_norm_noise = np.concatenate((mag_norm_noise0, mag_norm_noise1), axis=1)
            mag_norm_warp_noise = np.power(mag_norm_noise, 2.0 / 3.0)
            ref = np.concatenate([ref, mag_norm_warp_noise], axis=-1)

        # Phase
        elif feature_type == 'cos_xn':
            mag_clean, pha_clean_0 \
                = ssp.wav2magpha_half(mic0_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_clean, pha_clean_1 \
                = ssp.wav2magpha_half(mic1_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            pha_clean = np.concatenate((pha_clean_0, pha_clean_1), axis=1)
            mag_noise0, pha_noise0 = \
                ssp.wav2magpha_half(wav_noise_mic0, fs, shps.window_analysis,
                                    shps.frm_size, shps.ratio)
            mag_noise1, pha_noise1 = \
                ssp.wav2magpha_half(wav_noise_mic1, fs, shps.window_analysis,
                                    shps.frm_size, shps.ratio)
            pha_noise = np.concatenate((pha_noise0, pha_noise1), axis=1)

            cos_xn = np.cos(pha_clean - pha_noise)
            ref = np.concatenate([ref, cos_xn], axis=-1)

        elif feature_type == 'cos_xy':
            mag_clean, pha_clean_0 \
                = ssp.wav2magpha_half(mic0_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_clean, pha_clean_1 \
                = ssp.wav2magpha_half(mic1_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            pha_clean = np.concatenate((pha_clean_0, pha_clean_1), axis=1)
            mag_noisy0, pha_noisy0 \
                = ssp.wav2magpha_half(mic0_noisy, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_noisy1, pha_noisy1 \
                = ssp.wav2magpha_half(mic1_noisy, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            pha_noisy = np.concatenate((pha_noisy0, pha_noisy1), axis=1)
            pha_xy = pha_clean - pha_noisy
            cos_xy = np.cos(pha_xy)
            ref = np.concatenate([ref, cos_xy], axis=-1)

        elif feature_type == 'sin_xy':
            mag_clean, pha_clean_0 \
                = ssp.wav2magpha_half(mic0_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_clean, pha_clean_1 \
                = ssp.wav2magpha_half(mic1_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            pha_clean = np.concatenate((pha_clean_0, pha_clean_1), axis=1)
            mag_noisy0, pha_noisy0 \
                = ssp.wav2magpha_half(mic0_noisy, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_noisy1, pha_noisy1 \
                = ssp.wav2magpha_half(mic1_noisy, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            pha_noisy = np.concatenate((pha_noisy0, pha_noisy1), axis=1)
            pha_xy = pha_clean - pha_noisy
            sin_xy = np.sin(pha_xy)
            ref = np.concatenate([ref, sin_xy], axis=-1)

    return inp, ref

if __name__ == '__main__':
    main()
