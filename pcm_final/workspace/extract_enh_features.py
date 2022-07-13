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

    if len(sys.argv) != 6:
        print('Invalid input arguments')
        print('python extract_features.py [type_name] [type_name] [dir_name] [dir_name] [fea_name]')
        exit()

    # Initialize setup.
    featype_input, featype_refer, wav_dir_name, fea_dir_name, fea_name = sys.argv[1:]
    shps = ssp.initialize_params(ratio=0.25)

    # Grammar for features
    featype_input_list = featype_input.split('+')
    featype_refer_list = featype_refer.split('+')
    featype_input_list
    featype_refer_list

    # Save address of clean and noisy wave data.
    addr_list = scan_directory('%s/noisy' % wav_dir_name)

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
        inp, ref = extract_features(addr_noisy, addr_clean,
                                    featype_input_list,
                                    featype_refer_list, shps)
        write_tfrecord_writer(inp, ref, addr_tfrecord)
        print("%s -> %s" % (addr_noisy, addr_tfrecord))

def scan_directory(dir_name):
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
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".wav"):
                addr_noisy = filepath
                addr_clean = filepath.replace('/noisy/', '/clean/')
                addr_list.append([addr_noisy, addr_clean])
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


def extract_features(addr_noisy, addr_clean,
                     featype_input_list, featype_refer_list, shps):
    """Extract features

    Args:

    Returns:
        data: output features
    """
    # Read noisy and clean wavefile
    print(addr_noisy)

    fs, audio = wav.read(addr_noisy)
    wav_noisy = audio.astype(np.float32) / (2 ** 15)
    if len(wav_noisy.shape) == 2:
        wav_noisy = (wav_noisy[:, 0] + wav_noisy[:, 1]) * 0.5
    fs, audio = wav.read(addr_clean)
    wav_clean = audio.astype(np.float32) / (2 ** 15)
    if len(wav_clean.shape) == 2:
        wav_clean = (wav_clean[:, 0] + wav_clean[:, 1]) * 0.5
    wav_noise = wav_noisy - wav_clean

    vma = np.max(np.abs(wav_clean))
    wav_norm_clean = wav_clean / vma
    wav_norm_noise = wav_noise / vma

    logmag_noisy, pha_noisy \
        = ssp.wav2logmagpha_half(wav_noisy, fs, shps.window_analysis,
                                 shps.frm_size, shps.ratio)

    inp = np.zeros([logmag_noisy.shape[0], 0])
    for feature_type in featype_input_list:
        if feature_type == 'logmag_noisy':
            inp = np.concatenate([inp, logmag_noisy], axis=-1)
        elif feature_type == 'pha_noisy':
            inp = np.concatenate([inp, pha_noisy], axis=-1)
        elif feature_type == 'bpd_noisy':
            bpd_noisy = ssp.pha2bpd(pha_noisy, fs, shps.frm_size, shps.ratio)
            inp = np.concatenate([inp, bpd_noisy[:, :128]], axis=-1)

    ref = np.zeros([inp.shape[0], 0])
    for feature_type in featype_refer_list:
        # Time-domain clean
        if feature_type == 'frm_hann_clean':
            frm_hann_clean = ssp.enframe(wav_clean, fs, shps.window_analysis,
                                         shps.frm_size, shps.ratio)
            ref = np.concatenate([ref, frm_hann_clean], axis=-1)

        elif feature_type == 'frm_hann_norm_clean':
            frm_hann_norm_clean = ssp.enframe(wav_norm_clean, fs,
                                              shps.window_analysis,
                                              shps.frm_size, shps.ratio)
            ref = np.concatenate([ref, frm_hann_norm_clean], axis=-1)

        elif feature_type == 'frm_rect_clean':
            frm_rect_clean = ssp.enframe(wav_clean, fs, np.ones(512),
                                         shps.frm_size, shps.ratio)
            frm_rect_clean = frm_rect_clean[:, :128]
            ref = np.concatenate([ref, frm_rect_clean], axis=-1)

        elif feature_type == 'frm_rect_norm_clean':
            frm_rect_norm_clean = ssp.enframe(wav_norm_clean, fs,
                                              np.ones(512),
                                              shps.frm_size, shps.ratio)
            frm_rect_norm_clean = frm_rect_norm_clean[:, :128]
            ref = np.concatenate([ref, frm_rect_norm_clean], axis=-1)

        elif feature_type == 'frm_rect_mulaw_clean':
            mu = 15.0
            frm_rect_norm_clean = ssp.enframe(wav_norm_clean, fs,
                                              np.ones(512),
                                              shps.frm_size, shps.ratio)
            frm_rect_norm_clean = frm_rect_norm_clean[:, :128]
            frm_rect_mulaw_clean = ssp.mulaw(frm_rect_norm_clean, mu)
            ref = np.concatenate([ref, frm_rect_mulaw_clean], axis=-1)

        elif feature_type == 'sig_max':
            num_frms = logmag_noisy.shape[0]
            sig_max = np.ones([num_frms, 1])
            sig_max[:] = vma
            ref = np.concatenate([ref, sig_max], axis=-1)

        # Freq-domain clean
        elif feature_type == 'logmag_clean':
            logmag_clean, pha_clean \
                = ssp.wav2logmagpha_half(wav_clean, fs, shps.window_analysis,
                                         shps.frm_size, shps.ratio)
            ref = np.concatenate([ref, logmag_clean], axis=-1)

        elif feature_type == 'mag_clean':
            mag_clean, pha_clean \
                = ssp.wav2magpha_half(wav_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            ref = np.concatenate([ref, mag_clean], axis=-1)

        elif feature_type == 'mag_norm_clean':
            mag_norm_clean, pha_clean \
                = ssp.wav2magpha_half(wav_norm_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            ref = np.concatenate([ref, mag_norm_clean], axis=-1)

        elif feature_type == 'mag_warp_clean':
            mag_clean, pha_clean \
                = ssp.wav2magpha_half(wav_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_warp_clean = np.power(mag_clean, 2.0 / 3.0)
            ref = np.concatenate([ref, mag_warp_clean], axis=-1)
        
        elif feature_type == 'mag_norm_warp_clean':
            mag_norm_clean, pha_clean \
                = ssp.wav2magpha_half(wav_norm_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_norm_warp_clean = np.power(mag_norm_clean, 2.0 / 3.0)
            ref = np.concatenate([ref, mag_norm_warp_clean], axis=-1)

        elif feature_type == 'melmag40_norm_warp_clean':
            tri_mel_fbank120 = librosa.filters.mel(fs, 512, n_mels=40).transpose()
            mag_norm_clean, pha_clean \
                = ssp.wav2magpha_half(wav_norm_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            melmag_norm_clean = np.matmul(mag_norm_clean, tri_mel_fbank120)
            melmag_norm_warp_clean = np.power(melmag_norm_clean, 2.0 / 3.0)
            ref = np.concatenate([ref, melmag_norm_warp_clean], axis=-1)

        elif feature_type == 'spec_clean':
            real_clean, imag_clean \
                = ssp.wav2realimag_half(wav_clean, fs, shps.window_analysis,
                                        shps.frm_size, shps.ratio)
            spec_clean = np.concatenate([real_clean, imag_clean], 1)
            ref = np.concatenate([ref, spec_clean], axis=-1)

        # Freq-domain noise
        elif feature_type == 'logmag_noise':
            logmag_noise, pha_noise \
                = ssp.wav2logmagpha_half(wav_noise, fs, shps.window_analysis,
                                         shps.frm_size, shps.ratio)
            ref = np.concatenate([ref, logmag_noise], axis=-1)

        elif feature_type == 'mag_noise':
            mag_noise, pha_noise \
                = ssp.wav2magpha_half(wav_noise, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            ref = np.concatenate([ref, mag_noise], axis=-1)

        elif feature_type == 'mag_norm_noise':
            mag_norm_noise, pha_noise \
                = ssp.wav2magpha_half(wav_norm_noise, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            ref = np.concatenate([ref, mag_norm_noise], axis=-1)

        elif feature_type == 'mag_warp_noise':
            mag_noise, pha_noise \
                = ssp.wav2magpha_half(wav_noise, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_warp_noise = np.power(mag_noise, 2.0 / 3.0)
            ref = np.concatenate([ref, mag_warp_noise], axis=-1)

        elif feature_type == 'mag_norm_warp_noise':
            mag_norm_noise, pha_noise \
                = ssp.wav2magpha_half(wav_norm_noise, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_norm_warp_noise = np.power(mag_norm_noise, 2.0 / 3.0)
            ref = np.concatenate([ref, mag_norm_warp_noise], axis=-1)

        elif feature_type == 'melmag40_norm_warp_noise':
            tri_mel_fbank120 = librosa.filters.mel(fs, 512,
                                                   n_mels=40).transpose()
            mag_norm_noise, pha_noise \
                = ssp.wav2magpha_half(wav_norm_noise, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            melmag_norm_noise = np.matmul(mag_norm_noise, tri_mel_fbank120)
            melmag_norm_warp_noise = np.power(melmag_norm_noise, 2.0 / 3.0)
            ref = np.concatenate([ref, melmag_norm_warp_noise], axis=-1)

        # T-F mask
        elif feature_type == 'irm':
            mag_clean, pha_clean \
                = ssp.wav2magpha_half(wav_clean, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            mag_noise, pha_noise \
                = ssp.wav2magpha_half(wav_noise, fs, shps.window_analysis,
                                      shps.frm_size, shps.ratio)
            irmw = (mag_clean ** 2) / ((mag_clean ** 2) + (mag_noise ** 2))
            irmp = np.sqrt(irmw)
            ref = np.concatenate([ref, irmp], axis=-1)

        elif feature_type == 'cirm_real':
            real_clean, imag_clean \
                = ssp.wav2realimag_half(wav_clean, fs, shps.window_analysis,
                                        shps.frm_size, shps.ratio)
            real_noisy, imag_noisy \
                = ssp.wav2realimag_half(wav_noisy, fs, shps.window_analysis,
                                        shps.frm_size, shps.ratio)
            cirm_real_tmp = real_noisy * real_clean + imag_noisy * imag_clean
            denorm = real_noisy * real_noisy + imag_noisy * imag_noisy
            cirm_real = cirm_real_tmp / denorm
            cirm_real = np.maximum(np.minimum(cirm_real, 20.0), -20.0)
            ref = np.concatenate([ref, cirm_real], axis=-1)

        elif feature_type == 'cirm_imag':
            real_clean, imag_clean \
                = ssp.wav2realimag_half(wav_clean, fs, shps.window_analysis,
                                        shps.frm_size, shps.ratio)
            real_noisy, imag_noisy \
                = ssp.wav2realimag_half(wav_noisy, fs, shps.window_analysis,
                                        shps.frm_size, shps.ratio)
            cirm_imag_tmp = real_noisy * imag_clean - imag_noisy * real_clean
            denorm = real_noisy * real_noisy + imag_noisy * imag_noisy
            cirm_imag = cirm_imag_tmp / denorm
            cirm_imag = np.maximum(np.minimum(cirm_imag, 20.0), -20.0)
            ref = np.concatenate([ref, cirm_imag], axis=-1)

        # Phase
        elif feature_type == 'cos_xn':
            if pha_clean is None:
                mag_clean, pha_clean \
                    = ssp.wav2magpha_half(wav_clean, fs, shps.window_analysis,
                                          shps.frm_size, shps.ratio)
            if pha_noise is None:
                mag_noise, pha_noise \
                    = ssp.wav2magpha_half(wav_noise, fs, shps.window_analysis,
                                          shps.frm_size, shps.ratio)
            cos_xn = np.cos(pha_clean - pha_noise)
            ref = np.concatenate([ref, cos_xn], axis=-1)

        elif feature_type == 'pha_xy':
            if pha_clean is None:
                mag_clean, pha_clean \
                    = ssp.wav2magpha_half(wav_clean, fs, shps.window_analysis,
                                          shps.frm_size, shps.ratio)
            if pha_noisy is None:
                mag_noisy, pha_noisy \
                    = ssp.wav2magpha_half(wav_noisy, fs, shps.window_analysis,
                                          shps.frm_size, shps.ratio)
            pha_xy = pha_clean - pha_noisy
            ref = np.concatenate([ref, pha_xy], axis=-1)

        elif feature_type == 'cos_xy':
            if pha_clean is None:
                mag_clean, pha_clean \
                    = ssp.wav2magpha_half(wav_clean, fs, shps.window_analysis,
                                          shps.frm_size, shps.ratio)
            if pha_noisy is None:
                mag_noisy, pha_noisy \
                    = ssp.wav2magpha_half(wav_noisy, fs, shps.window_analysis,
                                          shps.frm_size, shps.ratio)
            pha_xy = pha_clean - pha_noisy
            cos_xy = np.cos(pha_xy)
            ref = np.concatenate([ref, cos_xy], axis=-1)

        elif feature_type == 'sin_xy':
            if pha_clean is None:
                mag_clean, pha_clean \
                    = ssp.wav2magpha_half(wav_clean, fs, shps.window_analysis,
                                          shps.frm_size, shps.ratio)
            if pha_noisy is None:
                mag_noisy, pha_noisy \
                    = ssp.wav2magpha_half(wav_noisy, fs, shps.window_analysis,
                                          shps.frm_size, shps.ratio)
            pha_xy = pha_clean - pha_noisy
            sin_xy = np.sin(pha_xy)
            ref = np.concatenate([ref, sin_xy], axis=-1)

    return inp, ref


if __name__ == '__main__':
    main()
