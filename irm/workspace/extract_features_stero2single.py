"""
example
python extract_features.py logmag_noisy ../data/speech/train feature_data/train_input
"""
import os
import pickle
import sys
import numpy as np
import scipy.io.wavfile as wav
import ssp_library as ssp
import pdb

def main():
    if len(sys.argv) != 4:
        print('Invalid input arguments')
        print('python extract_features.py [type_name] [dir_name] [output_name]')
        exit()

    # Initialize setup.
    feature_type, wav_dir_name, addr_feature = sys.argv[1:]
    shps = ssp.initialize_params(ratio=0.5)

    # Save address of clean and noisy wave data.
    addr = scan_directory_multi_stereo('%s/noisy' % wav_dir_name)

    # Get feature data from address of database.
    data = read_data_from_addr(addr, feature_type, shps)

    # Save feature data.
    if os.path.isdir('feature_data') is False:
        os.system('mkdir feature_data')
    with open(addr_feature, 'wb') as handle:
        print("Writing feature data to '%s'." % addr_feature)
        pickle.dump(np.array(data), handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Complete.\n")


def scan_directroy(dir_name):
    """Scan directory and save address of clean/noisy wav data.

    Args:
        dir_name: directroy name to scan

    Returns:
        addr: all address list of clean/noisy wave data in subdirectory
    """
    if os.path.isdir(dir_name) is False:
        print("[Error] There is no directory '%s'." % dir_name)
        exit()
    else:
        print("Scanning a directory %s " % dir_name)

    addr = []
    for subdir, dirs, files in os.walk(dir_name):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".wav"):
                addr_noisy = filepath
                addr_clean = filepath.replace('/noisy/', '/clean/')
                addr.append([addr_noisy, addr_clean])
    return addr

def scan_directory_multi_stereo(dir_name):
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
                dump = data.split('_')
                clean_name = dump[1] + '_' + dump[2] + '.wav' 
                dump_path = path.split('/')
                clean_path = dump_path[0] + '/' + dump_path[1] + '/' + dump_path[2] + '/' + dump_path[3] + '/' + dump_path[4] + '/' + dump_path[5] + '/'  + 'clean'  
                addr_clean = clean_path + os.sep + clean_name 
                addr_list.append([addr_noisy, addr_clean])
    return addr_list 

def read_data_from_addr(addr, feature_type, shps):
    data = []
    for i in range(len(addr)):
        addr_noisy = addr[i][0] 
        addr_clean = addr[i][1]
        print("    [%d/%d] %s" % (i + 1, len(addr), addr_noisy))

        # Read wave fileis.
        fs, audio = wav.read(addr_noisy)
        mic0_audio = audio[:,0]
        mic1_audio = audio[:,1]
        wav_noisy = mic1_audio.astype(np.float16) / (2 ** 15)
        fs, audio = wav.read(addr_clean)
        wav_clean = audio.astype(np.float16) / (2 ** 15)
        wav_noise = wav_noisy - wav_clean

        if feature_type == 'logmag_noisy':
            logmag_noisy, pha_noisy \
                = ssp.wav2logmagpha_half(wav_noisy, fs, shps.window_analysis, 
                                        shps.frm_size, shps.ratio)
            data.append(np.float16(logmag_noisy))

        elif feature_type == 'logmag_clean':
            logmag_clean, pha_clean \
                = ssp.wav2logmagpha_half(wav_clean, fs, shps.window_analysis, 
                                         shps.frm_size, shps.ratio)
            data.append(np.float16(logmag_clean))

        elif feature_type == 'logmag_noise':
            logmag_noise, pha_noise \
                = ssp.wav2logmagpha_half(wav_noise, fs, shps.window_analysis, 
                                         shps.frm_size, shps.ratio)
            data.append(np.float16(logmag_noise))

        elif feature_type == 'ibm':
            mag_clean, pha_clean \
                = ssp.wav2magpha_half(wav_clean, fs, shps.window_analysis, 
                                         shps.frm_size, shps.ratio)
            mag_noise, pha_noise \
                = ssp.wav2magpha_half(wav_noise, fs, shps.window_analysis, 
                                         shps.frm_size, shps.ratio)
            ibm = np.log10((mag_clean ** 2) / (mag_noise ** 2))
            ibm[np.where(ibm > 0)] = 1
            ibm[np.where(ibm < 0)] = 0
            data.append(np.float16(ibm))

        elif feature_type == 'irm':
            mag_clean, pha_clean \
                = ssp.wav2magpha_half(wav_clean, fs, shps.window_analysis, 
                                         shps.frm_size, shps.ratio)
            mag_noise, pha_noise \
                = ssp.wav2magpha_half(wav_noise, fs, shps.window_analysis, 
                                         shps.frm_size, shps.ratio)

            irm = np.sqrt((mag_clean ** 2) / ((mag_clean ** 2) + (mag_noise ** 2)))
            data.append(np.float16(irm))

    return data

if __name__ == '__main__':
    main()






 
