import soundfile as sf
import os
import numpy as np
from pystoi.stoi import stoi

"""Read data & Compute STOI"""

snr_set_string = ['m5dB', '0dB', '5dB', '10dB']

# snr_type = 'm5dB'

noise_subset = ['destroyerengine', 'destroyerops', 'factory2', 'leopard', 'm109', 'machinegun']
# noise_type = 'destroyerengine'



for noise_type in noise_subset:
    for snr_type in snr_set_string:

        clean = '../data/speech/test/clean_test'
        enhanced = '../data/speech/test/noisy_test/' + noise_type + os.sep + snr_type

        # mode = 'noisy'
        # model_name = 'sample_' + mode + '_' + mode

        clean_list = []
        for (path1, dir1, files1) in sorted(os.walk(clean)):
            for data1 in files1:
                ext1 = os.path.splitext(data1)[-1]
                if ext1 == '.wav':
                    clean_list.append("%s/%s" % (path1, data1))

        enhanced_list = []
        for (path2, dir2, files2) in sorted(os.walk(enhanced)):
            for data2 in files2:
                ext2 = os.path.splitext(data2)[-1]
                if ext2 == '.wav':
                    enhanced_list.append("%s/%s" % (path2, data2))

        stoi_list = []
        for src, enh in zip(clean_list, enhanced_list):
            os.path.isdir(src)
            clean_signal, fs = sf.read(src)
            enhanced_signal, fs = sf.read(enh)
            d = stoi(clean_signal, enhanced_signal, fs, extended = False)

            stoi_list.append(d)

        stoivalue = np.mean(stoi_list)
        print(stoivalue)

