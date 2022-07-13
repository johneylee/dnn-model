import os
import sys
import numpy as np
import scipy.io.wavfile as wav
from collections import namedtuple
import torch
from os import makedirs


for (path, dir, files) in os.walk("/home/leesunghyun/Downloads/IRM/workspace/data_lg/beamforming/MVDR_sl_eye/MVDR_sl_eye/TEST_CORE_directive_2mic"):
    for data in files:
        ext = os.path.splitext(data)[-1]
        if ext == '.wav':
            addr_input = "%s/%s" % (path, data)
            addr_new = path.replace('TEST_CORE_directive_2mic', 'namechange')
            makedirs(addr_new, exist_ok=True)
            data_split = data.split('_')
            data_new = data_split[1] + "_" + data_split[2] + ".wav"
            addr_new_data = addr_new + '/' + data_new
            os.rename(addr_input, addr_new_data)

        # if os.path.isfile(addr_input) is False:
        #     print('[Error] There is no input file.')
        #     exit()
        # perform_enhancement(addr_input, model_name, mode, shps)
#
#
# for (path, dir, files) in os.walk("../data_lg/beamforming/noisymchstft"):
#     for data in files:
#         ext = os.path.splitext(data)[-1]
#         if ext == '.npy':
#             addr_input = "%s/%s" % (path, data)
#             stft_file = np.load(addr_input)
#             data_out0 = (istft(stft_file[:,0,:], frame_size, overlap_factor) * 32768).astype(np.int16) # write의 target
#
#             data_out1 = (istft(stft_file[:,1,:], frame_size, overlap_factor) * 32768).astype(np.int16)
#
#             addr_new0 = path.replace('data', 'mic0')
#             addr_new1 = path.replace('data', 'mic1')
#             makedirs(addr_new0, exist_ok=True)
#             makedirs(addr_new1, exist_ok=True)
#
#             addr_out = addr_input.replace('.noisyMchSTFT.npy', '_' + 'e.wav')
#             enhanced_out0 = addr_out.replace('data', 'mic0')
#             enhanced_out1 = addr_out.replace('data', 'mic1')
#
#             wav.write(enhanced_out0, 16000, data_out0)
#             wav.write(enhanced_out1, 16000, data_out1)