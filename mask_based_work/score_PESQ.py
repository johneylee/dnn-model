import os
import numpy as np
import glob
"""
    PESQ batch
    leesunghyun
    data: 2019.08.13
    
"""


"""Read data & Compute PESQ"""

snr_type = '0dB'
noise_type = 'factory2'
clean = '../data/speech/test/clean'
enhanced = '../spectrum/speech/test/noisy/' + noise_type + os.sep + snr_type

mode = 'spectrum'
model_name = 'sample_' + mode + '_' + mode

clean_list = []
for (path1, dir1, files1) in sorted(os.walk(clean)):
    for data1 in files1:
        ext = os.path.splitext(data1)[-1]
        if ext == '.wav':
            clean_list.append("%s/%s" % (path1, data1))

enhanced_list = []
for (path2, dir2, files2) in sorted(os.walk(enhanced)):
    for data2 in files2:
        ext = os.path.splitext(data2)[-1]
        if ext == '.wav':
            enhanced_list.append("%s/%s" % (path2, data2))

for clean, enhanced in zip(clean_list, enhanced_list):
    os.path.isdir(clean)
    os.system('PESQ +16000' + ' ' + clean + ' ' + enhanced)


"""Compute average for PESQ"""

Raw = []
LQO = []
with open('./pesq_results.txt') as f:
    read_data = f.readlines()
    for line in read_data[1:]:
        x = line.split("	 ")
        Raw.append(x[2])
        LQO.append(x[3])

f.close()

MOS_Raw = []
for i in Raw:
    MOS_Raw.append(float(i))
MOS_LQO = []
for j in LQO:
    MOS_LQO.append(float(j))

MOS_Raw_mean = np.mean(MOS_Raw)
MOS_LQO_mean = np.mean(MOS_LQO)

print(np.shape(MOS_Raw))
print('{} {} {} PESQ: {}'.format(mode, noise_type, snr_type, MOS_Raw_mean))
print(np.shape(MOS_LQO))
print('{} {} {} PESQ: {}'.format(mode, noise_type, snr_type, MOS_LQO_mean))
