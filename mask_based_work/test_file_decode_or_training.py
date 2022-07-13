import os
import numpy as np
import matplotlib.pyplot as plt

irm_base_path = '/home/leesunghyun/Downloads/IRM/workspace/present_work/irm_mask/speech/test/noisy'
for (path1, dir, files1) in os.walk(irm_base_path):
    for data1 in files1:
        ext = os.path.splitext(data1)[-1]
        files1 = sorted(files1)
        if ext == '.npy':
            addr_input = "%s/%s" % (path1, data1)

# spectro_base_path = '/home/leesunghyun/Downloads/IRM/workspace/workspace/feature_data/train_target_clean'
# for (path2, dir, files2) in os.walk(spectro_base_path):
#     for data2 in files2:
#         ext = os.path.splitext(data2)[-1]
#         files2 = sorted(files2)
#         if ext == '.npy':
#             addr_input2 = "%s/%s" % (path2, data2)

#
irm_mask = np.load(path1 + '/' + files1[3])
#
# spectro_mask = np.load(path2 + '/' + files2[3])

plt.imshow(irm_mask)
plt.show()
# plt.imshow(spectro_mask)
# plt.show()
