import torch
relu = torch.nn.ReLU()
sigmo = torch.nn.Sigmoid()

"""Train"""
train_input_data_path = './feature_data/train_input_small'
train_refer_data_path = './feature_data/train_ref_irm_small'
devel_input_data_path = './feature_data/devel_input_small'
devel_refer_data_path = './feature_data/devel_ref_irm_small'


job_type = 'train'
job_dir = './job/small_dnn_test6'
mode = 'spectrogram'    # irm, ibm, spectrogram
network_type = 'dnn'   # lstm, dnn
activation_func = relu

fs = 16e3
input_dim = 257
output_dim = 257
num_layer = 1
layer_size = 512
num_context_window = 2
batch_size = 1

learning_rate = 0.0005
max_epochs = 10
gpu_id = 0
gpu_frac = 0.90

""""Decode"""
network_model_name = './job/dnn_layer2_new_spectrogram'
clean_directory = '../data_small/speech/test/noisy'