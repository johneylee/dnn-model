import torch
relu = torch.nn.ReLU()
sigmo = torch.nn.Sigmoid()

"""Train"""
train_input_data_path = './features_mic0/train_logmag'
train_refer_data_path = './features_mic0/train_irm'
devel_input_data_path = './features_mic0/devel_logmag'
devel_refer_data_path = './features_mic0/devel_irm'


job_type = 'train'
job_dir = './job/dnn_mic0'
mode = 'irm'    # irm, ibm, spectrogram
network_type = 'dnn'   # lstm, dnn
activation_func = relu

fs = 16e3
input_dim = 257
output_dim = 257
num_layer = 4
layer_size = 1024
num_context_window = 2
batch_size = 1

learning_rate = 0.0005
max_epochs = 100
gpu_id = 0
gpu_frac = 0.90

""""Decode"""
network_model_name = './job/dnn_mic0'
clean_directory = '../data_lg_new/speech/test/noisy'
#clean_directory = '../data_lg_new/speech/test/clean'
