# coding: utf-8
"""
Train network for speech enhancement

usage: train_network.py <name> <struct> <mode> <opt> <dir_fea> <featype_input> <featype_refer> [options]

options:
    --gpu_id=<int>                  GPU ID [default: 0]
    --rnn_type=<str>                Type of RNN layers [default: lstm]
    --fc_type=<str>                 Type of FC layers [default: tanh]
    --learning_rate=<float>         Learning rate [default: 0.0005]
    --max_epochs=<int>              Maximum number of epochs [default: 100]
    --batch_size=<int>              Batch size [default: 32]
    --thr_clip=<float>              Threshold for clipping [default: 1.0]
    --flag_loss=<int>               Flag for loss function [default: 0]
    --flag_fc_init=<str>            Flag [default: uniform]
    --warp_val=<float>              Magnitude warping factor [default: 1.0]
    --warp_idxs=<str>               Index for processing magnitude warping
                                    [default: []]
    --weights=<str>                 Weight values (string) for loss function
                                    [default: [1,0,0]]
    --load_model=<str>              Pretrained Model [default: ]
    --load_stat=<str>               Pretrained Model [default: ]
"""
import os
import tensorflow as tf
import numpy as np
import se_net as se_net
import data_prep as data_prep
from docopt import docopt

os.environ["CUDA_VISIBLE_DEVICES"]="1"
#from tensorflow.python.client import device_lib
#
#print(device_lib.list_local_devices())


def main():

    # Load system arguments
    args = docopt(__doc__)
    
    # Read input arguments
    name, struct, mode, opt, dir_fea, featype_inp, featype_ref, \
    gpu_id, rnn_type, fc_type, learning_rate, max_epochs, batch_size, \
    thr_clip, flag_fc_init, weights \
        = read_args(args)

    #name = 'models/multi_test_ete'
    #struct = 'pcm_pha'
    #mode = 'params+etdr+sc'
    #opt = 'opt'
    #dir_fea = 'features_updown_target2_small'
    #featype_inp = 'logmag_noisy+pha_noisy+bpd_noisy'
    #featype_ref = 'sig_max+frm_rect_norm_clean+mag_norm_warp_clean+mag_norm_warp_noise+cos_xn+cos_xy+sin_xy'
    #gpu_id = 1
    #rnn_type = 'blstmblockfused'
    #fc_type = 'tanh'
    #learning_rate = 0.0005
    #max_epochs = 20
    #batch_size = 1
    #thr_clip = 0.5
    #weights = '[1,1,2]'


    obj_mode = mode.split('+')
    featype_inp_list = featype_inp.split('+')
    featype_ref_list = featype_ref.split('+')
    
    idx_input_list, idx_refer_list \
        = calculate_feature_dimension(featype_inp_list, featype_ref_list)
    
    obj_weight = process_flag_string_float(weights)

    if len(obj_mode) is not len(obj_weight):
        print('Error: the dimension of "obj_mode" and "obj_weight" '
              'are not the same')
        exit(-1)
    else:
        norm = 0.0
        for idx in range(len(obj_mode)):
            if obj_mode[idx] is not 'sc':
                norm += obj_weight[idx]
        for idx in range(len(obj_weight)):
            obj_weight[idx] /= norm
    print(obj_mode)
    print(obj_weight)
    print('----------------------------------------'
          '----------------------------------------')
    print('  Name               : %s' % name)
    print('  Learning mode      : %s' % mode)
    print('  Option             : %s' % opt)
    print('  Feature Dir.       : %s' % dir_fea)
    print('----------------------------------------'
          '----------------------------------------')
    print('  RNN type           : %s' % rnn_type)
    print('  FC type            : %s' % fc_type)
    print('  Learning rate      : %s' % learning_rate)
    print('  Max. epochs        : %s' % max_epochs)
    print('  Batch size         : %s' % batch_size)
    print('  Clipping thr.      : %s' % thr_clip)
    print('----------------------------------------'
          '----------------------------------------')
    print('  Input feature type : %s' % featype_inp)
    print('  Refer feature type : %s' % featype_ref)
    print('----------------------------------------'
          '----------------------------------------')

    # Load input and reference features
    filelist_train = gen_filelist(dir_fea + '/', "train_", ".tfrecord")
    filelist_devel = gen_filelist(dir_fea + '/', "devel_", ".tfrecord")

    # Define tensorflow session.
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % gpu_id
    options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=options))

    # Set network structures
    num_rnn_layers = 3
    if rnn_type == 'lstm':
        num_lstm_cells = 512
    elif rnn_type == 'blstm':
        num_lstm_cells = 303
    else:
        num_lstm_cells = 512
    num_fc_layers = 2
    num_fc_nodes = 1024

    # Define hyper parameters for data preparation
    dhps = data_prep.hparams(batch_size=batch_size, buffer_size=1024,
                             num_parallel_reads=32, num_parallel_calls=32)

    # Define hyper parameters for learning.
    lhps = se_net.hparams(name=name, struct=struct, opt=opt,
                         obj_mode=obj_mode, obj_weight=obj_weight,
                         rnn_type=rnn_type, fc_type=fc_type,
                         featype_inp_list=featype_inp_list,
                         featype_ref_list=featype_ref_list,
                         num_rnn_layers=num_rnn_layers,
                         num_lstm_cells=num_lstm_cells,
                         num_fc_layers=num_fc_layers,
                         num_fc_nodes=num_fc_nodes,
                         learning_rate=learning_rate, thr_clip=thr_clip,
                         max_epochs=max_epochs, batch_size=batch_size)
    data_pipeline_train = data_prep.DataPreparation()
    data_pipeline_train.initialize(dhps, featype_inp_list, idx_input_list,
                                   featype_ref_list, idx_refer_list)
    data_pipeline_train.load_features(sess, filelist_train,
                                      flag_shuffle=1, flag_norm=1)
    data_pipeline_devel = data_prep.DataPreparation()
    data_pipeline_devel.initialize(dhps, featype_inp_list, idx_input_list,
                                   featype_ref_list, idx_refer_list)
    data_pipeline_devel.load_features(sess, filelist_devel,
                                      flag_shuffle=0, flag_norm=0)

    # Initialize network for speech enhancement.
    net = se_net.SeNetwork()
    net.set_params(lhps)
    net.initialize_model(sess, data_pipeline_train)
    net.train(sess, data_pipeline_train, data_pipeline_devel)


def read_args(args):
    # Model name
    name = args["<name>"]
    # Struct
    struct = args["<struct>"]
    # Learning mode
    mode = args["<mode>"]
    # Structure type
    opt = args["<opt>"]
    # Feature directory
    dir_fea = args["<dir_fea>"]
    # Grammar for features
    featype_input = args["<featype_input>"]
    featype_refer = args["<featype_refer>"]
    # Optional arguments
    gpu_id = int(args["--gpu_id"])
    gpu_id = 0 if gpu_id is None else int(gpu_id)
    # Optional arguments
    rnn_type = args["--rnn_type"]
    fc_type = args["--fc_type"]
    learning_rate = args["--learning_rate"]
    max_epochs = args["--max_epochs"]
    batch_size = args["--batch_size"]
    thr_clip = args["--thr_clip"]
    flag_fc_init = args["--flag_fc_init"]
    weights = args["--weights"]
    # Optional arguments
    rnn_type = 'lstm' if rnn_type is None else rnn_type
    fc_type = 'tanh' if fc_type is None else fc_type
    learning_rate = 0.0005 if learning_rate is None else float(learning_rate)
    max_epochs = 100 if max_epochs is None else int(max_epochs)
    batch_size = 32 if batch_size is None else int(batch_size)
    thr_clip = 1.0 if thr_clip is None else float(thr_clip)
    flag_fc_init = 'uniform' if flag_fc_init is None else flag_fc_init
    weights = '[1.0]' if weights is None else weights

    return name, struct, mode, opt, dir_fea, featype_input, featype_refer, \
           gpu_id, rnn_type, fc_type, learning_rate, max_epochs, batch_size, \
           thr_clip, flag_fc_init, weights


def process_flag_string_float(flag_string):
    """
    """
    flag_string = flag_string.replace('[', '')
    flag_string = flag_string.replace(']', '')
    flag_string = flag_string.replace('(', '')
    flag_string = flag_string.replace(')', '')
    flag_string = flag_string.split(',')
    if flag_string[0] == '':
        flag = [-1]
    else:
        flag = [float(flag_string[i]) for i in range(len(flag_string))]
    return flag


def gen_filelist(dir_fea, prefix, suffix):
    filelist = []
    for subdir, dirs, files in os.walk(dir_fea):
        for file in files:
            filepath = subdir + os.sep + file
            if file.startswith(prefix) and file.endswith(suffix):
                filelist.append(filepath)
    return filelist


def calculate_feature_dimension(featype_input_list, featype_refer_list):
    dim = 514
    idx_inp = []
    for idx in range(len(featype_input_list)):
        if featype_input_list[idx] == 'logmag_noisy':
            idx_inp.append(dim)
        elif featype_input_list[idx] == 'pha_noisy':
            idx_inp.append(dim)
        elif featype_input_list[idx] == 'bpd_noisy':
            idx_inp.append(256)
        else:
            idx_inp.append(dim)

    idx_ref = []
    for idx in range(len(featype_refer_list)):
        if featype_refer_list[idx] == 'sig_max':
            idx_ref.append(1)
        elif featype_refer_list[idx] in ['frm_hann_clean',
                                         'frm_hann_norm_clean']:
            dim_win = (dim - 1) * 2
            idx_ref.append(dim_win)
        elif featype_refer_list[idx] in ['frm_rect_clean',
                                         'frm_rect_norm_clean',
                                         'frm_rect_mulaw_clean']:
            dim_win = int((dim - 1) * 2 / 4)
            idx_ref.append(dim_win)
        else:
            idx_ref.append(dim)
    return idx_inp, idx_ref


if __name__ == '__main__':
    main()
