#!/usr/bin/python

"""
Speech enhancement neural network libraries
It contains functions of network using tensorflow.
Functions include define parameters of network, save and load network model,
define RNN/fully connected layer, feed layers, define speech enhancement
network and train/evaluate network.
"""
import os
import time
import numpy as np
import ssplib as ssp
import enhlib as enh
import tensorflow as tf
import librosa
from tensorflow.contrib import rnn
from collections import namedtuple
import matplotlib.pyplot as plt
import pdb
# Define namedtuple of hyper parameters for learning.
hparams = namedtuple('hparams',
                     'name, struct, opt, '
                     'obj_mode, obj_weight, '
                     'rnn_type, fc_type, '
                     'featype_inp_list, featype_ref_list, '
                     'num_rnn_layers, num_lstm_cells, '
                     'num_fc_layers, num_fc_nodes, '
                     'learning_rate, thr_clip, max_epochs, '
                     'batch_size')


class TensorData:
    def __init__(self):
        self.data = None
        self.data_vnorm = None
        self.data_znorm = None
        self.mu = None
        self.sig = None


class SeNetwork:
    """
    """

    def __init__(self):
        self.hps = None
        self.decode = None
        #self.pcmout = None
        self.attout_r = None
        self.attout_i = None
        #self.conv1out = None
        self.pcmout_r = None
        self.pcmout_i = None
        self.conv1out_r_1 = None
        self.conv1out_i_1 = None
        self.conv1out_r_2 = None
        self.conv1out_i_2 = None
        self.err1 = None
        self.err2 = None
        self.optimization = None
        self.data_inp = None
        self.data_ref = None
        self.handle = None
        self.feature_info = None
        self.saver_export = None
        self.rnn = None
        self.mu_inp = None
        self.mu_ref = None
        self.sig_inp = None
        self.sig_ref = None
        self.total_params = 0
        self.dir_to_save = None
        self.data_pipeline_train = None
        self.data_pipeline_devel = None
        return

    def set_params(self, hps):
        # Set hyperparameters
        self.hps = hps
        # Name directory for save
        self.dir_to_save = hps.name

    def calculate_total_params(self):
        # Calculate the size of total network.
        self.total_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            self.total_params += variable_parameters

    def initialize_model(self, sess, data_pipeline_train):
        """
        """
        # Initialize parameters and TensorFlow network.
        self.mu_inp = data_pipeline_train.mu_inp
        self.sig_inp = data_pipeline_train.sig_inp
        self.mu_ref = data_pipeline_train.mu_ref
        self.sig_ref = data_pipeline_train.sig_ref
        self.feature_info = data_pipeline_train.feature_info

        # Define speech enhancement network.
        self.create_placeholder()
        self.define_graph()
        self.calculate_total_params()

        # Initialize all variables in the network.
        sess.run(tf.global_variables_initializer())

        tf.add_to_collection('optimization', self.optimization)
        tf.add_to_collection('err1', self.err1)
        tf.add_to_collection('err2', self.err2)
        tf.add_to_collection('decode', self.decode)
        tf.add_to_collection('att_r', self.attout_r)
        tf.add_to_collection('att_i', self.attout_i)
        #tf.add_to_collection('pcmout', self.pcmout)
        #tf.add_to_collection('conv1out', self.conv1out)
        tf.add_to_collection('pcmout_r', self.pcmout_r)
        tf.add_to_collection('pcmout_i', self.pcmout_i)
        tf.add_to_collection('conv1out_r_1', self.conv1out_r_1)
        tf.add_to_collection('conv1out_i_1', self.conv1out_i_1)
        tf.add_to_collection('conv1out_r_2', self.conv1out_r_2)
        tf.add_to_collection('conv1out_i_2', self.conv1out_i_2)
        tf.add_to_collection('data_inp', self.data_inp)
        tf.add_to_collection('data_ref', self.data_ref)
        tf.add_to_collection('handle', self.handle)

        # Save the statistical parameters.
        if os.path.isdir(self.dir_to_save) is False:
            print(self.dir_to_save)
        os.system('mkdir ' + self.dir_to_save)
        os.system('mkdir ' + self.dir_to_save + '/opt')
        os.system('mkdir ' + self.dir_to_save + '/tmp')
        np.save(self.dir_to_save + '/opt/mu_inp.npy', self.mu_inp)
        np.save(self.dir_to_save + '/opt/sig_inp.npy', self.sig_inp)
        np.save(self.dir_to_save + '/opt/mu_ref.npy', self.mu_ref)
        np.save(self.dir_to_save + '/opt/sig_ref.npy', self.sig_ref)

    def export_model(self, sess, addr_model):
        """
        """
        self.saver_export = tf.train.Saver(max_to_keep=1000)
        self.saver_export.save(sess, addr_model)

    def create_placeholder(self):
        """Create placeholder.
        """
        self.handle = tf.placeholder(tf.string, shape=[])
        self.data_inp \
            = tf.placeholder(tf.float32,
                             [None, None, self.feature_info.inp.dim],
                             name='data_inp')

    def define_layers(self):
        """Define RNN and fully connected layers.
        """
        # Define Uni/Bi-directional LSTM layers
        num_rnn_layers = self.hps.num_rnn_layers
        num_lstm_cells = self.hps.num_lstm_cells
        self.rnn = []
        if self.hps.rnn_type == 'blstmblockfused':
            for idx in range(num_rnn_layers):
                sublayer_fw \
                    = rnn.LSTMBlockFusedCell(num_units=num_lstm_cells,
                                             use_peephole=True,
                                             name='lstm_' + str(idx) + '_fw')
                sublayer_bw \
                    = rnn.LSTMBlockFusedCell(num_units=num_lstm_cells,
                                             use_peephole=True,
                                             name='lstm_' + str(idx) + '_bw')
                sublayer_bw \
                    = tf.contrib.rnn.TimeReversedFusedRNN(sublayer_bw)
                self.rnn.append([sublayer_fw, sublayer_bw])

    def feed_rnn_layers(self, data_inp, length, num_rnn_layers,
                        num_lstm_cells):
        """Connect RNN layers.
        """
        self.rnn = []
        if self.hps.rnn_type == 'blstmblockfused':
            for idx in range(num_rnn_layers):
                sublayer_fw \
                    = rnn.LSTMBlockFusedCell(num_units=num_lstm_cells,
                                             use_peephole=True,
                                             name='lstm' + str(idx) + '_fw')
                sublayer_bw \
                    = rnn.LSTMBlockFusedCell(num_units=num_lstm_cells,
                                             use_peephole=True,
                                             name='lstm' + str(idx) + '_bw')
                sublayer_bw \
                    = tf.contrib.rnn.TimeReversedFusedRNN(sublayer_bw)
                self.rnn.append([sublayer_fw, sublayer_bw])
                
        # Transpose tensor
        data_inp_tr = tf.transpose(data_inp, [1, 0, 2])
        if self.hps.rnn_type == 'blstmblockfused':
            h_blstm_tr = data_inp_tr
            for idx in range(num_rnn_layers):
                h_lstm_tr_fw, _ \
                    = self.rnn[idx][0](inputs=h_blstm_tr,
                                       sequence_length=length,
                                       dtype=tf.float32)
                h_lstm_tr_bw, _ \
                    = self.rnn[idx][1](inputs=h_blstm_tr,
                                       sequence_length=length,
                                       dtype=tf.float32)
                h_blstm_tr = tf.concat([h_lstm_tr_fw, h_lstm_tr_bw], axis=2)
            # Transpose tensor
            out = tf.transpose(h_blstm_tr, [1, 0, 2])
        else:
            out = data_inp

        return out

    def define_optimization(self, loss):
        """Define optimizer.
        """
        learning_rate = self.hps.learning_rate

        optimizer \
            = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads, variables = zip(*optimizer.compute_gradients(loss))
        if self.hps.thr_clip > 0.0:
            grads, _ = tf.clip_by_global_norm(grads, self.hps.thr_clip)
        self.optimization = optimizer.apply_gradients(zip(grads, variables))

    def process_io_features(self):
        print('----------------------------------------'
              '----------------------------------------')
        # Input features
        inp = {}
        begin = 0
        for idx in range(len(self.feature_info.inp.dim_idx)):
            size = self.feature_info.inp.dim_idx[idx]
            fea_type = self.feature_info.inp.fea_type[idx]
            print('inp[%d:%d]: %s' % (begin, begin + size, fea_type))
            inp[fea_type] = TensorData()
            inp[fea_type].data \
                = tf.slice(self.data_inp, [0, 0, begin], [-1, -1, size])
            inp[fea_type].mu = tf.slice(self.mu_inp, [begin], [size])
            inp[fea_type].sig = tf.slice(self.sig_inp, [begin], [size])
            inp[fea_type].data_znorm \
                = (inp[fea_type].data - inp[fea_type].mu) / inp[fea_type].sig
            begin = begin + size
        print('----------------------------------------'
              '----------------------------------------')
        # Output features
        ref = {}
        begin = 0
        for idx in range(len(self.feature_info.ref.dim_idx)):

            size = self.feature_info.ref.dim_idx[idx]
            fea_type = self.feature_info.ref.fea_type[idx]
            print('ref[%d:%d]: %s' % (begin, begin + size, fea_type))
            ref[fea_type] = TensorData()
            ref[fea_type].data \
                = tf.slice(self.data_ref, [0, 0, begin], [-1, -1, size])
            ref[fea_type].mu = tf.slice(self.mu_ref, [begin], [size])
            ref[fea_type].sig = tf.slice(self.sig_ref, [begin], [size])

            if fea_type in ['irm', 'cirm_real', 'cirm_imag',
                            'mag_clean', 'mag_noise',
                            'mag_warp_clean', 'mag_warp_noise',
                            'mag_norm_warp_clean', 'mag_norm_warp_noise',
                            'frm_hann_clean', 'frm_hann_norm_clean',
                            'frm_rect_clean', 'frm_rect_norm_clean',
                            'frm_rect_mulaw_clean']:
                ref[fea_type].data_vnorm \
                    = ref[fea_type].data / ref[fea_type].sig
            else:
                ref[fea_type].data_znorm \
                    = (ref[fea_type].data - ref[fea_type].mu) / ref[fea_type].sig
            begin = begin + size
        print('----------------------------------------'
              '----------------------------------------')
        return inp, ref

    def define_graph(self):

        opt = self.hps.opt
        iterator = tf.data.Iterator.from_string_handle(
            self.handle, output_types=(tf.float32, tf.float32))
        next_elements = iterator.get_next()

        dim = 514
        structure = self.hps.struct
        if structure in ['pcm']:
            dim_out = dim * 4
        elif structure in ['pcm_pha']:
            dim_out = dim * 3

        else:
            dim_out = dim

        # Read placeholder
        self.data_inp = next_elements[0]
        self.data_ref = next_elements[1]
        inp, ref = self.process_io_features()

        # Graph
        noisy_logmag_znorm = inp['logmag_noisy'].data_znorm
        noisy_logmag = inp['logmag_noisy'].data
        noisy_pha = inp['pha_noisy'].data
        seq_length = self.get_length(noisy_logmag)
        rnn_mask = tf.sign(tf.reduce_max(tf.abs(noisy_logmag), 2))

        noisy_mag = tf.exp(noisy_logmag)
        rnn_out = self.feed_rnn_layers(noisy_logmag_znorm, seq_length,
                                       num_rnn_layers=self.hps.num_rnn_layers,
                                       num_lstm_cells=self.hps.num_lstm_cells)
        fc_h1 = tf.contrib.layers.fully_connected(rnn_out,
                                                  self.hps.num_fc_nodes,
                                                  activation_fn=tf.tanh)
        fc_out = tf.contrib.layers.fully_connected(fc_h1, dim_out,
                                                   activation_fn=None)

        # Estimation
        est_spec_real = None
        est_spec_imag = None
        est_mag_warp_clean_vnorm = None
        est_mag_warp_noise_vnorm = None
        est_cos_xn = None
        est_cos_xy = None
        est_sin_xy = None

        if structure == 'real':
            est_mask_real \
                = enh.define_real(fc_out, opt)
            est_mag = tf.multiply(est_mask_real, noisy_mag, name='est_mag')
            est_spec_real = est_mag * tf.cos(noisy_pha)
            est_spec_imag = est_mag * tf.sin(noisy_pha)

        elif structure == 'pcm_pha':
            noisy_bpd_norm = inp['bpd_noisy'].data_znorm

            est_mag_warp_clean_vnorm, est_mag_warp_noise_vnorm, est_cos_xn \
                = enh.define_pcm_mag(fc_out)

            sig = ref['mag_norm_warp_clean'].sig
            # eq 12
            est_mask_mag \
                = enh.parametric_complex_tfmask_function_mag\
                (est_mag_warp_clean_vnorm, est_mag_warp_noise_vnorm,
                 est_cos_xn, sig)
          
            est_mag = tf.multiply(est_mask_mag, noisy_mag, name='est_mag')

            est_logmag_fixed = tf.log(tf.stop_gradient(est_mag) + 1e-7)

            est_logmag_fixed_znorm = (est_logmag_fixed - inp[
                'logmag_noisy'].mu) / inp['logmag_noisy'].sig

            # Phase difference of speech and noisy spectra
            phanet_in = tf.concat([est_logmag_fixed_znorm, noisy_bpd_norm],
                                  axis=-1)

            rnn_out_pha \
                = self.feed_rnn_layers(phanet_in, seq_length,
                                       num_rnn_layers=2,
                                       num_lstm_cells=256)
            fc_h1_pha \
                = tf.contrib.layers.fully_connected(rnn_out_pha, 512,
                                                    activation_fn=tf.tanh)
            est_pha_xy \
                = tf.contrib.layers.fully_connected(fc_h1_pha, dim,
                                                    activation_fn=None)

            est_cos_xy = tf.cos(est_pha_xy)
            est_sin_xy = tf.sin(est_pha_xy)

            est_mask_real = est_mask_mag * est_cos_xy
            est_mask_imag = est_mask_mag * est_sin_xy

            est_spec_real_raw, est_spec_imag_raw \
                = enh.complex_domain_tf_masking(est_mask_real, est_mask_imag,
                                                noisy_mag, noisy_pha)
            # separate real and imag
            est_spec_real_mic0 = est_spec_real_raw[:,:,:257]
            est_spec_real_mic1 = est_spec_real_raw[:,:,257:]            
            est_spec_imag_mic0 = est_spec_imag_raw[:,:,:257]
            est_spec_imag_mic1 = est_spec_imag_raw[:,:,257:]

            # matrix multiplication
            mul_main_real_mic1 = tf.matmul(est_spec_real_mic0, tf.transpose(est_spec_real_mic1, perm=[0,2,1]))
            mul_main_real_mic0 = tf.matmul(est_spec_real_mic1, tf.transpose(est_spec_real_mic0, perm=[0,2,1]))
            mul_main_imag_mic1 = tf.matmul(est_spec_imag_mic0, tf.transpose(est_spec_imag_mic1, perm=[0,2,1]))
            mul_main_imag_mic0 = tf.matmul(est_spec_imag_mic1, tf.transpose(est_spec_imag_mic0, perm=[0,2,1]))

            # calculate softmax score
            soft_max_real_mic1 = tf.nn.softmax(mul_main_real_mic1, axis=-1)
            soft_max_real_mic0 = tf.nn.softmax(mul_main_real_mic0, axis=-1)
            soft_max_imag_mic1 = tf.nn.softmax(mul_main_imag_mic1, axis=-1)
            soft_max_imag_mic0 = tf.nn.softmax(mul_main_imag_mic0, axis=-1)

            # multiply the main channel
            atte_est_spec_real_mic0 = tf.matmul(tf.transpose(est_spec_real_mic0, perm=[0,2,1]), tf.transpose(soft_max_real_mic0, perm=[0,2,1]))
            atte_est_spec_real_mic1 = tf.matmul(tf.transpose(est_spec_real_mic1, perm=[0,2,1]), tf.transpose(soft_max_real_mic1, perm=[0,2,1]))
            atte_est_spec_imag_mic0 = tf.matmul(tf.transpose(est_spec_imag_mic0, perm=[0,2,1]), tf.transpose(soft_max_imag_mic0, perm=[0,2,1]))
            atte_est_spec_imag_mic1 = tf.matmul(tf.transpose(est_spec_imag_mic1, perm=[0,2,1]), tf.transpose(soft_max_imag_mic1, perm=[0,2,1]))

            #est_spec_real_att = tf.concat([tf.transpose(atte_est_spec_real_mic0, perm=[0,2,1]), tf.transpose(atte_est_spec_real_mic1, perm=[0,2,1])], -1)
            #est_spec_imag_att = tf.concat([tf.transpose(atte_est_spec_imag_mic0, perm=[0,2,1]), tf.transpose(atte_est_spec_imag_mic1, perm=[0,2,1])], -1)

            # applying 1D convolution layer
            
            # add attention + origin : residual connection
            #est_spec_real_ = est_spec_real_att + est_spec_real_raw 
            #est_spec_imag_ = est_spec_imag_att + est_spec_imag_raw

            # concate origin, add attetion
            #est_spec_real_ = tf.concat([est_spec_real_raw, est_spec_real_att], -1)
            #est_spec_imag_ = tf.concat([est_spec_imag_raw, est_spec_imag_att], -1)
       
            # complex network
            #y_real = tf.realnetwork(x_real) - tf.imagnetwork(x_imag)
            #y_imag = tf.realnetwork(x_imag) + tf.imagnetwork(x_real)
            
            #est_spec_real_1 = tf.layers.conv1d(est_spec_real_, 771, 11, padding='same', activation='relu')
            #est_spec_real_2 = tf.layers.conv1d(est_spec_real_1, 514, 11, padding='same', activation='relu')
            #est_spec_real = tf.layers.conv1d(est_spec_real_2, 257, 5, padding='same')
            #est_spec_imag_1 = tf.layers.conv1d(est_spec_imag_, 771, 11, padding='same', activation='relu')
            #est_spec_imag_2 = tf.layers.conv1d(est_spec_imag_1, 514, 11, padding='same', activation='relu')
            #est_spec_imag = tf.layers.conv1d(est_spec_imag_2, 257, 5, padding='same')

            # applying 2D convolution layer
            est_spec_real_mic0 = tf.expand_dims(est_spec_real_mic0, axis = -1)
            est_spec_real_mic1 = tf.expand_dims(est_spec_real_mic1, axis = -1)
            atte_est_spec_real_mic0 = tf.expand_dims(tf.transpose(atte_est_spec_real_mic0, perm=[0,2,1]), axis = -1)
            atte_est_spec_real_mic1 = tf.expand_dims(tf.transpose(atte_est_spec_real_mic1, perm=[0,2,1]), axis = -1)
            est_spec_imag_mic0 = tf.expand_dims(est_spec_imag_mic0, axis = -1)
            est_spec_imag_mic1 = tf.expand_dims(est_spec_imag_mic1, axis = -1)
            atte_est_spec_imag_mic0 = tf.expand_dims(tf.transpose(atte_est_spec_imag_mic0, perm=[0,2,1]), axis = -1)
            atte_est_spec_imag_mic1 = tf.expand_dims(tf.transpose(atte_est_spec_imag_mic1, perm=[0,2,1]), axis = -1)

            # non process
            #est_spec_real_ = tf.concat([est_spec_real_mic0, est_spec_real_mic1], axis = -1)
            #est_spec_imag_ = tf.concat([est_spec_imag_mic0, est_spec_imag_mic1], axis = -1)
            # add process
            #est_spec_real_ = tf.concat([est_spec_real_mic0 + atte_est_spec_real_mic0, est_spec_real_mic1 + atte_est_spec_real_mic1], axis = -1)
            #est_spec_imag_ = tf.concat([est_spec_imag_mic0 + atte_est_spec_imag_mic0, est_spec_imag_mic1 + atte_est_spec_imag_mic1], axis = -1)
            # concat process
            est_spec_real_ = tf.concat([est_spec_real_mic0, est_spec_real_mic1, atte_est_spec_real_mic0, atte_est_spec_real_mic1], axis = -1)
            est_spec_imag_ = tf.concat([est_spec_imag_mic0, est_spec_imag_mic1, atte_est_spec_imag_mic0, atte_est_spec_imag_mic1], axis = -1)


            est_spec_real_1 = tf.keras.layers.Conv2D(filters = 16,
                                                    kernel_size = 3,
                                                    padding = 'same',
                                                    activation = tf.nn.leaky_relu,
                                                    data_format = 'channels_last')(est_spec_real_) 
            est_spec_real_2 = tf.keras.layers.Conv2D(filters = 8,
                                                    kernel_size = 3,
                                                    padding = 'same',
                                                    activation = tf.nn.leaky_relu,
                                                    data_format = 'channels_last')(est_spec_real_1)
            est_spec_real = tf.keras.layers.Conv2D(filters = 1,
                                                    kernel_size = 3,
                                                    padding = 'same',
                                                    activation = tf.nn.leaky_relu,
                                                    data_format = 'channels_last')(est_spec_real_2)
            est_spec_imag_1 = tf.keras.layers.Conv2D(filters = 16,
                                                    kernel_size = 3,
                                                    padding = 'same',
                                                    activation = tf.nn.leaky_relu,
                                                    data_format = 'channels_last')(est_spec_imag_)
            est_spec_imag_2 = tf.keras.layers.Conv2D(filters = 8,
                                                    kernel_size = 3,
                                                    padding = 'same',
                                                    activation = tf.nn.leaky_relu,
                                                    data_format = 'channels_last')(est_spec_imag_1)
            est_spec_imag = tf.keras.layers.Conv2D(filters = 1,
                                                    kernel_size = 3,
                                                    padding = 'same',
                                                    activation = tf.nn.leaky_relu,
                                                    data_format = 'channels_last')(est_spec_imag_2)
            est_spec_real = tf.squeeze(est_spec_real, axis = -1)
            est_spec_imag = tf.squeeze(est_spec_imag, axis = -1) 
        # Network out for decoding
        est_spec_concat = tf.concat([est_spec_real, est_spec_imag], 2)
        network_out = tf.cast(est_spec_concat, dtype=tf.float32)

        # Learning
        obj_mode = self.hps.obj_mode
        obj_weight = self.hps.obj_weight
        obj = 0.0
        for idx in range(len(obj_mode)):
            if obj_mode[idx] == 'etdr':
                obj_etdr = enh.define_obj_etdr(ref, est_spec_real,
                                               est_spec_imag,
                                               dim, seq_length, rnn_mask)
                obj += obj_weight[idx] * obj_etdr

            elif obj_mode[idx] == 'params':
                obj_params \
                    = enh.define_obj_params(ref, est_mag_warp_clean_vnorm,
                                            est_mag_warp_noise_vnorm,
                                            est_cos_xn, est_cos_xy,
                                            est_sin_xy, rnn_mask)
                obj += obj_weight[idx] * obj_params
            elif obj_mode[idx] == 'sc':
                obj_sc = enh.define_obj_sc(ref, est_mask_mag,
                                           noisy_mag, noisy_pha,
                                           est_cos_xy, est_sin_xy, dim,
                                           seq_length, rnn_mask)
                obj += obj_weight[idx] * obj_sc

        # self.pcmout = [est_spec_real_raw, est_spec_imag_raw]
        # self.conv1out = [_est_spec_real, _est_spec_imag]
        self.attout_r = est_spec_real_
        self.attout_i = est_spec_imag_
        self.pcmout_r = est_spec_real_raw
        self.pcmout_i = est_spec_imag_raw
        self.conv1out_r_1 = est_spec_real_1
        self.conv1out_i_1 = est_spec_imag_1
        self.conv1out_r_2 = est_spec_real_2
        self.conv1out_i_2 = est_spec_imag_2
        self.decode = network_out
        self.err1 = obj
        if 'etdr' in obj_mode:
            self.err2 = obj_etdr
        else:
            self.err2 = obj
        # Define optimizer.
        self.define_optimization(obj)

    def train(self, sess, data_pipeline_train, data_pipeline_devel):
        """Train network.
        """
        # Set batch size for training and evaluating.
        num_batches_train = data_pipeline_train.num_batches
        num_batches_devel = data_pipeline_devel.num_batches
        handle_train = data_pipeline_train.handle
        handle_devel = data_pipeline_devel.handle

        # Write log file.
        dir_to_save = self.dir_to_save
        fp = open(dir_to_save + '/opt/log.txt', 'w')
        self.write_status_to_log_file(fp)

        # Start training.
        vec_loss1 = np.zeros(self.hps.max_epochs + 1)
        vec_loss2 = np.zeros(self.hps.max_epochs + 1)
        for i in range(self.hps.max_epochs + 1):
            start_time = time.time()

            # Train data.
            if i > 0:
                loss_train_arr = np.zeros(num_batches_train)
                for j in range(num_batches_train):
                    loss_train_arr[j], _ \
                        = sess.run([self.err1, self.optimization],
                                   feed_dict={self.handle: handle_train})

                loss_train = np.mean(loss_train_arr)
            
            else:
                loss_train = 0.0

            # Evaluate data.
            loss_devel_arr = np.zeros(num_batches_devel)
            err_devel_arr = np.zeros(num_batches_devel)
            for j in range(num_batches_devel):
                loss_devel_arr[j], err_devel_arr[j] \
                    = sess.run([self.err1, self.err2],
                               feed_dict={self.handle: handle_devel})
            loss_devel = float(np.mean(loss_devel_arr))
            err_devel = float(np.mean(err_devel_arr))
            vec_loss1[i] = loss_devel
            vec_loss2[i] = err_devel

            # Print progress.
            print('epoch %d: %.6f %.6f %.6f (%.0f sec)'
                  % (i, loss_train, loss_devel, err_devel,
                     time.time() - start_time))
            fp.write('epoch %d: %.6f %.6f %.6f (%.0f sec)\n'
                     % (i, loss_train, loss_devel, err_devel,
                        time.time() - start_time))
            # Save models.
            self.export_model(sess, '%s/tmp/%d.ckpt' % (dir_to_save, i))

        # Copy optimum model that has minimum MSE.
        min_index1 = int(np.argmin(vec_loss1))
        os.system('cp %s/tmp/%d.ckpt.data-00000-of-00001 \
                                %s/opt/opt.ckpt.data-00000-of-00001' %
                  (dir_to_save, min_index1, dir_to_save))
        os.system('cp %s/tmp/%d.ckpt.index %s/opt/opt.ckpt.index' %
                  (dir_to_save, min_index1, dir_to_save))
        os.system('cp %s/tmp/%d.ckpt.meta %s/opt/opt.ckpt.meta' %
                  (dir_to_save, min_index1, dir_to_save))
        fp.close()

    def write_status_to_log_file(self, fp):
        """Write log file.
        """
        fp.write('%d-%d-%d %d:%d:%d\n' %
                 (time.localtime().tm_year, time.localtime().tm_mon,
                  time.localtime().tm_mday, time.localtime().tm_hour,
                  time.localtime().tm_min, time.localtime().tm_sec))
        fp.write('mode          : %s\n' % self.hps.obj_mode)
        fp.write('opt           : %s\n' % self.hps.opt)
        fp.write('inp feature   : %s\n' % self.hps.featype_inp_list)
        fp.write('ref feature   : %s\n' % self.hps.featype_ref_list)
        fp.write('rnn type      : %s\n' % self.hps.rnn_type)
        fp.write('fc type       : %s\n' % self.hps.fc_type)
        fp.write('learning rate : %g\n' % self.hps.learning_rate)
        fp.write('batch size    : %d\n' % self.hps.batch_size)
        fp.write('clipping thr  : %.2f\n' % self.hps.thr_clip)
        fp.write('obj_weight    : %s\n' % self.hps.obj_weight)
        fp.write('total params  : %d (%.2f M, %.2f MBytes)\n' %
                 (self.total_params,
                  self.total_params / 1000000.0,
                  self.total_params * 4.0 / 1000000.0))
        for variable in tf.trainable_variables():
            fp.write('%s\n' % variable)

    @staticmethod
    def get_length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length
