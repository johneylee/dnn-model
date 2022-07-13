"""
Speech enhancment network library for training
"""

import os
import time
import numpy as np
import tensorflow as tf
from collections import namedtuple
from tools_for_network import *

# Define namedtuple of hyper parameters for learning.
hparams = namedtuple('hparams',
                     'mode, '
                     'job_dir,'
                     'num_context_window, '
                     'dnn_struct, '
                     'learning_rate, '
                     'max_epochs')


class SpeechEnhancementNetwork:
    def __init__(self):
        print('***********************************************************')
        print('*    Python library for DNN-based speech enhancement      *')
        print('*                        using Google\'s TensorFlow API    *')
        print('***********************************************************')
        self.hps = None
        self.decode = None
        self.error = None
        self.optimization = None
        self.data_input = None
        self.data_refer = None
        self.weight = None
        self.bias = None
        self.total_params = 0
        self.dir_to_save = None
        self.saver_export = None
        return

    def set_params(self, hps):
        # Set hyperparameters.
        self.hps = hps
        # Name directroy for save.
        self.dir_to_save = hps.job_dir + '_%s' % hps.mode
        create_folder(self.dir_to_save)

    def calculate_total_params(self):
        # Calculate the size of total network.
        self.total_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            self.total_params += variable_parameters

    def initialize_model(self, sess):
        print('Initialzie network...')
        # Initialize parameters and tensorflow network.

        # Define speech enhancement network.
        self.create_placeholder()
        self.define_layers()

        # Select mode.
        if self.hps.mode == 'direct':
            self.define_direct_network()
        elif self.hps.mode == 'spectrum':
            self.define_spec_network()
        elif self.hps.mode == 'irm':
            self.define_irm_network()
        elif self.hps.mode == 'ibm':
            self.define_ibm_network()
        else:
            print('Invalid speech enhancement mode parameter')

        # Calculate the total number of parameters.
        self.calculate_total_params()

        # Initialize all variables in the network.
        sess.run(tf.global_variables_initializer())
        tf.add_to_collection('decode', self.decode)
        tf.add_to_collection('optimization', self.optimization)
        tf.add_to_collection('data_input', self.data_input)
        tf.add_to_collection('data_refer', self.data_refer)

    def export_model(self, sess, addr_model):
        # Save tensorflow model.

        self.saver_export = tf.train.Saver(max_to_keep=1000)
        self.saver_export.save(sess, addr_model)

    def run_train_step(self, sess, inp, ref):
        # Training the whole network.
        return sess.run(self.optimization,
                        feed_dict={self.data_input: inp, 
                                   self.data_refer: ref})

    def run_eval_step(self, sess, inp, ref):
        # Evaluate the whole network and get mean sqaure error of network.
        to_return = self.error
        return sess.run(to_return,
                        feed_dict={self.data_input: inp, 
                                   self.data_refer: ref})

    def create_placeholder(self):
        # Creat placeholder.
        self.data_input \
            = tf.placeholder(tf.float32, shape=[None, self.hps.dnn_struct[0]])
        self.data_refer \
            = tf.placeholder(tf.float32, shape=[None, self.hps.dnn_struct[-1]])


    def define_layers(self):
        # Define DNN layers.
        nodes = self.hps.dnn_struct
        self.weight = []
        self.bias = []
        with tf.name_scope("DNN"):
            for i in range(1, len(nodes)):        
                W = tf.get_variable("W%d" % i, shape=[nodes[i-1],nodes[i]], initializer=tf.contrib.layers.xavier_initializer())
                self.weight.append(W)
                
                B = tf.Variable(tf.random_normal([nodes[i]]))
                self.bias.append(B)

    def feed_layers(self, data_input):
        # Connect DNN layers.
        for i in range(len(self.weight) - 1):
            if i == 0:
                layer_input = data_input
            else:
                layer_input = H
            H = tf.nn.sigmoid(tf.add(tf.matmul(layer_input, self.weight[i]), self.bias[i]))
        return H


    def define_optimization(self, loss):
        # Define optimizer.
        self.optimization = tf.train.AdamOptimizer(self.hps.learning_rate).minimize(loss)

    def define_spec_network(self):
        # Define loss for mse(enhanced log mag - clean log mag)
        last_hidden_layer = self.feed_layers(self.data_input)
        est_mask = tf.sigmoid(tf.add(tf.matmul(last_hidden_layer, self.weight[-1]), self.bias[-1]))
        input_dim = est_mask.shape[-1]
        noisy_mag = self.data_input[:, self.hps.num_context_window \
                * input_dim:(self.hps.num_context_window + 1) * input_dim]
        enhanced_mag = noisy_mag * est_mask
        loss = tf.reduce_mean(tf.square(enhanced_mag - self.data_refer))

        self.error = loss
        #Define optimizer
        self.define_optimization(loss)
        self.decode = tf.cast(enhanced_mag, dtype = tf.float32)

    def define_direct_network(self):        
        # Define output function and loss
        last_hidden_layer = self.feed_layers(self.data_input)
        estimation = tf.add(tf.matmul(last_hidden_layer, self.weight[-1]), self.bias[-1])
        loss = tf.reduce_mean(tf.square(estimation - self.data_refer))
        self.error = loss

        # Define optimizer.
        self.define_optimization(loss)
        self.decode = tf.cast(estimation, dtype=tf.float32)

    def define_irm_network(self):        
        # Define output function and loss
        last_hidden_layer = self.feed_layers(self.data_input)
        est_mask = tf.sigmoid(tf.add(tf.matmul(last_hidden_layer, self.weight[-1]), self.bias[-1]))
        loss = tf.reduce_mean(tf.square(est_mask - self.data_refer))
        self.error = loss

        # Define optimizer.
        self.define_optimization(loss)
        self.decode = tf.cast(est_mask, dtype=tf.float32)

    def define_ibm_network(self):        
        # Define output function and loss
        last_hidden_layer = self.feed_layers(self.data_input)
        est_mask = tf.sigmoid(tf.add(tf.matmul(last_hidden_layer, self.weight[-1]), self.bias[-1]))
        loss = tf.reduce_mean(tf.square(est_mask - self.data_refer))
        self.error = loss

        # Define optimizer.
        self.define_optimization(loss)
        self.decode = tf.cast(est_mask, dtype=tf.float32)

    def train(self, sess, train_input, train_refer, devel_input, devel_refer):

        # Train network.
        print('Get statistical parameter...')
        mu_input, sig_input, mu_refer, sig_refer \
            = get_statistics(train_input, train_refer)

        # Save statistical parameter.
        print('Save statistical parameter...')
        np.save(self.dir_to_save + '/mu_input.npy', mu_input)
        np.save(self.dir_to_save + '/sig_input.npy', sig_input)
        np.save(self.dir_to_save + '/mu_refer.npy', mu_refer)
        np.save(self.dir_to_save + '/sig_refer.npy', sig_refer)

        if self.hps.mode in ['spectrum', 'direct']:
            # Normalize batch data.
            print('Normalize batch data...')
            train_input = normalize_batch(train_input, mu_input, sig_input)
            train_refer = normalize_batch(train_refer, mu_refer, sig_refer)
            devel_input = normalize_batch(devel_input, mu_input, sig_input)
            devel_refer = normalize_batch(devel_refer, mu_refer, sig_refer)

        elif self.hps.mode in ['irm', 'ibm']:
            # Normalize batch data.
            print('Normalize batch data...')
            train_input = normalize_batch(train_input, mu_input, sig_input)
            devel_input = normalize_batch(devel_input, mu_input, sig_input)

        # Make context window data
        print('Make context window data...')
        train_input = make_context_data(train_input, 
                                        self.hps.num_context_window)
        devel_input = make_context_data(devel_input, 
                                        self.hps.num_context_window)

        # Start training.
        num_utt_train = len(train_input)
        num_utt_devel = len(devel_input)
        idx_set = range(len(train_input))

        # Write log file.
        fp = open(self.dir_to_save + '/log.txt', 'w')
        self.write_status_to_log_file(fp)

        print('Learning...')
        mse_devel_total = np.zeros(self.hps.max_epochs + 1)
        for i in range(self.hps.max_epochs + 1):
            start_time = time.time()

            # Train data.
            if i > 0:
                idx_set = np.random.permutation(idx_set)
                mse_train_arr = np.zeros(num_utt_train)
                for j in range(num_utt_train):
                    idxs = idx_set[j]
                    inp, ref = train_input[idxs], train_refer[idxs]
                    self.run_train_step(sess, inp, ref)
                    mse_train_arr[j] = self.run_eval_step(sess, inp, ref)
                mse_train = np.mean(mse_train_arr)
            else:
                mse_train = 0

            # Evaluate data.
            ptr = 0
            mse_devel_arr = np.zeros(num_utt_devel)
            for j in range(num_utt_devel):
                inp, ref = devel_input[j], devel_refer[j]
                mse_devel_arr[j] = self.run_eval_step(sess, inp, ref)
            mse_devel = np.mean(mse_devel_arr)
            mse_devel_total[i] = mse_devel

            # Print progress.
            print('step %d:  %.6f  %.6f  takes %.2f seconds' % 
                        (i, mse_train, mse_devel, time.time() - start_time))
            fp.write('step %d:  %.6f  %.6f  takes %.2f seconds\n' % 
                        (i, mse_train, mse_devel, time.time() - start_time))
            # Save models.
            self.export_model(sess, '%s/%d.ckpt' % (self.dir_to_save, i))

        fp.close()
        print('Training is finished...')

        # Copy optimum model that has minimum MSE.
        print('Save optimum models...')
        min_index = np.argmin(mse_devel_total)
        os.system('cp %s/%d.ckpt.data-00000-of-00001 \
                        %s/opt.ckpt.data-00000-of-00001' %
                        (self.dir_to_save, min_index, self.dir_to_save))
        os.system('cp %s/%d.ckpt.index %s/opt.ckpt.index' %
                        (self.dir_to_save, min_index, self.dir_to_save))
        os.system('cp %s/%d.ckpt.meta %s/opt.ckpt.meta' %
                        (self.dir_to_save, min_index, self.dir_to_save))

    def write_status_to_log_file(self, fp):
        # Write log file.            

        fp.write('%d-%d-%d %d:%d:%d\n' %
                (time.localtime().tm_year, time.localtime().tm_mon,
                 time.localtime().tm_mday, time.localtime().tm_hour,
                 time.localtime().tm_min, time.localtime().tm_sec))
        fp.write('mode                : %s\n' % self.hps.mode)
        fp.write('dnn struct          : %s\n' % self.hps.dnn_struct)
        fp.write('learning rate       : %g\n' % self.hps.learning_rate)
        fp.write('context window size : %g\n' % self.hps.num_context_window)
        fp.write('total params   : %d (%.2f M, %.2f MBytes)\n' %
                                        (self.total_params,
                                         self.total_params / 1000000.0,
                                         self.total_params * 4.0 / 1000000.0))
