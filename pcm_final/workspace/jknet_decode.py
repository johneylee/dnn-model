#!/usr/bin/python

"""
Speech enhancement neural network libraries
It contains functions of network using tensorflow.
Functions include define parameters of network, save and load network model, 
define RNN/fully connected layer, feed layers, define speech enhancement 
network and train/evaluate network.
"""
import os
import numpy as np
import ssplib as ssp
import tensorflow as tf
from tensorflow.contrib import rnn


class SpeechEnhancementNetwork:
    """Define class of the speech enhancement network for decoding.

    Members:
        hps: hyperparameters of network
        decode: output of network
        err: mean square of estimated data and reference
        optimization: optimizer of network
        data_input: input data of network
        data_reference: reference data of network
        blstm_size: size of BLSTM layer
        length: sequence length vector (the number of frame of each uttrence)
        mask: Binary tensor mask whose components are 0 for padded data and 1 
              for not padded data (Auxiliary tensor for not calculating mse 
              for zero padded data)
        saver_import: tensorflow saver for loading
        saver_export: tensorflow saver for saving
        rnn: RNN layer
        fc: fully connected layer
        mu_input: mean of input data
        mu_reference: mean of reference data
        sig_input: standard deviation of input data
        sig_reference: standard deviation of reference data
        total_params: the total number of parameters of network
        dir_to_save: directory name for save
    """

    def __init__(self):
        print('***********************************************************')
        print('*    Python library for BLSTM-based speech enhancement    *')
        print('*                        using Google\'s TensorFlow API    *')
        print('***********************************************************')
        self.decode = None
        self.data_inp = None
        self.data_ref = None
        self.dim_inp = None
        self.dim_ref = None
        self.batch_size = None
        self.length = None
        self.saver_import = None
        self.mu_inp = None
        self.mu_ref = None
        self.sig_inp = None
        self.sig_ref = None
        return

    @staticmethod
    def check_model_file_existence(model_name):
        if os.path.isfile('%s/opt.ckpt.meta' % model_name) is False:
            print("[Error] There is no model file '%s'." % model_name)
            exit()
        if os.path.isfile('%s/mu_inp.npy' % model_name) is False:
            print("[Error] There is no model file '%s'." % model_name)
            exit()
        if os.path.isfile('%s/mu_ref.npy' % model_name) is False:
            print("[Error] There is no model file '%s'." % model_name)
            exit()
        if os.path.isfile('%s/sig_inp.npy' % model_name) is False:
            print("[Error] There is no model file '%s'." % model_name)
            exit()
        if os.path.isfile('%s/sig_ref.npy' % model_name) is False:
            print("[Error] There is no model file '%s'." % model_name)
            exit()

    def import_model(self, sess, model_name):

        self.check_model_file_existence(model_name)
        addr_model = '%s/opt.ckpt' % model_name
        self.saver_import \
            = tf.train.import_meta_graph(addr_model + '.meta')
        self.saver_import.restore(sess, addr_model)
        self.decode = tf.get_collection("decode")[0]
        self.data_inp = tf.get_collection('data_inp')[0]
        self.data_ref = tf.get_collection('data_ref')[0]
        self.mu_inp = np.load('./%s/mu_inp.npy' % model_name)
        self.sig_inp = np.load('./%s/sig_inp.npy' % model_name)
        self.mu_ref = np.load('./%s/mu_ref.npy' % model_name)
        self.sig_ref = np.load('./%s/sig_ref.npy' % model_name)

    def run_decode_batch_step(self, sess, buf,
                              window_analysis,
                              window_synthesis,
                              length_wav, ratio):

        seqlen = self.get_length_np(buf)
        inp, mask = self.make_data_for_batch(buf, seqlen)
        batch_size = inp.shape[0]
        spec_concat_est = sess.run(self.decode,
                                   feed_dict={self.data_inp: inp})
        dim = spec_concat_est.shape[-1] / 2
        spec_est_half \
            = spec_concat_est[:, :, 0:dim] + 1j * spec_concat_est[:, :, dim:]

        enhanced = []
        for n in range(batch_size):
            tmp = ssp.spec_half2wav(spec_est_half[n],
                                    window_analysis,
                                    window_synthesis,
                                    length_wav[n], ratio)
            tmp_int = tmp * (2 ** 15)
            tmp_int = tmp_int.astype(np.int16)
            enhanced.append(tmp_int)
        return enhanced

    def make_data_for_batch(self, data, length):
        """Make zero-padded data and binary mask.
        """
        max_len = np.max(length)
        data1 = self.zero_padding(data, max_len)
        mask = self.length_vector2mask(length)
        return data1, mask

    @staticmethod
    def zero_padding(list_mat, max_frmlen):
        """Zero-pad batch data by maximum frame length of total data.

        Args:
            list_mat: batch data
            max_frmlen: maximum frame length of total data

        Returns:
            x_pad: zero-padded batch data
                   (batch size X maximum frame length X dim)
        """
        num_batches = len(list_mat)
        dim = list_mat[0].shape[1]
        x_pad = np.zeros((num_batches, max_frmlen, dim), dtype='float32')
        for batch_idx in range(num_batches):
            l_tmp = list_mat[batch_idx].shape[0]
            x_zero = np.zeros((max_frmlen - l_tmp, dim), dtype='float32')
            x_tmp = np.concatenate((list_mat[batch_idx], x_zero), 0)
            x_pad[batch_idx] = x_tmp
        return x_pad

    @staticmethod
    def length_vector2mask(length):
        """Convert length vector to binary mask.

        Args:
            length: length vector which has #frame of each utterance in batch
                    (# batches size by 1)

        Returns:
            mask: binary mask matrix (batch size X max length of length vector)
        """
        maxl = np.max(length)
        mask = np.ones((len(length), maxl))
        for idx in range(len(length)):
            mask[idx, length[idx]:] = 0
        for idx in range(len(length)):
            mask[idx, :] = mask[idx, :] / length[idx]
        return mask

    @staticmethod
    def convert_magpha2concatenated_spec(mag, pha):
        """Convert magnitude and phase to complex spectrum.

        Args:
            mag: magnitude
            pha: phase

        Returns:
            spec_concat: concatenated data [real part, imaginary part]
            spec_real: real part of complex data
            spec_imag: imaginary part of complex data
        """
        spec_real = mag * tf.cos(pha)
        spec_imag = mag * tf.sin(pha)
        spec_concat = tf.concat([spec_real, spec_imag], 2)
        return spec_concat, spec_real, spec_imag

    @staticmethod
    def get_length_np(data):
        """Get length of numpy list data

        Args:
            data: numpy list data

        Returns:
            data_length: length vector
        """
        num_samps = len(data)
        data_length = np.zeros(num_samps, dtype='int32')
        for i in range(num_samps):
            data_length[i] = np.int32(data[i].shape[0])
        return data_length
