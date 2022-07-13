#!/usr/bin/python

"""
"""

import numpy as np
import tensorflow as tf
from collections import namedtuple
import pdb

hparams = namedtuple('hparams',
                     'batch_size, '
                     'buffer_size, '
                     'num_parallel_reads, '
                     'num_parallel_calls')


class DataPreparation:
    """Define class of the speech enhancement network for training.

    Members:

    """

    def __init__(self):
        self.dhps = None
        self.data_inp = None
        self.data_ref = None
        self.handle = None
        self.feature_info = None
        self.iterator = None
        self.num_batches = None
        self.handle = None
        self.mu_inp = None
        self.sig_inp = None
        self.mu_ref = None
        self.sig_ref = None
        return

    def initialize(self, dhps,
                   featype_inp_list, dim_idx_inp,
                   featype_ref_list, dim_idx_ref):
        # Set hyperparameters
        self.dhps = dhps
        self.feature_info = namedtuple('Container', ['inp', 'ref'])
        self.feature_info.inp = namedtuple('Container',
                                           ['fea_type', 'dim_idx', 'dim'])
        self.feature_info.ref = namedtuple('Container',
                                           ['fea_type', 'dim_idx', 'dim'])
        self.feature_info.inp.fea_type = featype_inp_list
        self.feature_info.ref.fea_type = featype_ref_list
        self.feature_info.inp.dim_idx = dim_idx_inp
        self.feature_info.ref.dim_idx = dim_idx_ref
        self.feature_info.inp.dim = np.sum(self.feature_info.inp.dim_idx)
        self.feature_info.ref.dim = np.sum(self.feature_info.ref.dim_idx)

    def load_features(self, sess, filelist, flag_shuffle, flag_norm):
        # Set batch size for training and evaluating.
        num_utts = len(filelist)
        self.num_batches = int(num_utts / self.dhps.batch_size)
        self.handle \
            = self.generate_iterator_handles(sess, filelist,
                                             flag_shuffle=flag_shuffle)
        if flag_norm == 1:
            self.calculate_normalization_parameters(sess, filelist)

    def generate_iterator_handles(self, sess, filelist, flag_shuffle=0):
        buffer_size = self.dhps.buffer_size
        num_parallel_calls = self.dhps.num_parallel_calls
        num_parallel_reads = self.dhps.num_parallel_reads
        keys_to_features = {
            'inp': tf.FixedLenSequenceFeature([self.feature_info.inp.dim],
                                              dtype=tf.float32,
                                              allow_missing=True),
            'ref': tf.FixedLenSequenceFeature([self.feature_info.ref.dim],
                                              dtype=tf.float32,
                                              allow_missing=True)}
        # Generates training data
        dataset = tf.data.TFRecordDataset(filelist,
                                          num_parallel_reads=num_parallel_reads).repeat()
        dataset = dataset.map(
            lambda record: tf.parse_single_example(record, keys_to_features),
            num_parallel_calls=64)
        dataset \
            = dataset.map(lambda features: [features['inp'],
                                            features['ref']],
                          num_parallel_calls=num_parallel_calls)
        if flag_shuffle == 1:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset \
            = dataset.apply(tf.contrib.data.bucket_by_sequence_length
                                  (element_length_func=self.length_fn,
                                   bucket_batch_sizes=[self.dhps.batch_size],
                                   bucket_boundaries=[])).prefetch(2)
        iterator = dataset.make_one_shot_iterator()
        handle = sess.run(iterator.string_handle())

        return handle

    def calculate_normalization_parameters(self, sess, filenames):
        num_parallel_calls = self.dhps.num_parallel_calls
        num_parallel_reads = self.dhps.num_parallel_reads
        dataset_stats = tf.data.TFRecordDataset(filenames,
                                                num_parallel_reads=num_parallel_reads).repeat()
        keys_to_features = {
            'inp': tf.FixedLenSequenceFeature([self.feature_info.inp.dim],
                                              dtype=tf.float32,
                                              allow_missing=True),
            'ref': tf.FixedLenSequenceFeature([self.feature_info.ref.dim],
                                              dtype=tf.float32,
                                              allow_missing=True)}
        dataset_stats = dataset_stats.map(
            lambda record: tf.parse_single_example(record, keys_to_features),
            num_parallel_calls=num_parallel_calls)
        dataset_stats = dataset_stats.map(
            lambda features: [features['inp'], features['ref']],
            num_parallel_calls=num_parallel_calls).prefetch(2)
        iterator_stats = dataset_stats.make_one_shot_iterator()
        num_utts = len(filenames)
        inp_utt, ref_utt = iterator_stats.get_next()

        di = self.feature_info.inp.dim
        ref_dim_idx = self.feature_info.ref.dim_idx
        do = len(ref_dim_idx)
        len_buf = np.zeros([num_utts])
        mu_inp_buf = np.zeros([num_utts, di])
        squared_mu_inp_buf = np.zeros([num_utts, di])
        mu_ref_buf = np.zeros([num_utts, do])
        squared_mu_ref_buf = np.zeros([num_utts, do])
        for idx in range(num_utts):
            inp, ref = sess.run([inp_utt, ref_utt])
            inp_np = np.array(inp)
            ref_np = np.array(ref)
            len_utt, mu_inp_utt, squared_mu_inp_utt, \
            mu_ref_utt, squared_mu_ref_utt \
                = self.calculate_utterance_level_statistics(inp_np, ref_np,
                                                            ref_dim_idx)
            len_buf[idx] = len_utt
            mu_inp_buf[idx, :] = mu_inp_utt
            squared_mu_inp_buf[idx, :] = squared_mu_inp_utt
            mu_ref_buf[idx, :] = mu_ref_utt
            squared_mu_ref_buf[idx, :] = squared_mu_ref_utt

        mu_inp, sig_inp \
            = self.get_mu_sig(mu_inp_buf, squared_mu_inp_buf, len_buf)
        mu_ref_tmp, sig_ref_tmp \
            = self.get_mu_sig(mu_ref_buf, squared_mu_ref_buf, len_buf)

        print('[mu (reference)]')
        print(mu_ref_tmp)
        print('[sigma (reference)]')
        print(sig_ref_tmp)
	
        mu_ref = []
        sig_ref = []
        for idx_fea in range(len(self.feature_info.ref.dim_idx)):
            tmp = mu_ref_tmp[idx_fea] * np.ones(
                self.feature_info.ref.dim_idx[idx_fea])

            mu_ref \
                = np.concatenate([mu_ref, tmp], 0)
            tmp = sig_ref_tmp[idx_fea] * np.ones(
                self.feature_info.ref.dim_idx[idx_fea])
            sig_ref \
                = np.concatenate([sig_ref, tmp], 0)

        self.mu_inp = np.float32(mu_inp)
        self.sig_inp = np.float32(sig_inp)
        self.mu_ref = np.float32(mu_ref)
        self.sig_ref = np.float32(sig_ref)

    @staticmethod
    def length_fn(x, y):
        return tf.shape(x)[0]

    @staticmethod
    def calculate_utterance_level_statistics(inp, ref, idx_ref):
        # input
        mu_inp = np.mean(inp, 0)
        squared_mu_inp = np.mean(inp * inp, 0)
        numfrms = inp.shape[0]
        # reference
        ed = 0
        mu_ref = np.zeros([len(idx_ref)])
        squared_mu_ref = np.zeros([len(idx_ref)])
        for idx_fea in range(len(idx_ref)):
            st = ed
            ed = int(st + idx_ref[idx_fea])
            mu_ref[idx_fea] = np.mean(ref[:, st:ed])
            squared_mu_ref[idx_fea] \
                = np.mean(ref[:, st:ed] * ref[:, st:ed])
        return numfrms, mu_inp, squared_mu_inp, mu_ref, squared_mu_ref

    @staticmethod
    def get_mu_sig(mu_utt, sqmu_utt, len_utt):
        mu_utt = np.array(mu_utt)
        sqmu_utt = np.array(sqmu_utt)
        num_utts = mu_utt.shape[0]
        len_utt = np.array(len_utt)
        len_utt_mat = np.zeros(mu_utt.shape)
        for idx in range(num_utts):
            len_utt_mat[idx, :] = len_utt[idx]
        mu = (1.0 / np.sum(len_utt_mat, 0)) * np.sum(len_utt_mat * mu_utt, 0)
        sig2 = (1.0 / np.sum(len_utt_mat, 0)) * np.sum(
            len_utt_mat * (sqmu_utt - (mu * mu)), 0)
        sig = np.sqrt(sig2)
        return mu, sig
