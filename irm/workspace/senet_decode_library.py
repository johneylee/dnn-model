import os
import time
import config as cfg
import numpy as np
import tensorflow as tf
from tools_for_network import *

class SpeechEnhancementNetwork:
    def __init__(self):
        print('***********************************************************')
        print('*    Python library for DNN-based speech enhancement      *')
        print('*                        using Google\'s TensorFlow API    *')
        print('***********************************************************')
        self.decode = None
        self.data_input = None
        self.data_refer = None
        self.saver_import = None
        self.mu_input = None
        self.mu_refer = None
        self.sig_input = None
        self.sig_refer = None
        return

    @staticmethod
    def check_model_file_existence(model_name):
        if os.path.isfile('%s/opt.ckpt.meta' % model_name) is False:
            print("[Error] There is no model file '%s'." % model_name)
            exit()
        if os.path.isfile('%s/mu_input.npy' % model_name) is False:
            print("[Error] There is no model file '%s'." % model_name)
            exit()
        if os.path.isfile('%s/mu_refer.npy' % model_name) is False:
            print("[Error] There is no model file '%s'." % model_name)
            exit()
        if os.path.isfile('%s/sig_input.npy' % model_name) is False:
            print("[Error] There is no model file '%s'." % model_name)
            exit()
        if os.path.isfile('%s/sig_refer.npy' % model_name) is False:
            print("[Error] There is no model file '%s'." % model_name)
            exit()

    def import_model(self, sess, model_name):
        # Load tensorflow model.
            
        self.check_model_file_existence(model_name)
        addr_model = '%s/opt.ckpt' % model_name
        self.saver_import \
            = tf.train.import_meta_graph(addr_model + '.meta',
                                         clear_devices=True)
        self.saver_import.restore(sess, addr_model)
        self.decode = tf.get_collection("decode")[0]
        self.data_input = tf.get_collection('data_input')[0]
        self.data_refer = tf.get_collection('data_refer')[0]
        self.mu_input = np.load('./%s/mu_input.npy' % model_name)
        self.sig_input = np.load('./%s/sig_input.npy' % model_name)
        self.mu_refer = np.load('./%s/mu_refer.npy' % model_name)
        self.sig_refer = np.load('./%s/sig_refer.npy' % model_name)

    def run_decode_step(self, sess, inp, mode):
        
        # Get output from network.
        inp_norm = (inp - self.mu_input) / self.sig_input
        data = []
        data.append(inp_norm)
        data = make_context_data(data, cfg.num_context_window)
        estimation = sess.run(self.decode, 
                                   feed_dict={self.data_input: data[0]})
        if mode == 'ibm':
            return np.round(estimation)
        elif mode == 'irm':
            return estimation
        elif mode == 'direct':
            estimation_denorm = estimation * self.sig_refer + self.mu_refer
            return estimation_denorm
        elif mode == 'spectrum':
            estimation_denorm = estimation * self.sig_refer + self.mu_refer
            return estimation_denorm
