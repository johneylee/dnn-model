import numpy as np
import scipy.io.wavfile as wav
import soundfile as sf
import time
import os
import random
import subprocess
import sys
import pdb
import librosa

import sys
import util.ssplib as ssplib
import util.minselib as se

def dotproduct(y, y_hat):
    return tf.reduce_sum(tf.multiply(tf.squeeze(y), tf.squeeze(y_hat)), 1, keep_dims=True )


def weighted_sdr_loss_1ch(output, clean, noisy):
    """
    from equation 9 in https://openreview.net/pdf?id=SkeRTsAcYm :
    x = noisy, y = clean, y_hat = output    
    """
    y = clean
    y_hat = output
    z = noisy - clean
    z_hat = noisy - output
    
    y_norm = tf.norm(tf.squeeze(y), axis = 1, keep_dims= True)
    z_norm = tf.norm(tf.squeeze(z), axis = 1, keep_dims= True)
    y_hat_norm = tf.norm(tf.squeeze(y_hat), axis = 1, keep_dims= True)
    z_hat_norm = tf.norm(tf.squeeze(z_hat), axis = 1, keep_dims= True)
    
    def loss_sdr(a, a_hat, a_norm, a_hat_norm):
        return dotproduct(a, a_hat) / (a_norm*a_hat_norm + 1e-8)
        
    alpha = tf.square(y_norm) / (tf.square(y_norm) + tf.square(z_norm) + 1e-8)
    loss_wSDR = -alpha * loss_sdr(y, y_hat, y_norm, y_hat_norm) - (1 - alpha) * loss_sdr(z, z_hat, z_norm, z_hat_norm)

    return tf.reduce_mean(loss_wSDR)

