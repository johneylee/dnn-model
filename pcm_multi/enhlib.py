#!/usr/bin/python

"""
"""

import tensorflow as tf
import numpy as np
import ssplib as ssp

def define_pcm_mag(fc_out):
    dim = 514
    fc_out1 = tf.slice(fc_out, [0, 0, 0 * dim], [-1, -1, dim])
    fc_out2 = tf.slice(fc_out, [0, 0, 1 * dim], [-1, -1, dim])
    fc_out3 = tf.slice(fc_out, [0, 0, 2 * dim], [-1, -1, dim])

    # Clean speech magnitude
    est_mag_warp_clean_vnorm = tf.nn.relu(fc_out1)
    # Noise magnitude
    est_mag_warp_noise_vnorm = tf.nn.relu(fc_out2)
    # Phase difference of speech to noise spectra
    est_cos_xn = 0.995 * tf.nn.tanh(fc_out3)
    
    return est_mag_warp_clean_vnorm, est_mag_warp_noise_vnorm, est_cos_xn


def define_obj_etdr(ref, est_spec_real, est_spec_imag,
                    dim, seq_length, rnn_mask):
    # TDR with normalization
    frm_size = 257

    numfrm_max = tf.reduce_max(seq_length)
    batch_size = tf.cast(tf.reduce_sum(tf.sign(seq_length)), tf.int32)
    rdftmat_real, rdftmat_imag, irdftmat_real, irdftmat_imag \
        = calculate_rdft_irdft_matrices(batch_size, frm_size)

    est_frm_hann_clean \
       = tf.matmul(est_spec_real, irdftmat_real) - tf.matmul(
       est_spec_imag, irdftmat_imag)

    est_frm_rect_clean \
       = overlap_and_add(est_frm_hann_clean, batch_size,
                         numfrm_max, frm_size)

    sig_max = tf.reduce_max(ref['sig_max'].data, axis=[1])
    sig_max = tf.expand_dims(sig_max, [1])
    est_frm_rect_norm_clean = est_frm_rect_clean / sig_max
    est_frm_rect_norm_clean_vnorm \
        = est_frm_rect_norm_clean / ref['frm_rect_norm_clean'].sig[0:128]

    obj = masked_mse(est_frm_rect_norm_clean_vnorm,
                     ref['frm_rect_norm_clean'].data_vnorm[:,:,0:128],
                     rnn_mask)
    return obj


def define_obj_params(ref, est_mag_warp_clean_vnorm, est_mag_warp_noise_vnorm,
                      est_cos_xn, est_cos_xy, est_sin_xy, rnn_mask):
    # Params
    alpha = 2.0 / 3.0
    sig_max = tf.reduce_max(ref['sig_max'].data, axis=[1])
    sig_max = tf.expand_dims(sig_max, [1])
    sig_max_warp = tf.pow(sig_max, alpha)

    est_mag_norm_warp_clean_vnorm \
        = est_mag_warp_clean_vnorm / sig_max_warp
    est_mag_norm_warp_noise_vnorm \
        = est_mag_warp_noise_vnorm / sig_max_warp
    est_cos_xn_znorm = (est_cos_xn - ref['cos_xn'].mu) / ref['cos_xn'].sig
    est_cos_xy_znorm = (est_cos_xy - ref['cos_xy'].mu) / ref['cos_xy'].sig
    est_sin_xy_znorm = (est_sin_xy - ref['sin_xy'].mu) / ref['sin_xy'].sig

    obj_p1 = masked_mse(est_mag_norm_warp_clean_vnorm,
                        ref['mag_norm_warp_clean'].data_vnorm,
                        rnn_mask)
    obj_p2 = masked_mse(est_mag_norm_warp_noise_vnorm,
                        ref['mag_norm_warp_noise'].data_vnorm,
                        rnn_mask)
    obj_p3 = masked_mse(est_cos_xn_znorm,
                        ref['cos_xn'].data_znorm, rnn_mask)
    obj_p4 = masked_mse(est_cos_xy_znorm,
                        ref['cos_xy'].data_znorm, rnn_mask)
    obj_p5 = masked_mse(est_sin_xy_znorm,
                        ref['sin_xy'].data_znorm, rnn_mask)
    obj = 0.2 * (obj_p1 + obj_p2 + obj_p3 + obj_p4 + obj_p5)

    return obj


def define_obj_sc(ref, est_mask_mag, noisy_mag, noisy_pha,
                  est_cos_xy, est_sin_xy, dim, seq_length, rnn_mask):

    alpha = 2.0 / 3.0

    dim = 257
    winsize = (dim - 1) * 2 # original dim = 514, but we must use 257
    win_np = np.sqrt(ssp.hann_window(winsize))
    wa = tf.cast(win_np, dtype=tf.float32)
    sig_max = tf.reduce_max(ref['sig_max'].data, axis=[1])
    sig_max = tf.expand_dims(sig_max, [1])
    numfrm_max = tf.reduce_max(seq_length)
    batch_size = tf.cast(tf.reduce_sum(tf.sign(seq_length)), tf.int32)
    rdftmat_real, rdftmat_imag, irdftmat_real, irdftmat_imag \
        = calculate_rdft_irdft_matrices(batch_size, dim)
    est_mask_mag_fixed = tf.stop_gradient(est_mask_mag)
    est_mask_real_fixed = est_mask_mag_fixed * est_cos_xy
    est_mask_imag_fixed = est_mask_mag_fixed * est_sin_xy

    # Compute t-f masking
    est_spec_real_fixed, est_spec_imag_fixed \
        = complex_domain_tf_masking(est_mask_real_fixed, est_mask_imag_fixed,
                                    noisy_mag, noisy_pha)

    est_mag_fixed = tf.sqrt(
        est_spec_real_fixed * est_spec_real_fixed
        + est_spec_imag_fixed * est_spec_imag_fixed + 1e-7)
    
    # Separate enhanced spectrum
    est_spec_real_fixed_mic0 = tf.matmul(est_spec_real_fixed[:, :, 0:257], irdftmat_real)
    est_spec_real_fixed_mic1 = tf.matmul(est_spec_real_fixed[:, :, 257:], irdftmat_real)
    est_spec_imag_fixed_mic0 = tf.matmul(est_spec_imag_fixed[:, :, 0:257], irdftmat_imag)
    est_spec_imag_fixed_mic1 = tf.matmul(est_spec_imag_fixed[:, :, 257:], irdftmat_imag)

    est_frm_win_fixed_mic0 = est_spec_real_fixed_mic0 - est_spec_imag_fixed_mic0
    est_frm_win_fixed_mic1 = est_spec_real_fixed_mic1 - est_spec_imag_fixed_mic1

    # Overlap and add process for each microphone
    est_frm_fixed_mic0 = overlap_and_add_full(est_frm_win_fixed_mic0, batch_size, numfrm_max, dim)
    est_frm_fixed_mic1 = overlap_and_add_full(est_frm_win_fixed_mic1, batch_size, numfrm_max, dim)

    # Get the reconstructed spectrum for each microphone
    est_respec_real_fixed_mic0 = tf.matmul(est_frm_fixed_mic0 * wa, rdftmat_real)
    est_respec_real_fixed_mic1 = tf.matmul(est_frm_fixed_mic1 * wa, rdftmat_real)
    est_respec_imag_fixed_mic0 = tf.matmul(est_frm_fixed_mic0 * wa, rdftmat_imag)
    est_respec_imag_fixed_mic1 = tf.matmul(est_frm_fixed_mic1 * wa, rdftmat_imag)
    
    # Concatenate two spectrum
    est_respec_real_fixed = tf.concat([est_respec_real_fixed_mic0, est_respec_real_fixed_mic1], -1)
    est_respec_imag_fixed = tf.concat([est_respec_imag_fixed_mic0, est_respec_imag_fixed_mic1], -1)

    est_remag_fixed = tf.sqrt(
        est_respec_real_fixed * est_respec_real_fixed
        + est_respec_imag_fixed * est_respec_imag_fixed + 1e-7)

    sc1 = tf.pow(est_mag_fixed / sig_max, alpha) / ref['mag_norm_warp_clean'].sig
    sc2 = tf.pow(est_remag_fixed / sig_max, alpha) / ref['mag_norm_warp_clean'].sig
    obj = masked_mse(sc1, sc2, rnn_mask)

    return obj


def parametric_complex_tfmask_function_mag(est_mag_warp_clean_vnorm,
                                           est_mag_warp_noise_vnorm,
                                           est_cos_xn, sig):
    """
    """
    alpha = 2.0 / 3.0
    est_mag_warp_clean \
        = est_mag_warp_clean_vnorm * sig
    est_mag_clean = tf.pow(est_mag_warp_clean, 1.0 / alpha) + 1e-7
    est_mag_warp_noise \
        = est_mag_warp_noise_vnorm * sig
    est_mag_noise = tf.pow(est_mag_warp_noise, 1.0 / alpha) + 1e-7
    est_mask_mag \
        = tf.sqrt(tf.pow(est_mag_clean, 2.0)
                  / (tf.pow(est_mag_clean, 2.0)
                     + tf.pow(est_mag_noise, 2.0)
                     + 2 * est_cos_xn * est_mag_clean * est_mag_noise))
    
    return est_mask_mag


def masked_mse(est, ref, rnn_mask):
    """Calculate mean squared error of estimated data and reference data.
    """
    square_error_ntf = tf.square(ref - est)
    square_error_nt = tf.reduce_mean(square_error_ntf, 2) * rnn_mask
    square_error_n = tf.reduce_sum(square_error_nt, 1) / tf.reduce_sum(
        rnn_mask, 1)
    mse = tf.reduce_mean(square_error_n, 0)
    return mse


def complex_domain_tf_masking(est_mask_real, est_mask_imag,
                              noisy_mag, noisy_pha):
    noisy_spec_real = noisy_mag * tf.cos(noisy_pha)
    noisy_spec_imag = noisy_mag * tf.sin(noisy_pha)
    est_spec_real = est_mask_real * noisy_spec_real \
                    - est_mask_imag * noisy_spec_imag
    est_spec_imag = est_mask_real * noisy_spec_imag \
                    + est_mask_imag * noisy_spec_real
    return est_spec_real, est_spec_imag


def calculate_rdft_irdft_matrices(batch_size, dim):
    """
    """
    win_samp = (dim - 1) * 2 # if dim = 257, win_samp = 512 / if dim = 514, win_samp = 1026
    dftmat = tf.spectral.fft(tf.eye(win_samp,
                             batch_shape=[batch_size],
                             dtype=tf.complex64))
    rdftmat = tf.spectral.rfft(tf.eye(win_samp,
                               batch_shape=[batch_size],
                               dtype=tf.float32))
    rdftmat_real = tf.real(rdftmat)
    rdftmat_imag = tf.imag(rdftmat)
    eye257 = tf.eye(dim, batch_shape=[batch_size], dtype=tf.float32)
    half2full_real = tf.concat([eye257, eye257[:, :, -2:0:-1]], 2)
    half2full_imag = tf.concat([eye257, -1.0 * eye257[:, :, -2:0:-1]], 2)
    idftmat_real = tf.real(dftmat) / win_samp
    idftmat_imag = -1.0 * tf.imag(dftmat) / win_samp
    irdftmat_real = tf.matmul(half2full_real, idftmat_real)
    irdftmat_imag = tf.matmul(half2full_imag, idftmat_imag)

    return rdftmat_real, rdftmat_imag, irdftmat_real, irdftmat_imag


def overlap_and_add(time_signal, batch_size, numfrms, dim):
    """
    """
    # Define window-related parameters
    winsize = (dim - 1) * 2
    incsize = int(winsize / 4)
    win_np = np.sqrt(ssp.hann_window(winsize))
    wa = tf.cast(win_np, dtype=tf.float32)
    ws = tf.cast(win_np, dtype=tf.float32)
    was = wa * ws
    comph = was[0:incsize] \
            + was[incsize:incsize * 2] \
            + was[incsize * 2:incsize * 3] \
            + was[incsize * 3:incsize * 4]
    time_signal_win = time_signal * ws

    # Calculate time domain shift matrix
    eye_time = tf.eye(numfrms + 3, batch_shape=[batch_size],
                      dtype=tf.float32)
    time_l3 = eye_time[:, :numfrms, 3:3 + numfrms]
    time_l2 = eye_time[:, :numfrms, 2:2 + numfrms]
    time_l1 = eye_time[:, :numfrms, 1:1 + numfrms]
    time_l0 = eye_time[:, :numfrms, 0:0 + numfrms]
   
    # Calculate frequency domain shift matrix
    eye_freq = tf.eye(winsize, batch_shape=[batch_size], dtype=tf.float32)
    freq_l3 = eye_freq[:, :, incsize*3:incsize*4]
    freq_l2 = eye_freq[:, :, incsize*2:incsize*3]
    freq_l1 = eye_freq[:, :, incsize:incsize*2]
    freq_l0 = eye_freq[:, :, 0:incsize]

    # Multiply pre-calculated shift matrix
    mat_l3 = tf.matmul(tf.matmul(time_l3, time_signal_win), freq_l3)

    mat_l2 = tf.matmul(tf.matmul(time_l2, time_signal_win), freq_l2)
    mat_l1 = tf.matmul(tf.matmul(time_l1, time_signal_win), freq_l1)
    mat_l0 = tf.matmul(tf.matmul(time_l0, time_signal_win), freq_l0)

    time_signal_ola = (mat_l3 + mat_l2 + mat_l1 + mat_l0) / comph

    return time_signal_ola


def overlap_and_add_full(mat, batch_size, numfrm, dim):
    # Define window-related parameters
    #dim = 257
    winsize = (dim - 1) * 2
    incsize = int(winsize / 4)
    win_np = np.sqrt(ssp.hann_window(winsize))
    wa = tf.cast(win_np, dtype=tf.float32)
    ws = tf.cast(win_np, dtype=tf.float32)
    was = wa * ws
    comph = was[0:incsize] \
            + was[incsize:incsize * 2] \
            + was[incsize * 2:incsize * 3] \
            + was[incsize * 3:incsize * 4]

    comph = tf.concat([comph, comph, comph, comph], 0)
    mat_win = mat * ws

    # Calculate time domain shift matrix
    eye_time = tf.eye(numfrm + 6, batch_shape=[batch_size],
                      dtype=tf.float32)
    time_l3 = eye_time[:, 0:0 + numfrm, 3:3 + numfrm]
    time_l2 = eye_time[:, 1:1 + numfrm, 3:3 + numfrm]
    time_l1 = eye_time[:, 2:2 + numfrm, 3:3 + numfrm]
    time_r1 = eye_time[:, 4:4 + numfrm, 3:3 + numfrm]
    time_r2 = eye_time[:, 5:5 + numfrm, 3:3 + numfrm]
    time_r3 = eye_time[:, 6:6 + numfrm, 3:3 + numfrm]

    # Calculate frequency domain shift matrix
    eye_freq = tf.eye(winsize * 3 - 2 * incsize, batch_shape=[batch_size],
                      dtype=tf.float32)
    sp_freq = winsize - incsize

    freq_l3 = eye_freq[:,
              sp_freq - 3 * incsize:sp_freq - 3 * incsize + winsize,
              sp_freq:sp_freq + winsize]    

    freq_l2 = eye_freq[:,
              sp_freq - 2 * incsize:sp_freq - 2 * incsize + winsize,
              sp_freq:sp_freq + winsize]
    freq_l1 = eye_freq[:,
              sp_freq - 1 * incsize:sp_freq - 1 * incsize + winsize,
              sp_freq:sp_freq + winsize]
    freq_r1 = eye_freq[:,
              sp_freq + 1 * incsize:sp_freq + 1 * incsize + winsize,
              sp_freq:sp_freq + winsize]
    freq_r2 = eye_freq[:,
              sp_freq + 2 * incsize:sp_freq + 2 * incsize + winsize,
              sp_freq:sp_freq + winsize]
    freq_r3 = eye_freq[:,
              sp_freq + 3 * incsize:sp_freq + 3 * incsize + winsize,
              sp_freq:sp_freq + winsize]

    # Multiply pre-calcuated shift matrix
    mat_l3 = tf.matmul(tf.matmul(time_l3, mat_win), freq_l3)
    mat_l2 = tf.matmul(tf.matmul(time_l2, mat_win), freq_l2)
    mat_l1 = tf.matmul(tf.matmul(time_l1, mat_win), freq_l1)
    mat_r1 = tf.matmul(tf.matmul(time_r1, mat_win), freq_r1)
    mat_r2 = tf.matmul(tf.matmul(time_r2, mat_win), freq_r2)
    mat_r3 = tf.matmul(tf.matmul(time_r3, mat_win), freq_r3)

    frm_syn = (mat_l3 + mat_l2 + mat_l1 + mat_win
               + mat_r1 + mat_r2 + mat_r3) / comph

    return frm_syn


