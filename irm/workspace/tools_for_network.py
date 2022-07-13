import os
import sys
import time
import pickle
import config as cfg
import numpy as np

def get_structure_params(input_dim=257, output_dim=257, layer_size=512, 
                         num_layer=3, num_context_window=2):
    dnn_struct = [input_dim * (num_context_window * 2 + 1)]
    for i in range(num_layer):
        dnn_struct.append(layer_size)
    dnn_struct.append(output_dim)
    return dnn_struct

def get_global_mu_sig(data):
    """Compute global mean and standard deviation of input data.

    Args:
        data: the number of element in list is the number of uttr
              element of list is (#frame) by (dim: freq bin)

    Returns:
        mu: global mean
        sig: global standard deviation
    """
    # Initialize array.
    start_time = time.time()
    num_utts = len(data)
    mu_utt = np.zeros(num_utts, dtype=np.float32)
    tmp_utt = np.zeros(num_utts, dtype=np.float32)
    num_utt = np.zeros(num_utts, dtype=np.float32)

    # Get mean.
    for n in range(num_utts):
        mu_utt[n] = np.mean(data[n])
        num_utt[n] = data[n].shape[0] * data[n].shape[1]
    mu = (1.0 / np.sum(num_utt)) * np.sum(num_utt * mu_utt)

    # Get standard deviation.
    for n in range(num_utts):
        tmp_utt[n] = np.mean(np.square(data[n] - mu))
    sig2 = (1.0 / np.sum(num_utt)) * np.sum(num_utt * tmp_utt)
    sig = np.sqrt(sig2)
    # print('  mu = %.4f sig = %.4f  takes %.2f second' %
    #       (mu, sig, time.time() - start_time))
    return np.float16(mu), np.float16(sig)


def get_global_mu_sig_partial(data, idx):
    """Compute partial global mean and standard deviation of input data.

    Args:
        data: the number of element in list is the number of uttr.
              element of list is (#frame) by (dim: freq bin) 
        idx: part of index to want to get (index of column)

    Returns:
        mu: global mean of part to get
        sig: global standard deviation to get
    """
    # Initialize array.
    start_time = time.time()
    num_utts = len(data)
    mu_utt = np.zeros(num_utts, dtype=np.float32)
    tmp_utt = np.zeros(num_utts, dtype=np.float32)
    num_utt = np.zeros(num_utts, dtype=np.float32)

    # Get mean.
    for n in range(num_utts):
        mu_utt[n] = np.mean(data[n][:, idx])
        num_utt[n] = data[n][:, idx].shape[0] * data[n][:, idx].shape[1]
    mu = (1.0 / np.sum(num_utt)) * np.sum(num_utt * mu_utt)

    # Get standard deviation.
    for n in range(num_utts):
        tmp_utt[n] = np.mean(np.square(data[n][:, idx] - mu))
    sig2 = (1.0 / np.sum(num_utt)) * np.sum(num_utt * tmp_utt)
    sig = np.sqrt(sig2)
    # print('  mu = %.4f sig = %.4f  takes %.2f second' %
    #       (mu, sig, time.time() - start_time))
    return np.float16(mu), np.float16(sig)


def get_mu_sig(data):
    """Compute mean and standard deviation vector of input data

    Args:
        data: the number of element in list is the number of uttr.
              element of list is (#frame) by (dim: freq bin) 

    Returns:
        mu: mean vector (#dim by one)
        sig: standard deviation vector (#dim by one)
    """
    # Initialize array.
    num_utts = len(data)
    dim = data[0].shape[1]
    mu_utt = np.zeros((num_utts, dim), dtype=np.float32)
    tmp_utt = np.zeros((num_utts, dim), dtype=np.float32)
    num_utt = np.zeros((num_utts, dim), dtype=np.float32)

    # Get mean.
    for n in range(num_utts):
        mu_utt[n, :] = np.mean(data[n], 0)
        num_utt[n, :] = data[n].shape[0]
    sum_tmp = np.sum(num_utt[:, 0])
    mu = np.sum(num_utt * mu_utt / sum_tmp, 0)

    # Get standard deviation.
    for n in range(num_utts):
        tmp_utt[n, :] = np.mean(np.square(data[n] - mu), 0)
    sig2 = np.sum(num_utt * tmp_utt / sum_tmp, 0)
    sig = np.sqrt(sig2)

    # Assign unit variance.
    for n in range(len(sig)):
        if sig[n] < 1e-5:
            sig[n] = 1.0
    return np.float16(mu), np.float16(sig)


def normalize(data, mu, sigma):
    return (data - mu) / sigma

def get_statistics(inp, refer):
    """Get statistical parameter of input data.

    Args:
        inp: input data
        refer: reference data

    Returns:
        mu_inp: mean vector of input data
        sig_inp: standard deviation vector of input data
        mu_refer: mean vector of reference data
        sig_refer: standard defviation vector of reference data
    """
    dim = 257
    win_samp = (dim - 1) * 2
    mu_inp, sig_inp = get_mu_sig(inp)
    mu_refer1, sig_refer1 \
        = get_global_mu_sig_partial(refer, range(0, dim))

    mu_refer = mu_refer1 * np.ones(dim)
    sig_refer = sig_refer1 * np.ones(dim)
    return mu_inp, sig_inp, mu_refer, sig_refer

def create_folder(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def open_data(*arg):
    """Open numpy data from directory.

    Args:
        arg: file names

    Returns:
        data: data files
    """
    data = []
    for i in range(len(arg)):
        if os.path.exists(arg[i]):
            with open(arg[i], 'rb') as handle:
                data.append((pickle.load(handle)))
        else:
            print("[Error] There is no data file '%s'" % arg[i])
            exit()
    return data

def normalize_batch(data, mu, sigma):
    """Normalize data and get length (#frame of utterence) of each data.

    Args:
        data: the data to normalize
        mu: the mean to normalize
        sig: the standard deviation to normalize

    Returns:
        data: normalized data
        data_length: length vector of each data
    """
    num_samps = len(data)
    for i in range(num_samps):
        data[i] = normalize(data[i], mu, sigma)
    return data

def make_context_data(data, L):

    for k in range(len(data)):
        context_data = data[k]
        for i in range(1, L+1):
            previous_data = np.delete(data[k], np.s_[-i::1], 0)
            future_data  = np.delete(data[k], np.s_[0:i:1], 0)

            start_frame = np.array([data[k][0,:]])
            last_frame  = np.array([data[k][-1,:]])

            dusf = start_frame
            dulf = last_frame

            for j in range(i-1):
                dusf = np.concatenate((dusf,start_frame))
                dulf = np.concatenate((dulf,last_frame))
            previous_data = np.concatenate((dusf, previous_data))
            future_data  = np.concatenate((future_data, dulf))

            context_data = np.concatenate((context_data, previous_data), 1)
            context_data = np.concatenate((context_data, future_data), 1)
        data[k] = context_data

    return data