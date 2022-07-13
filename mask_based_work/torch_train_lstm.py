import os
import time
import numpy as np
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Data configuration
train_input_data_path = './feature_data/train_input_small'
train_refer_data_path = './feature_data/train_ref_irm_small'
devel_input_data_path = './feature_data/devel_input_small'
devel_refer_data_path = './feature_data/devel_ref_irm_small'

# open feature data
def open_data(*arg):        # input data load
    """Open numpy data from directory.

    Args:
        arg: file names

    Returns:
        data: data files
    """
    data = []
    for i in range(len(arg)):           # number of input data: 4
        if os.path.exists(arg[i]):
            with open(arg[i], 'rb') as handle:
                data.append((pickle.load(handle)))
        else:
            print("[Error] There is no data file '%s'" % arg[i])
            exit()
    return data

# Load data.
print('Load data...')
[train_input, train_refer, devel_input, devel_refer]\
	= open_data(train_input_data_path, train_refer_data_path,
                devel_input_data_path, devel_refer_data_path)

# construct layer
num_context_window = 2
input_dim = 257
hidden_dim = 512
output_dim = 257
num_train = 8
num_layers = 1

class lstm_sh(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):
        super(lstm_sh, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        # lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        lstm_out, self.hidden = self.lstm(input.view(len(input), 1, -1))
        # lstm_out = sigmo(lstm_out)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        # y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        y_pred = self.linear(lstm_out.view(len(input), -1))
        y_pred = sigmo(y_pred)
        # return y_pred.view(-1)
        return y_pred


model = lstm_sh(input_dim, hidden_dim, batch_size=num_train,\
             output_dim=output_dim, num_layers=num_layers)
# relu = torch.nn.ReLU()
sigmo = torch.nn.Sigmoid()
# model = sigmo(model)

learning_rate = 0.0005
training_epochs = 5
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
cost_func = torch.nn.MSELoss()

"""get parameter"""
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

"""normalize"""
def normalize(data, mu, sigma):
    # mean, variance -> size 맞춰주기
        return (data - mu) / sigma
def normalize_batch(data, mu, sigma):       # perform normalize
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

"""make context window"""
def make_context_data(data, L):

    for k in range(len(data)):
        context_data = data[k]
        for i in range(1, L+1):
            previous_data = np.delete(data[k], np.s_[-i::1], 0) #
            future_data  = np.delete(data[k], np.s_[0:i:1], 0)
            data[k]
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

"""create folder"""
def create_folder(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


"""Train network."""

job_dir = './job/small_lstm'
mode = 'irm'
dir_to_save = job_dir + '_%s' % mode
create_folder(dir_to_save)


print('Get statistical parameter...')
mu_input, sig_input, mu_refer, sig_refer \
    = get_statistics(train_input, train_refer)

# Save statistical parameter.
print('Save statistical parameter...')
np.save(dir_to_save + '/mu_input.npy', mu_input)
np.save(dir_to_save + '/sig_input.npy', sig_input)
np.save(dir_to_save + '/mu_refer.npy', mu_refer)
np.save(dir_to_save + '/sig_refer.npy', sig_refer)

# Normalize data

print('Normalize batch data...')
train_input = normalize_batch(train_input, mu_input, sig_input)
devel_input = normalize_batch(devel_input, mu_input, sig_input)

# # Make context window data
# print('Make context window data...')
# train_input = make_context_data(train_input, num_context_window)
# devel_input = make_context_data(devel_input, num_context_window)

# Start training.
num_utt_train = len(train_input)
num_utt_devel = len(devel_input)
idx_set = range(len(train_input))

# Write log file.
fp = open(dir_to_save + '/log.txt', 'w')
def write_status_to_log_file(fp):
    # Write log file.

    fp.write('%d-%d-%d %d:%d:%d\n' %
             (time.localtime().tm_year, time.localtime().tm_mon,
              time.localtime().tm_mday, time.localtime().tm_hour,
              time.localtime().tm_min, time.localtime().tm_sec))
    fp.write('mode                : %s\n' % mode)
    fp.write('learning rate       : %g\n' % learning_rate)
    fp.write('context window size : %g\n' % num_context_window)
write_status_to_log_file(fp)

mse_devel_total = np.zeros(training_epochs + 1)
for i in range(training_epochs + 1):
    start_time = time.time()
    avg_cost = 0
    # Train data
    model.train()

    mse_train_arr = torch.zeros(num_utt_train)
    for j in range(num_utt_train):  # all data from train_input
        inp_t, ref_t = torch.Tensor(train_input[j]), torch.Tensor(train_refer[j])
        # inp_t = inp_t.unsqueeze(0)
        optimizer.zero_grad()
        est_mask = model(inp_t)
        mse_train_arr_ = cost_func(est_mask, ref_t)
        mse_train_arr_.backward()
        mse_train_arr[j] = mse_train_arr_.data
        optimizer.step()
    mse_train = torch.mean(mse_train_arr)

    model.eval()
    mse_devel_arr = np.zeros(num_utt_devel)
    for j in range(num_utt_devel):
        inp_d, ref_d = torch.Tensor(devel_input[j]), torch.Tensor(devel_refer[j])
        est_mask = model(inp_d)
        mse_devel_arr[j] = cost_func(est_mask, ref_d)
    mse_devel = np.mean(mse_devel_arr)
    mse_devel_total[i] = mse_devel

    # Print progress.
    print('step %d:  %.6f  %.6f  takes %.2f seconds' %
          (i, mse_train, mse_devel, time.time() - start_time))
    fp.write('step %d:  %.6f  %.6f  takes %.2f seconds\n' %
             (i, mse_train, mse_devel, time.time() - start_time))
    # Save models.
    torch.save(model, '{}/{}.pt'.format(dir_to_save, i))

fp.close()
print('Training is finished...')

# Copy optimum model that has minimum MSE.
print('Save optimum models...')
min_index = np.argmin(mse_devel_total)
os.system('cp %s/%d.pt \
                    %s/opt.pt.data' %
          (dir_to_save, min_index, dir_to_save))