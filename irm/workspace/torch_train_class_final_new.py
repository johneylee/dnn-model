import os
import time
import numpy as np
import torch
import pickle
import torch.nn as nn
import config as cfg
from copy import deepcopy

class SpeechEnhancementNetwork:
    def __init__(self):
        # Load data.
        print('Load data...')
        [self.train_input, self.train_refer, self.devel_input, self.devel_refer] \
            = self.open_data(cfg.train_input_data_path, cfg.train_refer_data_path,
                        cfg.devel_input_data_path, cfg.devel_refer_data_path)

        # set train origin data for training
        self.train_input_origin = deepcopy(self.train_input) # [T, 257]
        self.devel_input_origin = deepcopy(self.devel_input)
        self.train_target_origin = deepcopy(self.train_refer)
        self.devel_target_origin = deepcopy(self.devel_refer)
        #inp_t_origin = torch.Tensor(train_input_origin[0])
        #print(inp_t_origin.size())

        # construct layer
        self.num_context_window = cfg.num_context_window
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.layer_size
        self.output_dim = cfg.output_dim
        self.num_layers = cfg.num_layer
        self.learning_rate = cfg.learning_rate
        self.training_epochs = cfg.max_epochs
        self.batch_size = cfg.batch_size
        self.network_name = cfg.network_type
        self.learning_rate = cfg.learning_rate

    def train(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.network_name == 'lstm':
            self.model = lstm_sh(self.input_dim, self.hidden_dim, self.batch_size, \
                                 output_dim=self.output_dim, num_layers=self.num_layers).to(device)
        if self.network_name == 'dnn':
            self.model = dnn_sh(self.input_dim, self.num_context_window, self.hidden_dim, \
                                self.output_dim, self.num_layers).to(device)

        self.sigmo = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.cost_func = torch.nn.MSELoss()

        """Train network."""
        self.job_dir = cfg.job_dir
        self.mode = cfg.mode
        self.dir_to_save = self.job_dir + '_%s' % self.mode
        self.create_folder(self.dir_to_save)

        print('Get statistical parameter...')
        mu_input, sig_input, mu_refer, sig_refer \
            = self.get_statistics(self.train_input, self.train_refer)

        # Save statistical parameter.
        print('Save statistical parameter...')
        np.save(self.dir_to_save + '/mu_input.npy', mu_input)
        np.save(self.dir_to_save + '/sig_input.npy', sig_input)
        np.save(self.dir_to_save + '/mu_refer.npy', mu_refer)
        np.save(self.dir_to_save + '/sig_refer.npy', sig_refer)

        # Normalize data
        print('Normalize batch data...')
        train_input_norm = self.normalize_batch(deepcopy(self.train_input), mu_input, sig_input) # (T, 257)
        devel_input_norm = self.normalize_batch(deepcopy(self.devel_input), mu_input, sig_input)
        train_refer_norm = self.normalize_batch(deepcopy(self.train_refer), mu_input, sig_input)
        devel_refer_norm = self.normalize_batch(deepcopy(self.devel_refer), mu_input, sig_input)

        #print(self.train_input_origin[0].shape)
        if self.network_name == 'dnn':
            # Make context window data
            print('Make context window data...')
            train_input_ctxt = self.make_context_data(deepcopy(train_input_norm), self.num_context_window)
            devel_input_ctxt = self.make_context_data(deepcopy(devel_input_norm), self.num_context_window)
            #train_input_ctxt = self.make_context_data(deepcopy(self.train_input), self.num_context_window)
            #devel_input_ctxt = self.make_context_data(deepcopy(self.devel_input), self.num_context_window)
        #print(self.train_input_origin[0].shape)

        print("# of model parameters: {:,}".format(self.count_parameters(self.model)))

        # Start training.
        self.num_utt_train = len(train_input_norm)
        #self.num_utt_train = len(deepcopy(self.train_input))
        self.num_utt_devel = len(devel_input_norm)
        #self.num_utt_devel = len(deepcopy(self.devel_input))
        self.idx_set = range(len(train_input_norm))
        #self.idx_set = range(len(deepcopy(self.train_input)))

        # Write log file.
        self.fp = open(self.dir_to_save + '/log.txt', 'w')
        self.write_status_to_log_file(self.fp)


        mse_devel_total = np.zeros(self.training_epochs + 1)
        for i in range(self.training_epochs + 1):
            start_time = time.time()

            # Train data load
            self.model.train()
            mse_train_arr = torch.zeros(self.num_utt_train)

            #print(self.train_input_origin[0].shape)
            # train loss
            if self.network_name == 'lstm':
                for j in range(self.num_utt_train):  # all data from train_input
                    #inp_t, ref_t = torch.Tensor(train_input_norm[j]), torch.Tensor(train_refer_norm[j])
                    inp_t, ref_t = torch.Tensor(train_input_norm[j]), torch.Tensor(self.train_target_origin[j])
                    #inp_t_origin = torch.Tensor(self.train_input_origin[j])

                    if torch.cuda.is_available():
                        inp_t = inp_t.cuda()
                        ref_t = ref_t.cuda()
                        #inp_t_origin = inp_t_origin.cuda()
                    self.optimizer.zero_grad()
                    est_mask = self.model(inp_t)

                    if self.mode == 'irm' or 'ibm':
                        mse_train_arr_ = self.cost_func(est_mask, ref_t)
                    if self.mode == 'spectrogram':
                        #print(inp_t_origin.size())
                        est_spectrogram = torch.log(est_mask * torch.exp(inp_t) + 1e-7)
                        #est_sepctrogram = (est_mask * inp_t_origin)
                        mse_train_arr_ = self.cost_func(est_spectrogram, ref_t)
                    mse_train_arr_.backward()
                    mse_train_arr[j] = mse_train_arr_.data
                    self.optimizer.step()
                mse_train = torch.mean(mse_train_arr)

            if self.network_name == 'dnn':
                idx_set = np.random.permutation(self.idx_set)
                for j in range(self.num_utt_train):  # all data from train_input
                    idxs = idx_set[j]
                    inp_t, ref_t = torch.Tensor(train_input_ctxt[idxs]), torch.Tensor(train_refer_norm[idxs])
                    #inp_t, ref_t = torch.Tensor(train_input_ctxt[idxs]), torch.Tensor(self.train_target_origin[idxs])
                    #inp_t_origin = torch.Tensor(self.train_input_origin[idxs])

                    #print(inp_t_origin.size())

                    if torch.cuda.is_available():
                        inp_t = inp_t.cuda()
                        ref_t = ref_t.cuda()
                        #inp_t_origin = inp_t_origin.cuda()

                    self.optimizer.zero_grad()
                    est_mask = self.model(inp_t)

                    if self.mode == 'irm' or 'ibm':
                        mse_train_arr_ = self.cost_func(est_mask, ref_t)
                    if self.mode == 'spectrogram':
                        est_spectrogram = torch.log(est_mask * torch.exp(inp_t[:,:257]) + 1e-7)
                        #est_spectrogram = est_mask * inp_t_origin
                        #est_spectrogram = est_mask * inp_t[:,:257]
                        mse_train_arr_ = self.cost_func(est_spectrogram, ref_t)
                    mse_train_arr_.backward()
                    mse_train_arr[j] = mse_train_arr_.data
                    self.optimizer.step()
                mse_train = torch.mean(mse_train_arr)

            # validation loss
            self.model.eval()
            mse_devel_arr = np.zeros(self.num_utt_devel)
            if self.network_name == 'lstm':
                for j in range(self.num_utt_devel):
                    #inp_d, ref_d = torch.Tensor(devel_input_norm[j]), torch.Tensor(devel_refer_norm[j])
                    inp_d, ref_d = torch.Tensor(devel_input_norm[j]), torch.Tensor(self.devel_target_origin[j])
                    #inp_d_origin = torch.Tensor(self.devel_input_origin[j])

                    if torch.cuda.is_available():
                        inp_d = inp_d.cuda()
                        ref_d = ref_d.cuda()
                        #inp_d_origin = inp_d_origin.cuda()
                    est_mask = self.model(inp_d)

                    if self.mode == 'irm' or 'ibm':
                        mse_devel_arr[j] = self.cost_func(est_mask, ref_d)
                    if self.mode == 'spectrogram':
                        est_spectrogram = torch.log(est_mask * torch.exp(inp_d) + 1e-7)
                        #est_spectrogram = (est_mask * inp_d_origin)
                        mse_devel_arr[j] = self.cost_func(est_spectrogram, ref_d)

            if self.network_name == 'dnn':
                for j in range(self.num_utt_devel):
                    inp_d, ref_d = torch.Tensor(devel_input_ctxt[j]), torch.Tensor(devel_refer_norm[j])
                    #inp_d, ref_d = torch.Tensor(devel_input_ctxt[j]), torch.Tensor(self.devel_target_origin[j])
                    #inp_d_origin = torch.Tensor(self.devel_input_origin[j])

                    if torch.cuda.is_available():
                        inp_d = inp_d.cuda()
                        ref_d = ref_d.cuda()
                        #inp_d_origin = inp_d_origin.cuda()

                    est_mask = self.model(inp_d)

                    if self.mode == 'irm' or 'ibm':
                        mse_devel_arr[j] = self.cost_func(est_mask, ref_d)
                    if self.mode == 'spectrogram':
                        est_spectrogram = torch.log(est_mask * torch.exp(inp_d[:,:257]) + 1e-7)
                        #est_spectrogram = est_mask * inp_d_origin
                        #est_spectrogram = est_mask * inp_d[:,:257]

                        mse_devel_arr[j] = self.cost_func(est_spectrogram, ref_d)

            mse_devel = np.mean(mse_devel_arr)
            mse_devel_total[i] = mse_devel


            # Print progress.
            print('step %d:  %.6f  %.6f  takes %.2f seconds' %
                  (i, mse_train, mse_devel, time.time() - start_time))
            self.fp.write('step %d:  %.6f  %.6f  takes %.2f seconds\n' %
                     (i, mse_train, mse_devel, time.time() - start_time))
            # Save models.
            torch.save(self.model, '{}/{}.pt'.format(self.dir_to_save, i))

        self.fp.close()
        print('Training is finished...')

        # Copy optimum model that has minimum MSE.
        print('Save optimum models...')
        min_index = np.argmin(mse_devel_total)
        os.system('cp %s/%d.pt \
                            %s/opt.pt.data' %
                  (self.dir_to_save, min_index, self.dir_to_save))

    """get parameter"""
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_mu_sig(self, data):
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

    def get_global_mu_sig_partial(self, data, idx):
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

    def get_statistics(self, inp, refer):
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
        mu_inp, sig_inp = self.get_mu_sig(inp)
        mu_refer1, sig_refer1 \
            = self.get_global_mu_sig_partial(refer, range(0, dim))

        mu_refer = mu_refer1 * np.ones(dim)
        sig_refer = sig_refer1 * np.ones(dim)
        return mu_inp, sig_inp, mu_refer, sig_refer

    """normalize"""

    def normalize(self, data, mu, sigma):
        # mean, variance -> size 맞춰주기
        return (data - mu) / sigma

    def normalize_batch(self, data, mu, sigma):  # perform normalize
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
            data[i] = self.normalize(data[i], mu, sigma)
        return data

    """make context window"""

    def make_context_data(self, data, L):

        for k in range(len(data)):
            context_data = data[k]
            for i in range(1, L + 1):
                previous_data = np.delete(data[k], np.s_[-i::1], 0)  # data[k]: input, np.s_[-i::1]: want to delete, 0: axis
                future_data = np.delete(data[k], np.s_[0:i:1], 0)
                start_frame = np.array([data[k][0, :]])
                last_frame = np.array([data[k][-1, :]])

                dusf = start_frame
                dulf = last_frame

                for j in range(i - 1):
                    dusf = np.concatenate((dusf, start_frame))
                    dulf = np.concatenate((dulf, last_frame))
                previous_data = np.concatenate((dusf, previous_data))
                future_data = np.concatenate((future_data, dulf))

                context_data = np.concatenate((context_data, previous_data), 1)
                context_data = np.concatenate((context_data, future_data), 1)
            data[k] = context_data

        return data

    """create folder"""

    def create_folder(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # open feature data
    def open_data(self, *arg):  # input data load
        """Open numpy data from directory.

        Args:
            arg: file names

        Returns:
            data: data files
        """
        data = []
        for i in range(len(arg)):  # number of input data: 4
            if os.path.exists(arg[i]):
                with open(arg[i], 'rb') as handle:
                    data.append((pickle.load(handle)))
            else:
                print("[Error] There is no data file '%s'" % arg[i])
                exit()
        return data

    """write log file"""
    def write_status_to_log_file(self, fp):
        # Write log file.

        fp.write('%d-%d-%d %d:%d:%d\n' %
                 (time.localtime().tm_year, time.localtime().tm_mon,
                  time.localtime().tm_mday, time.localtime().tm_hour,
                  time.localtime().tm_min, time.localtime().tm_sec))
        fp.write('mode                : %s\n' % self.mode)
        fp.write('learning rate       : %g\n' % self.learning_rate)
        fp.write('context window size : %g\n' % self.num_context_window)


class lstm_sh(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):
        super(lstm_sh, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.activation_func = cfg.activation_func
        # Define the LSTM layer
        self.lstm_sh = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

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
        lstm_out, self.hidden = self.lstm_sh(input.view(len(input), 1, -1))
        # lstm_out = sigmo(lstm_out)

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        # y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        y_pred = self.linear(lstm_out.view(len(input), -1))
        # y_pred = SpeechEnhancementNetwork.train.sigmo(y_pred)
        y_pred = sigmo(y_pred)
        # return y_pred.view(-1)
        return y_pred


class dnn_sh(nn.Module):
    def __init__(self, input_dim, num_context_window, hidden_dim, output_dim, num_layers):
        super(dnn_sh, self).__init__()
        self.activation_func = cfg.activation_func

        self.linear_in = torch.nn.Linear(input_dim * (num_context_window * 2 + 1), hidden_dim, bias=True)
        self.num_layer = num_layers
        self.layer_list = nn.ModuleList([])
        for _ in range(num_layers):
            layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.layer_list.append(layer)
        self.linear_out = torch.nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):

        x = self.linear_in(x)
        x = self.activation_func(x)
        for idx in range(self.num_layer):
            x = self.layer_list[idx](x)
            x = self.activation_func(x)
        x = self.linear_out(x)
        x = sigmo(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

sigmo = torch.nn.Sigmoid()

if __name__ == "__main__":

    #torch.cuda.set_device(cfg.gpu_id)
    #print(device)

    sunghyun = SpeechEnhancementNetwork()
    sunghyun.train()
