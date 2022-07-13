import os
import numpy as np
import scipy.io.wavfile as wav
from collections import namedtuple
import torch
from os import makedirs
import config as cfg
import torch.nn as nn
import pickle
import sys
from copy import deepcopy

class SpeechEnhancementDecoding:
    def __init__(self):

        # construct layer
        self.num_context_window = cfg.num_context_window
        self.model_name = cfg.network_model_name
        self.mode = cfg.mode
        self.network_name = cfg.network_type
        self.clean_dir = cfg.clean_directory

    def decode(self):

        """set parameters"""
        self.sparams = namedtuple('sparams',
                             'fs, '
                             'frm_size, '
                             'frm_samp, '
                             'ratio, '
                             'window_analysis, '
                             'window_synthesis')
        shps = self.initialize_params()

        """set cuda"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # import model
        self.check_model_file_existence(self.model_name)
        self.model = torch.load('%s/opt.pt.data' % self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.mu_input = np.load('./%s/mu_input.npy' % self.model_name)
        self.sig_input = np.load('./%s/sig_input.npy' % self.model_name)
        self.mu_refer = np.load('./%s/mu_refer.npy' % self.model_name)
        self.sig_refer = np.load('./%s/sig_refer.npy' % self.model_name)

        for (path, dir, files) in os.walk(self.clean_dir):
            for data in files:
                ext = os.path.splitext(data)[-1]
                if ext == '.wav':
                    addr_input = "%s/%s" % (path, data)
                    addr_new = path.replace('data', self.mode)
                    makedirs(addr_new, exist_ok=True)

                if os.path.isfile(addr_input) is False:
                    print('[Error] There is no input file.')
                    exit()
                
                self.perform_enhancement(addr_input, self.model_name, self.mode, shps)

    def perform_enhancement(self, addr_in, model_name, mode, shps):

        addr_out = addr_in.replace('.wav', '_' + model_name.split('/')[-1] + '.wav')
        #addr_out = addr_in.replace('.wav', '_' + model_name.split('/')[-1] + '.npy')
        print('%s' % addr_in)
        print(' -> %s' % addr_out)

        fs, input_wav = wav.read(addr_in)
        noisy_wav = input_wav.astype(np.float32) / (2 ** 15)
        length = len(noisy_wav)
        if mode == 'direct':
            logmag_noisy, pha_noisy \
                = self.wav2logmagpha_half(noisy_wav, shps.fs, shps.window_analysis, \
                                     shps.frm_size, shps.ratio)
            enhanced_logmag = self.run_decode_step(logmag_noisy, mode)
        elif mode in ['ibm', 'irm']:
            mag_noisy, pha_noisy \
                = self.wav2magpha_half(noisy_wav, shps.fs, shps.window_analysis, \
                                  shps.frm_size, shps.ratio)

            logmag_noisy = np.log(mag_noisy + 1e-7)
            mask = self.run_decode_step(logmag_noisy, mode).detach().cpu().numpy()
            enhanced_logmag = np.log(mask * mag_noisy + 1e-7)
        elif mode == 'spectrogram':
            mag_noisy, pha_noisy \
                = self.wav2magpha_half(noisy_wav, shps.fs, shps.window_analysis, \
                                  shps.frm_size, shps.ratio)

            logmag_noisy = np.log(mag_noisy + 1e-7)
            
            #logmag_norm, mask = self.run_decode_step(logmag_noisy, mode)
            #mask = mask.detach().cpu().numpy()
            #enhanced_input = logmag_norm * mask
            #enhanced_input_denorm = enhanced_input * self.sig_input + self.mu_input
            #enhanced_logmag = deepcopy(enhanced_input_denorm)
           
            ## type3
            #mask = self.run_decode_step(logmag_noisy, mode).detach().cpu().numpy()
            #enhanced_logmag = np.log(mask * mag_noisy + 1e-7)
            ## type4
            #inp_denorm, mask = self.run_decode_step(logmag_noisy, mode)
            #mask = mask.detach().cpu().numpy()
            #enhanced_logmag = deepcopy(inp_denorm * mask)
            ## type5
            #enhanced_spectrum = self.run_decode_step(logmag_noisy, mode).detach().cpu().numpy()
            #enhanced_logmag = enhanced_spectrum * self.sig_input + self.mu_input
            inp_norm, mask = self.run_decode_step(logmag_noisy, mode)
            mask = mask.detach().cpu().numpy()
            enhanced_loginp = np.log(mask * np.exp(inp_norm) + 1e-7)
            enhanced_logmag = enhanced_loginp * self.sig_input + self.mu_input

        enhanced = self.logmagpha_half2wav(enhanced_logmag, pha_noisy,
                                      shps.window_analysis,
                                      shps.window_synthesis,
                                      length, shps.ratio)
        enhanced_out = addr_out.replace('data', mode)
        wav.write(enhanced_out, shps.fs, enhanced)
        #np.save(enhanced_out, mask)
        
    def run_decode_step(self, inp, mode):
        # Get output from network.
        inp_norm = (inp - self.mu_input) / self.sig_input
        data = []
        data.append(inp_norm)
        if self.network_name == 'dnn':
            data = torch.Tensor(self.make_context_data(data, self.num_context_window)).squeeze(0)
        if self.network_name == 'lstm':
            data = torch.Tensor(data).squeeze(0)
        data = data.to(self.device)
        estimation = self.model(data)

        if mode == 'ibm':
            return np.round(estimation)
        elif mode == 'irm':
            return estimation
        elif mode == 'direct':
            estimation_denorm = estimation * self.sig_refer + self.mu_refer
            return estimation_denorm
        elif mode == 'spectrogram':
            ## type5
            #estimation = torch.log(estimation * torch.exp(data[:,:257]) + 1e-7)
            return inp_norm, estimation



    def check_model_file_existence(self, model_name):
        if os.path.isfile('%s/opt.pt.data' % model_name) is False:
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
    def initialize_params(self, fs=16000, frm_size=0.032, ratio=0.5):
        # Initialize speech hyper parameters.
        frm_samp = int(fs * frm_size)
        window_analysis \
            = np.float32(self.hann_window(frm_samp))
        window_synthesis = window_analysis
        shps = self.sparams(fs=fs, frm_size=frm_size,
                       frm_samp=frm_samp, ratio=ratio,
                       window_analysis=window_analysis,
                       window_synthesis=window_synthesis)
        return shps
    def hann_window(self, win_samp):
        tmp = np.arange(1, win_samp + 1, 1.0, dtype=np.float64)
        window = 0.5 - 0.5 * np.cos((2.0 * np.pi * tmp) / (win_samp + 1))
        return np.float32(window)
    def make_context_data(self, data, L):

        for k in range(len(data)):
            context_data = data[k]
            for i in range(1, L + 1):
                previous_data = np.delete(data[k], np.s_[-i::1], 0)  #
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
    def enframe(self, waveform, fs, window_analysis, frm_size, ratio):
        """Framize input waveform.

        Args:
            waveform: input speech signal
            fs: sampling frequency
            window_analysis: analysis window
            frm_size: size of frame (sec)
            ratio: overlap ratio (0.25 or 0.5)

        Returns:
            windowed_waveform: windowed signal
        """
        frm_samp = int(frm_size * fs)
        inc_samp = int(frm_samp * ratio)
        windowed_waveform = np.array([window_analysis * waveform[i:i + frm_samp]
                                      for i in range(0, len(waveform) - frm_samp, inc_samp)])
        return windowed_waveform
    def stft(self, waveform, fs, window_analysis, frm_size, ratio):
        """Perform Short time Fourier transform.

        Args:
            waveform: input signal
            fs: sampling frequency
            window_analysis: analysis window
            frm_size: frame size (sec)
            ratio: overlap ratio (0.25 or 0.5)

        Returns:
            spec_full: complex signal in frequency domain (#frame by FFT size)
        """
        frm = self.enframe(waveform, fs, window_analysis, frm_size, ratio)
        spec_full = np.fft.fft(frm)
        return spec_full
    def istft(self, spec_full, window_analysis, window_synthesis, length_waveform, ratio):
        """Inverse short time Fourier transform.

        Args:
            spec_full: complex signal in frequency domain (#frame by FFT size)
            window_analysis: analysis window
            window_synthesis: synthesis window
            length_waveform: length of synthesized waveform (sample)
            ratio: overlap ratio (0.25 or 0.5)

        Returns:
            waveform: time domain signal
        """
        waveform = np.zeros(length_waveform)
        frm_samp = spec_full.shape[1]
        inc_samp = int(frm_samp * ratio)
        window_mixed = window_analysis * window_synthesis
        window_frag = np.zeros(inc_samp)

        # Select shift rario.
        if ratio == 0.5:
            for idx in range(0, 2):
                st = idx * inc_samp
                ed = st + inc_samp
                window_frag += window_mixed[st:ed]
                denorm = np.concatenate([window_frag,
                                         window_frag])
        elif ratio == 0.25:
            for idx in range(0, 4):
                st = idx * inc_samp
                ed = st + inc_samp
                window_frag += window_mixed[st:ed]
                denorm = np.concatenate([window_frag,
                                         window_frag,
                                         window_frag,
                                         window_frag])
        else:
            print('only 50% and 25% OLA are available')
        for n, i in enumerate(range(0, length_waveform - frm_samp, inc_samp)):
            frm = np.real(np.fft.ifft(spec_full[n]))
            frm_windowed = frm * window_synthesis
            waveform[i:i + frm_samp] += frm_windowed / denorm
        return waveform
    def wav2logmagpha_half(self, waveform, fs, window_analysis, frm_size, ratio):
        """Convert waveform to log-magnitude and phase by using STFT.

        Args:
            waveform: input signal
            fs: sampling frequency
            window_analysis: analysis window
            frm_size: frame size (sec)
            ratio: overlap ratio (0.25 or 0.5)

        Returns:
            logmag_half: log-magnitude of waveform (#frames by #Nyquist bins)
            pha_half: phase in frequency domain (#frames by #Nyquist bins)
        """
        mag_half, pha_half = self.wav2magpha_half(waveform, fs,
                                             window_analysis, frm_size, ratio)
        logmag_half = np.log(mag_half + 1e-7)
        return logmag_half, pha_half
    def wav2magpha_half(self, waveform, fs, window_analysis, frm_size, ratio):
        """Convert waveform to magnitude and phase by using STFT.

        Args:
            waveform: input signal
            fs: sampling frequency
            window_analysis: analysis window
            frm_size: frame size (sec)
            ratio: overlap ratio (0.25 or 0.5)

        Returns:
            mag_half: magnitude of waveform (#frames by #Nyquist bins)
            pha_half: phase in frequency domain (#frames by #Nyquist bins)
        """
        spec_full = self.stft(waveform, fs, window_analysis, frm_size, ratio)
        num_frms, num_bins = spec_full.shape
        num_nyquist_bins = int(num_bins / 2 + 1)  # int(num_bins >> 1 + 1)

        # Get log-magnitude and phase.
        mag_full = np.absolute(spec_full)
        pha_full = np.angle(spec_full)

        # Slice redundant parts.
        mag_half = mag_full[:, 0:num_nyquist_bins]
        pha_half = pha_full[:, 0:num_nyquist_bins]
        return mag_half, pha_half
    def spec_full2wav(self, spec_full,
                      window_analysis, window_synthesis,
                      length_waveform, ratio):
        """Synthesis waveform from the complex spectrum.

        Args:
            spec_full: complex signal in frequency domain (#frame by FFT size)
            window_analysis: analysis window
            window_synthesis: synthesis window
            length_waveform: length of synthesized waveform (sample)
            ratio: overlap ratio (0.25 or 0.5)

        Returns:
            waveform_recons: synthesized waveform
        """
        waveform_recons = self.istft(spec_full, window_analysis, window_synthesis,
                                length_waveform, ratio) * (2 ** 15)
        waveform_recons = waveform_recons.astype(np.int16)
        return waveform_recons
    def logmagpha_half2wav(self, logmag_half, pha_half, \
                           window_analysis, window_synthesis, \
                           length_waveform, ratio):
        """Synthesis waveform from the log-magnitude and phase.

        Args:
            logmag_half: log-magnitude of waveform (#frames by #Nyquist bins)
            pha_half: phase in frequency domain (#frames by #Nyquist bins)
            window_analysis: analysis window
            window_synthesis: synthesis window
            length_waveform: length of synthesized waveform (smaple)
            ratio: overlap ratio (0.25 or 0.5)

        Returns:
            waveform_recons: synthesized
        """
        logmag_full = np.concatenate((logmag_half, logmag_half[:, -2:0:-1]),
                                     axis=1)
        pha_full = np.concatenate((pha_half, -1.0 * pha_half[:, -2:0:-1]), axis=1)
        spec_full = np.exp(logmag_full) * np.exp(1.0j * pha_full)
        waveform_recons = self.spec_full2wav(spec_full, window_analysis,
                                        window_synthesis, length_waveform, ratio)
        return waveform_recons

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

class lstm_sh(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):
        super(lstm_sh, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

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

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        # y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        y_pred = self.linear(lstm_out.view(len(input), -1))
        y_pred = sigmo(y_pred)
        # return y_pred.view(-1)
        return y_pred

sigmo = torch.nn.Sigmoid()

if __name__ == "__main__":
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(device)

    sunghyun = SpeechEnhancementDecoding()
    sunghyun.decode()
