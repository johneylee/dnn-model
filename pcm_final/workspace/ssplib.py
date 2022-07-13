#!/usr/bin/python

"""
Speech signal processing library
It contains functions about speech signal processing.
"""

import time
import numpy as np
#import scipy as sp
import scipy.io.wavfile as wav
#from scikits.talkbox import lpc
from collections import namedtuple


sparams = namedtuple('sparams',
                     'fs, '
                     'frm_size, '
                     'frm_samp, '
                     'ratio, '
                     'window_analysis, '
                     'window_synthesis')


def initialize_params(fs=16000, frm_size=0.032, ratio=0.25, flag=1):
    """Generate Hann window.

    Args:
        fs: sampling frequency in Hz
        frm_size: frame size in second
        ratio: shift ratio (0.25, 0.5 or 0.75)
        flag:

    Returns:
        shps: speech hyper-parameters
    """
    frm_samp = int(fs * frm_size)
    window_analysis \
        = np.float32(np.sqrt(hann_window(frm_samp)))
    if flag == 0:
        window_synthesis = np.float32(np.ones(frm_samp))
    else:
        window_synthesis = window_analysis
    shps = sparams(fs=fs, frm_size=frm_size,
                   frm_samp=frm_samp, ratio=ratio,
                   window_analysis=window_analysis, 
                   window_synthesis=window_synthesis)
    return shps


def hann_window(winsize_samp):
    """Generate Hann window.

    Args:
        winsize_samp: window length in sample

    Returns:
        window: vector of Hann window
    """
    tmp = np.arange(1, winsize_samp + 1, 1.0, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos((2.0 * np.pi * tmp) / (winsize_samp + 1))
    window = np.float32(window)
    return window


def enframe(waveform, fs, window, frm_size, ratio):
    """Framize input waveform.

    Args:
        waveform: input speech signal
        fs: sampling frequency
        window: analysis window
        frm_size: size of frame (sec)
        ratio: overlap ratio (0.25 or 0.5)

    Returns:
        waveform_win: windowed signal
    """
    frm_length_sample = int(frm_size * fs)
    inc_sample = int(frm_length_sample * ratio)

    numfrms = (len(waveform) - frm_length_sample + inc_sample) // inc_sample
    waveform_win= np.zeros([numfrms, frm_length_sample])
    for frmidx in range(numfrms):
        st = frmidx * inc_sample
        waveform_win[frmidx, :] = window * waveform[st:st + frm_length_sample]

    return waveform_win


def stft(waveform, fs, window_analysis, frm_size, ratio):
    """Perform short-time Fourier transform.

    Args:
        waveform: input signal
        fs: sampling frequency
        window_analysis: analysis window
        frm_size: frame size (sec)
        ratio: overlap ratio (0.25 or 0.5)

    Returns:
        spec_full: complex signal in frequency domain (#frame by FFT size)
    """
    frm = enframe(waveform, fs, window_analysis, frm_size, ratio)
    spec_full = np.fft.fft(frm)
    return spec_full


def rstft(waveform, fs, window_analysis, frm_size, ratio):
    """Perform short-time Fourier transform for real-valued signal.

    Args:
        waveform: input signal
        fs: sampling frequency
        window_analysis: analysis window
        frm_size: frame size (sec)
        ratio: overlap ratio (0.25 or 0.5)

    Returns:
        spec_full: complex signal in frequency domain (#frame by FFT size)
    """
    frm = enframe(waveform, fs, window_analysis, frm_size, ratio)
    spec_half = np.fft.rfft(frm)
    return spec_half


def istft(spec_full, window_analysis, window_synthesis,
          length_waveform, ratio):
    """Perform inverse short-time Fourier transform.

    Args:
        spec_full: complex signal in frequency domain (#frames by FFT size)
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
    elif ratio == 0.125:
        for idx in range(0, 8):
            st = idx * inc_samp
            ed = st + inc_samp
            window_frag += window_mixed[st:ed]
            denorm = np.concatenate([window_frag,
                                     window_frag,
                                     window_frag,
                                     window_frag,
                                     window_frag,
                                     window_frag,
                                     window_frag,
                                     window_frag])
    else:
        print('only 50%, 75%, 87.5% OLA are available')

    frm = np.real(np.fft.ifft(spec_full))
    for n, i in enumerate(range(0, length_waveform - frm_samp, inc_samp)):
        waveform[i:i + frm_samp] += frm[n] * window_synthesis / denorm
    return waveform


def irstft(spec_half, window_analysis, window_synthesis,
           length_waveform, ratio):
    """Perform inverse short-time Fourier transform for real-valued signal.

    Args:
        spec_half: complex signal in frequency domain
        window_analysis: analysis window
        window_synthesis: synthesis window
        length_waveform: length of synthesized waveform (sample)
        ratio: overlap ratio (0.25 or 0.5)

    Returns:
        waveform: time domain signal
    """
    waveform = np.zeros(length_waveform)
    frm_samp = window_analysis.shape[0]
    inc_samp = int(frm_samp * ratio)
    window_mixed = window_analysis * window_synthesis
    window_frag = np.zeros(inc_samp)

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
    elif ratio == 0.125:
        for idx in range(0, 8):
            st = idx * inc_samp
            ed = st + inc_samp
            window_frag += window_mixed[st:ed]
        denorm = np.concatenate([window_frag,
                                 window_frag,
                                 window_frag,
                                 window_frag,
                                 window_frag,
                                 window_frag,
                                 window_frag,
                                 window_frag])
    else:
        print('only 50%, 75%, 87.5% OLA are available')

    frm = np.fft.irfft(spec_half) * window_synthesis / denorm
    for n, i in enumerate(range(0, length_waveform - frm_samp, inc_samp)):
        waveform[i:i + frm_samp] += frm[n]
    return waveform


def wav2magpha_half(waveform, fs, window_analysis, frm_size, ratio):
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
    spec_half = rstft(waveform, fs, window_analysis, frm_size, ratio)

    # Get log-magnitude and phase.
    mag_half = np.absolute(spec_half)
    pha_half = np.angle(spec_half)

    return mag_half, pha_half


def wav2realimag_half(waveform, fs, window_analysis, frm_size, ratio):
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
    spec_full = stft(waveform, fs, window_analysis, frm_size, ratio)
    num_frms, num_bins = spec_full.shape
    num_nyquist_bins = int(num_bins / 2 + 1) # int(num_bins >> 1 + 1)
    spec_half = spec_full[:, 0:num_nyquist_bins]
    real_half = np.real(spec_half)
    imag_half = np.imag(spec_half)

    return real_half, imag_half


def wav2logmagpha_half(waveform, fs, window_analysis, frm_size, ratio):
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
    mag_half, pha_half = wav2magpha_half(waveform, fs, 
                                         window_analysis, frm_size, ratio)
    logmag_half = np.log(mag_half + 1e-7)
    return logmag_half, pha_half 


def logmagpha_half2wav(logmag_half, pha_half, 
                       window_analysis, window_synthesis, 
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
    waveform_recons = spec_full2wav(spec_full, window_analysis, 
                                    window_synthesis, length_waveform, ratio)
    return waveform_recons


def magpha_half2wav(mag_half, pha_half,
                    window_analysis, window_synthesis,
                    length_waveform, ratio):
    """Synthesis waveform from the log-magnitude and phase.

    Args:
        mag_half: magnitude of waveform (#frames by #Nyquist bins)
        pha_half: phase in frequency domain (#frames by #Nyquist bins)
        window_analysis: analysis window
        window_synthesis: synthesis window
        length_waveform: length of synthesized waveform (smaple)
        ratio: overlap ratio (0.25 or 0.5)

    Returns:
        waveform_recons: synthesized
    """
    spec_half = mag_half * np.exp(1.0j * pha_half)
    waveform_recons = spec_half2wav(spec_half, window_analysis,
                                    window_synthesis, length_waveform, ratio)
    return waveform_recons


def spec_half2wav(spec_half, 
                  window_analysis, window_synthesis, 
                  length_waveform, ratio):
    """Synthesis waveform from the half complex spectrum.

    Args:
        spec_half: half complex signal in frequency domain 
                   (#frame by #Nyquist bin)
        window_analysis: analysis window
        window_synthesis: synthesis window
        length_waveform: length of synthesized waveform (sample)
        ratio: overlap ratio (0.25 or 0.5)

    Returns:
        waveform_recons: synthesized waveform
    """
    waveform_recons = irstft(spec_half, window_analysis, window_synthesis,
                             length_waveform, ratio)
    return waveform_recons


def spec_full2wav(spec_full, 
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
    waveform_recons = istft(spec_full, window_analysis, window_synthesis, 
                            length_waveform, ratio)
    return waveform_recons


def pha2bpd(pha, fs, frm_size, ratio):
    fftsize = fs * frm_size
    incsize = fs * frm_size * ratio
    numfrms = pha.shape[0]
    numbins = pha.shape[1]
    tmp = np.concatenate([pha[0:1, :], np.diff(pha, axis=0)], axis=0)
    ifreq = np.angle(np.exp(1j * tmp))
    ohm = 2.0 * np.pi * np.arange(numbins, dtype=np.float32) / fftsize * incsize
    vec_one = np.ones([numfrms, 1])
    bpd = np.angle(np.exp(1j * (ifreq - (vec_one * ohm))))
    return bpd


def mulaw(frm_vnorm, mu):
    frm_vnorm_mulaw = np.sign(frm_vnorm) * np.log(
        1.0 + mu * np.abs(frm_vnorm)) / np.log(1.0 + mu)
    return frm_vnorm_mulaw
