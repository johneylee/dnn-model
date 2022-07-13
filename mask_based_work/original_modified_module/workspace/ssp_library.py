"""
Speech signal processing library
It contains functions about speech signal processing.
"""

import time
import numpy as np
import scipy.io.wavfile as wav
from collections import namedtuple


sparams = namedtuple('sparams',
                     'fs, '
                     'frm_size, '
                     'frm_samp, '
                     'ratio, '
                     'window_analysis, '
                     'window_synthesis')


def initialize_params(fs=16000, frm_size=0.032, ratio=0.5):
    # Initialize speech hyper parameters.
    frm_samp = int(fs * frm_size)
    window_analysis \
        = np.float32(hann_window(frm_samp))
    window_synthesis = window_analysis
    shps = sparams(fs=fs, frm_size=frm_size, 
                   frm_samp=frm_samp, ratio=ratio,
                   window_analysis=window_analysis, 
                   window_synthesis=window_synthesis)
    return shps


def hann_window(win_samp):
    tmp = np.arange(1, win_samp + 1, 1.0, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos((2.0 * np.pi * tmp) / (win_samp + 1))
    return np.float32(window)


def enframe(waveform, fs, window_analysis, frm_size, ratio):
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


def stft(waveform, fs, window_analysis, frm_size, ratio):
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
    frm = enframe(waveform, fs, window_analysis, frm_size, ratio)
    spec_full = np.fft.fft(frm)
    return spec_full


def istft(spec_full, window_analysis, window_synthesis, 
          length_waveform, ratio):
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
    spec_full = stft(waveform, fs, window_analysis, frm_size, ratio)
    num_frms, num_bins = spec_full.shape
    num_nyquist_bins = int(num_bins / 2 + 1) # int(num_bins >> 1 + 1)

    # Get log-magnitude and phase.
    mag_full = np.absolute(spec_full)
    pha_full = np.angle(spec_full)

    # Slice redundant parts.
    mag_half = mag_full[:, 0:num_nyquist_bins]
    pha_half = pha_full[:, 0:num_nyquist_bins]
    return mag_half, pha_half


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
                            length_waveform, ratio) * (2 ** 15)
    waveform_recons = waveform_recons.astype(np.int16)
    return waveform_recons