import numpy as np
from scipy.signal import butter, filtfilt, hilbert, welch
import pywt

def bandpass_filter(data, lowcut=70, highcut=200, fs=1000, order=4):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return filtfilt(b, a, data, axis=0)

def compute_pe(signal):
    return np.abs(hilbert(signal)) ** 2

def compute_hg(signal, fs=1000):
    return bandpass_filter(signal, 70, 200, fs)

def compute_wavelet_energy(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.array([np.sum(np.square(c)) for c in coeffs])

def compute_zcr(signal):
    return ((signal[:-1] * signal[1:]) < 0).sum(axis=0)

def extract_features(emg_window, fs=1000):
    features = {}
    features['PE'] = compute_pe(emg_window).mean(axis=0)
    features['HG'] = compute_hg(emg_window, fs).mean(axis=0)
    features['HG+WF'] = np.concatenate([features['HG'], compute_zcr(emg_window)])
    features['HG+WE'] = np.concatenate([features['HG'], compute_wavelet_energy(emg_window)])
    features['HG+PE'] = np.concatenate([features['HG'], features['PE']])
    return features