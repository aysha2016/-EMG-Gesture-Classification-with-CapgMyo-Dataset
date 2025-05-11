import numpy as np
from scipy.signal import welch
from scipy.fft import fft
import pywt
from typing import Dict

class FrequencyDomainFeatures:
    def compute_psd(self, signal: np.ndarray, fs: int) -> Dict[str, np.ndarray]:
        """Compute Power Spectral Density"""
        freqs, psd = welch(signal, fs=fs, nperseg=256)
        return {'freqs': freqs, 'psd': psd}
    
    def compute_fft(self, signal: np.ndarray) -> np.ndarray:
        """Compute Fast Fourier Transform"""
        return np.abs(fft(signal, axis=0))
    
    def compute_wavelet(self, signal: np.ndarray, wavelet: str = 'db4', level: int = 4) -> Dict[str, np.ndarray]:
        """Compute Wavelet Transform"""
        coeffs = pywt.wavedec(signal, wavelet, level=level, axis=0)
        return {f'coeff_{i}': c for i, c in enumerate(coeffs)}
    
    def compute_spectral_entropy(self, signal: np.ndarray, fs: int) -> np.ndarray:
        """Compute Spectral Entropy"""
        freqs, psd = welch(signal, fs=fs, nperseg=256)
        psd_norm = psd / np.sum(psd, axis=0)
        return -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=0)
    
    def extract(self, signal: np.ndarray, fs: int) -> Dict[str, np.ndarray]:
        """Extract all frequency domain features"""
        features = {}
        
        # PSD features
        psd_features = self.compute_psd(signal, fs)
        features.update({
            'psd_mean': np.mean(psd_features['psd'], axis=0),
            'psd_std': np.std(psd_features['psd'], axis=0)
        })
        
        # FFT features
        fft_features = self.compute_fft(signal)
        features.update({
            'fft_mean': np.mean(fft_features, axis=0),
            'fft_std': np.std(fft_features, axis=0)
        })
        
        # Wavelet features
        wavelet_features = self.compute_wavelet(signal)
        for key, value in wavelet_features.items():
            features[f'{key}_mean'] = np.mean(value, axis=0)
            features[f'{key}_std'] = np.std(value, axis=0)
        
        # Spectral entropy
        features['spectral_entropy'] = self.compute_spectral_entropy(signal, fs)
        
        return features