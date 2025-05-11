import numpy as np
from scipy.signal import hilbert
from typing import Dict

class TimeDomainFeatures:
    def compute_rms(self, signal: np.ndarray) -> np.ndarray:
        """Compute Root Mean Square"""
        return np.sqrt(np.mean(signal ** 2, axis=0))
    
    def compute_mav(self, signal: np.ndarray) -> np.ndarray:
        """Compute Mean Absolute Value"""
        return np.mean(np.abs(signal), axis=0)
    
    def compute_wl(self, signal: np.ndarray) -> np.ndarray:
        """Compute Waveform Length"""
        return np.sum(np.abs(np.diff(signal, axis=0)), axis=0)
    
    def compute_zc(self, signal: np.ndarray) -> np.ndarray:
        """Compute Zero Crossing Rate"""
        return np.sum(np.diff(np.signbit(signal), axis=0), axis=0)
    
    def compute_pe(self, signal: np.ndarray) -> np.ndarray:
        """Compute Power Envelope"""
        return np.abs(hilbert(signal)) ** 2
    
    def extract(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all time domain features"""
        features = {
            'rms': self.compute_rms(signal),
            'mav': self.compute_mav(signal),
            'wl': self.compute_wl(signal),
            'zc': self.compute_zc(signal),
            'pe': self.compute_pe(signal)
        }
        return features
