import numpy as np
from typing import Optional

class EMGAugmenter:
    def __init__(self, noise_level: float = 0.05, 
                 time_shift_range: int = 50,
                 amplitude_scale_range: tuple = (0.8, 1.2)):
        self.noise_level = noise_level
        self.time_shift_range = time_shift_range
        self.amplitude_scale_range = amplitude_scale_range
    
    def add_noise(self, signal: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to the signal"""
        noise = np.random.normal(0, self.noise_level, signal.shape)
        return signal + noise
    
    def time_shift(self, signal: np.ndarray) -> np.ndarray:
        """Apply random time shift to the signal"""
        shift = np.random.randint(-self.time_shift_range, self.time_shift_range)
        if shift > 0:
            return np.pad(signal[:-shift], (0, shift), mode='edge')
        else:
            return np.pad(signal[-shift:], (abs(shift), 0), mode='edge')
    
    def amplitude_scale(self, signal: np.ndarray) -> np.ndarray:
        """Scale signal amplitude"""
        scale = np.random.uniform(*self.amplitude_scale_range)
        return signal * scale
    
    def augment(self, signal: np.ndarray, 
                add_noise: bool = True,
                time_shift: bool = True,
                amplitude_scale: bool = True) -> np.ndarray:
        """Apply random augmentations to the signal"""
        augmented = signal.copy()
        
        if add_noise:
            augmented = self.add_noise(augmented)
        
        if time_shift:
            augmented = self.time_shift(augmented)
        
        if amplitude_scale:
            augmented = self.amplitude_scale(augmented)
        
        return augmented
