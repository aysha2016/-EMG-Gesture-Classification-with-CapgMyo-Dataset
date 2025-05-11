import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional
from ..utils.logging_utils import get_logger
from .augmentation import EMGAugmenter

logger = get_logger(__name__)

class EMGPreprocessor:
    def __init__(self, config: 'DataConfig'):
        self.config = config
        self.augmenter = EMGAugmenter()
        
    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to EMG signals"""
        lowcut = 20
        highcut = 500
        nyquist = 0.5 * self.config.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)
    
    def normalize(self, data: np.ndarray, 
                 mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize EMG signals"""
        if mean is None:
            mean = np.mean(data, axis=0)
        if std is None:
            std = np.std(data, axis=0) + 1e-6
            
        normalized = (data - mean) / std
        return normalized, mean, std
    
    def create_windows(self, data: np.ndarray, labels: np.ndarray,
                      augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Create windows from EMG signals"""
        X_windows = []
        y_windows = []
        
        for i in range(0, len(data) - self.config.window_size + 1, self.stride):
            window = data[i:i + self.config.window_size]
            window_label = labels[i + self.config.window_size // 2]
            
            if augment:
                window = self.augmenter.augment(window)
            
            X_windows.append(window)
            y_windows.append(window_label)
        
        return np.array(X_windows), np.array(y_windows)
    
    def preprocess_pipeline(self, data: np.ndarray, labels: np.ndarray,
                          augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        # Apply bandpass filter
        filtered_data = self.bandpass_filter(data)
        
        # Normalize
        normalized_data, mean, std = self.normalize(filtered_data)
        
        # Create windows
        X_processed, y_processed = self.create_windows(normalized_data, labels, augment)
        
        logger.info(f"Preprocessed data shape: {X_processed.shape}")
        return X_processed, y_processed
