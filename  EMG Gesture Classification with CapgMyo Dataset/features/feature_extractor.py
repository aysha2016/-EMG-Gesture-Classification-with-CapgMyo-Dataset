import numpy as np
from typing import Dict, Any
from .time_domain import TimeDomainFeatures
from .frequency_domain import FrequencyDomainFeatures
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class EMGFeatureExtractor:
    def __init__(self, config: 'DataConfig'):
        self.config = config
        self.time_features = TimeDomainFeatures()
        self.freq_features = FrequencyDomainFeatures()
        
    def extract_features(self, window: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all features from an EMG window"""
        features = {}
        
        # Time domain features
        features.update(self.time_features.extract(window))
        
        # Frequency domain features
        features.update(self.freq_features.extract(window, self.config.sampling_rate))
        
        return features
    
    def extract_batch(self, windows: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features from a batch of windows"""
        logger.info("Extracting features from batch...")
        batch_features = {}
        
        for i, window in enumerate(windows):
            window_features = self.extract_features(window)
            
            if i == 0:
                # Initialize batch_features with the first window's features
                for key, value in window_features.items():
                    batch_features[key] = np.zeros((len(windows), *value.shape))
            
            # Add features to batch
            for key, value in window_features.items():
                batch_features[key][i] = value
        
        logger.info(f"Extracted features for {len(windows)} windows")
        return batch_features
