import numpy as np
import os
import scipy.io as sio
from typing import Tuple, Dict, Any
import logging
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

class CapgMyoDataLoader:
    def __init__(self, config: 'DataConfig'):
        self.config = config
        self.stride = int(config.window_size * (1 - config.overlap))
        
    def load_subject_data(self, subject_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load data for a single subject"""
        X_subject = []
        y_subject = []
        
        for session in ['session1', 'session2']:
            session_path = os.path.join(subject_path, session)
            if not os.path.exists(session_path):
                continue
                
            for gesture in range(self.config.num_gestures):
                gesture_file = f'gesture{gesture+1}.mat'
                file_path = os.path.join(session_path, gesture_file)
                
                if not os.path.exists(file_path):
                    continue
                    
                try:
                    data = sio.loadmat(file_path)
                    emg_data = data['emg']
                    X_subject.append(emg_data)
                    y_subject.extend([gesture] * len(emg_data))
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {str(e)}")
                    
        return np.vstack(X_subject), np.array(y_subject)
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from all subjects"""
        logger.info("Loading CapgMyo dataset...")
        X_all = []
        y_all = []
        
        for subject_dir in sorted(os.listdir(self.config.data_dir)):
            if not subject_dir.startswith('subject'):
                continue
                
            subject_path = os.path.join(self.config.data_dir, subject_dir)
            logger.info(f"Processing {subject_dir}")
            
            X_subject, y_subject = self.load_subject_data(subject_path)
            X_all.append(X_subject)
            y_all.append(y_subject)
        
        if not X_all:
            raise ValueError("No data was loaded. Please check the data directory path.")
        
        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        
        logger.info(f"Loaded dataset shape: {X.shape}")
        return X, y
