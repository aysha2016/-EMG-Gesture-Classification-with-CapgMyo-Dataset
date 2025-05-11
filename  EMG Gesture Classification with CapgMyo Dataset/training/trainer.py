import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple
from ..utils.logging_utils import get_logger
from .callbacks import get_training_callbacks
from .metrics import compute_metrics

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, model: 'BaseModel', config: 'TrainingConfig'):
        self.model = model
        self.config = config
        self.history = None
        
    def train(self, train_data: Tuple[np.ndarray, np.ndarray],
              val_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Train the model"""
        logger.info("Starting model training...")
        
        # Prepare data
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Convert labels to one-hot encoding
        y_train_onehot = tf.keras.utils.to_categorical(y_train, self.model.config.num_classes)
        y_val_onehot = tf.keras.utils.to_categorical(y_val, self.model.config.num_classes)
        
        # Get callbacks
        callbacks = get_training_callbacks(self.config)
        
        # Train model
        self.history = self.model.model.fit(
            X_train, y_train_onehot,
            validation_data=(X_val, y_val_onehot),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed.")
        return self.history.history
    
    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        X_test, y_test = test_data
        y_pred = self.model.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        metrics = compute_metrics(y_test, y_pred_classes)
        logger.info(f"Test metrics: {metrics}")
        
        return metrics 