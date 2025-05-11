import tensorflow as tf
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def _build_model(self) -> tf.keras.Model:
        """Build LSTM model architecture"""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=self.config.input_shape),
            
            # LSTM layers
            *[tf.keras.layers.LSTM(
                units=units,
                return_sequences=(i < len(self.config.lstm_units) - 1)
            ) for i, units in enumerate(self.config.lstm_units)],
            
            # Dense layers
            *[tf.keras.layers.Dense(
                units=units,
                activation='relu'
            ) for units in self.config.dense_units],
            
            # Dropout
            tf.keras.layers.Dropout(self.config.dropout_rate),
            
            # Output layer
            tf.keras.layers.Dense(
                self.config.num_classes,
                activation='softmax'
            )
        ])
        
        return model
