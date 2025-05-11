import tensorflow as tf
from .base_model import BaseModel

class CNNModel(BaseModel):
    def _build_model(self) -> tf.keras.Model:
        """Build CNN model architecture"""
        model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=self.config.input_shape),
            
            # CNN layers
            *[tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=3,
                activation='relu',
                padding='same'
            ) for filters in self.config.cnn_filters],
            
            # Global pooling
            tf.keras.layers.GlobalAveragePooling1D(),
            
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
