import tensorflow as tf
from .base_model import BaseModel

class HybridModel(BaseModel):
    def _build_model(self) -> tf.keras.Model:
        """Build hybrid CNN-LSTM model architecture"""
        inputs = tf.keras.layers.Input(shape=self.config.input_shape)
        
        # CNN branch
        x_cnn = inputs
        for filters in self.config.cnn_filters:
            x_cnn = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=3,
                activation='relu',
                padding='same'
            )(x_cnn)
            x_cnn = tf.keras.layers.BatchNormalization()(x_cnn)
            x_cnn = tf.keras.layers.MaxPooling1D(pool_size=2)(x_cnn)
        
        # LSTM branch
        x_lstm = inputs
        for i, units in enumerate(self.config.lstm_units):
            x_lstm = tf.keras.layers.LSTM(
                units=units,
                return_sequences=(i < len(self.config.lstm_units) - 1)
            )(x_lstm)
        
        # Combine branches
        x = tf.keras.layers.Concatenate()([x_cnn, x_lstm])
        
        # Dense layers
        for units in self.config.dense_units:
            x = tf.keras.layers.Dense(units, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(
            self.config.num_classes,
            activation='softmax'
        )(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
