from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Dict, Any, Tuple

class BaseModel(ABC):
    def __init__(self, config: 'ModelConfig'):
        self.config = config
        self.model = self._build_model()
        
    @abstractmethod
    def _build_model(self) -> tf.keras.Model:
        """Build the model architecture"""
        pass
    
    def compile_model(self):
        """Compile the model with optimizer and loss"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def get_model_summary(self) -> str:
        """Get model summary as string"""
        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return '\n'.join(string_list)
    
    def save_weights(self, filepath: str):
        """Save model weights"""
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath: str):
        """Load model weights"""
        self.model.load_weights(filepath) 