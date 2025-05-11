import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any
from sklearn.metrics import confusion_matrix
from .logging_utils import get_logger

logger = get_logger(__name__)

class Visualizer:
    @staticmethod
    def plot_training_history(history: Dict[str, Any], save_path: str):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Training history plot saved to {save_path}")
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: str, class_names: list = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Gesture {i+1}' for i in range(cm.shape[0])]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Confusion matrix plot saved to {save_path}")
    
    @staticmethod
    def plot_feature_importance(feature_importance: Dict[str, float],
                              save_path: str):
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        plt.barh(features, importance)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {save_path}")
