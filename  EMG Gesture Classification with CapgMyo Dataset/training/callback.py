import tensorflow as tf
from typing import List

def get_training_callbacks(config: 'TrainingConfig') -> List[tf.keras.callbacks.Callback]:
    """Get training callbacks"""
    callbacks = [
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=config.lr_reduction_factor,
            patience=config.lr_reduction_patience,
            min_lr=config.min_lr,
            verbose=1
        ),
        
        # Model checkpointing
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1
        )
    ]
    
    return callbacks
