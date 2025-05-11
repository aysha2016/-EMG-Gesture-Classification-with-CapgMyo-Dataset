import argparse
import os
from typing import Dict, Any
import json
from ..config.config import Config, DataConfig, ModelConfig, TrainingConfig, ExportConfig
from ..data.data_loader import CapgMyoDataLoader
from ..data.preprocessing import EMGPreprocessor
from ..features.feature_extractor import EMGFeatureExtractor
from ..models.cnn_model import CNNModel
from ..models.lstm_model import LSTMModel
from ..training.trainer import ModelTrainer
from ..utils.visualization import Visualizer
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train EMG gesture classification model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save results')
    return parser.parse_args()

def load_config(config_path: str) -> Config:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return Config(
        data=DataConfig(**config_dict['data']),
        model=ModelConfig(**config_dict['model']),
        training=TrainingConfig(**config_dict['training']),
        export=ExportConfig(**config_dict['export'])
    )

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Initialize components
        data_loader = CapgMyoDataLoader(config.data)
        preprocessor = EMGPreprocessor(config.data)
        feature_extractor = EMGFeatureExtractor(config.data)
        
        # Load and preprocess data
        X, y = data_loader.load_all_data()
        X_processed, y_processed = preprocessor.preprocess_pipeline(X, y)
        
        # Extract features
        features = feature_extractor.extract_batch(X_processed)
        
        # Create model
        if config.model.model_type == 'cnn':
            model = CNNModel(config.model)
        elif config.model.model_type == 'lstm':
            model = LSTMModel(config.model)
        else:
            raise ValueError(f"Unsupported model type: {config.model.model_type}")
        
        # Initialize trainer
        trainer = ModelTrainer(model, config.training)
        
        # Train model
        history = trainer.train(
            (features, y_processed),
            (features_val, y_val)
        )
        
        # Evaluate model
        metrics = trainer.evaluate((features_test, y_test))
        
        # Save results
        results = {
            'history': history,
            'metrics': metrics
        }
        
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Generate plots
        visualizer = Visualizer()
        visualizer.plot_training_history(
            history,
            os.path.join(args.output_dir, 'training_history.png')
        )
        visualizer.plot_confusion_matrix(
            y_test,
            metrics['predictions'],
            os.path.join(args.output_dir, 'confusion_matrix.png')
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    main()
