import argparse
import os
import json
import numpy as np
from ..config.config import Config
from ..data.data_loader import CapgMyoDataLoader
from ..data.preprocessing import EMGPreprocessor
from ..features.feature_extractor import EMGFeatureExtractor
from ..models.cnn_model import CNNModel
from ..models.lstm_model import LSTMModel
from ..models.hybrid_model import HybridModel
from ..utils.visualization import Visualizer
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate EMG gesture classification model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save evaluation results')
    return parser.parse_args()

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
        
        # Load and preprocess test data
        X_test, y_test = data_loader.load_test_data()
        X_processed, y_processed = preprocessor.preprocess_pipeline(X_test, y_test)
        
        # Extract features
        features = feature_extractor.extract_batch(X_processed)
        
        # Load model
        model = create_model(config.model)
        model.load_weights(args.model_path)
        
        # Evaluate model
        y_pred = model.model.predict(features)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Compute metrics
        metrics = compute_metrics(y_processed, y_pred_classes)
        
        # Save results
        with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Generate plots
        visualizer = Visualizer()
        visualizer.plot_confusion_matrix(
            y_processed,
            y_pred_classes,
            os.path.join(args.output_dir, 'confusion_matrix.png')
        )
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in evaluation pipeline: {str(e)}")
        raise

if __name__ == '__main__':
    main()
