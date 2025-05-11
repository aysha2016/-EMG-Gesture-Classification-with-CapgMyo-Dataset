import argparse
import os
from ..config.config import Config
from ..models.cnn_model import CNNModel
from ..models.lstm_model import LSTMModel
from ..models.hybrid_model import HybridModel
from ..utils.export import export_model
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Export EMG gesture classification model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model')
    parser.add_argument('--export_dir', type=str, required=True,
                      help='Directory to save exported model')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create model
        model = create_model(config.model)
        
        # Load weights
        model.load_weights(args.model_path)
        
        # Export model
        export_model(model, config.export)
        
        logger.info("Model export completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in model export: {str(e)}")
        raise

if __name__ == '__main__':
    main()
