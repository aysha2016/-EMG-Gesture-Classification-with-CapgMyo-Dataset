import tensorflow as tf
import tf2onnx
import os
from typing import Optional
from .logging_utils import get_logger

logger = get_logger(__name__)

def export_to_tf(model: tf.keras.Model, export_dir: str):
    """Export model to TensorFlow format"""
    os.makedirs(export_dir, exist_ok=True)
    model.save(os.path.join(export_dir, 'model.h5'))
    tf.saved_model.save(model, os.path.join(export_dir, 'saved_model'))
    logger.info(f"Model exported to TensorFlow format in {export_dir}")

def export_to_onnx(model: tf.keras.Model, input_signature: tuple, export_path: str):
    """Export model to ONNX format"""
    spec = (tf.TensorSpec(input_signature, tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    
    with open(export_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    logger.info(f"Model exported to ONNX format at {export_path}")

def export_to_tflite(model: tf.keras.Model, export_path: str,
                    quantization: bool = False):
    """Export model to TFLite format"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open(export_path, "wb") as f:
        f.write(tflite_model)
    logger.info(f"Model exported to TFLite format at {export_path}")

def export_model(model: tf.keras.Model, config: 'ExportConfig',
                input_signature: Optional[tuple] = None):
    """Export model to specified format(s)"""
    if config.save_format == 'all':
        formats = ['tf', 'onnx', 'tflite']
    else:
        formats = [config.save_format]
    
    for fmt in formats:
        if fmt == 'tf':
            export_to_tf(model, os.path.join(config.export_dir, 'tf_model'))
        elif fmt == 'onnx':
            if input_signature is None:
                raise ValueError("input_signature is required for ONNX export")
            export_to_onnx(model, input_signature,
                         os.path.join(config.export_dir, 'model.onnx'))
        elif fmt == 'tflite':
            export_to_tflite(model,
                           os.path.join(config.export_dir, 'model.tflite'),
                           config.quantization)
