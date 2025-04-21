import tensorflow as tf
import numpy as np
import os

def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-6
    return (features - mean) / std

def save_model(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, "model.h5"))
    tf.saved_model.save(model, os.path.join(save_dir, "saved_model"))

def export_to_onnx(model, input_signature, export_path):
    import tf2onnx
    spec = (tf.TensorSpec(input_signature, tf.float32, name="input"),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    with open(export_path, "wb") as f:
        f.write(model_proto.SerializeToString())

def export_to_tflite(model, export_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(export_path, "wb") as f:
        f.write(tflite_model)