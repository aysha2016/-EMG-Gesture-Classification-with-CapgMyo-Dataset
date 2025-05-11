from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class DataConfig:
    data_dir: str
    window_size: int = 1000
    overlap: float = 0.5
    sampling_rate: int = 1000
    num_channels: int = 8
    num_gestures: int = 8
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42

@dataclass
class ModelConfig:
    model_type: str  # 'cnn', 'lstm', or 'hybrid'
    input_shape: Tuple[int, ...]
    num_classes: int
    learning_rate: float = 0.001
    dropout_rate: float = 0.3
    cnn_filters: Tuple[int, ...] = (64, 128, 256)
    lstm_units: Tuple[int, ...] = (64, 32)
    dense_units: Tuple[int, ...] = (128, 64)

@dataclass
class TrainingConfig:
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10
    lr_reduction_patience: int = 5
    lr_reduction_factor: float = 0.5
    min_lr: float = 1e-6

@dataclass
class ExportConfig:
    export_dir: str
    save_format: str = 'all'  # 'tf', 'onnx', 'tflite', or 'all'
    quantization: bool = False
    target_platform: Optional[str] = None

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    export: ExportConfig
