# 🧠 EMG Gesture Classification with CapgMyo Dataset

This project implements an end-to-end EMG gesture classification pipeline using the **CapgMyo** dataset. It extracts signal features, trains CNN/LSTM models, evaluates performance, and exports to ONNX/TFLite for deployment.

---

## 📁 Project Structure

```
EMG_CapgMyo_Colab_Scripts/
├── feature_extraction.py     # Extracts PE, HG, HG+WF, HG+WE features
├── model.py                  # CNN & LSTM architectures
├── utils.py                  # Normalization, saving, ONNX/TFLite export
├── EMG_CapgMyo_Training.ipynb (optional if included)
```

---

## 📊 Features Extracted

| Abbreviation | Meaning                         | Description |
|-------------|----------------------------------|-------------|
| PE          | Power Envelope                   | Captures slow-varying energy of EMG |
| HG          | High Gamma Band (70–200 Hz)      | Fine motor control & muscle activation |
| HG + WF     | High Gamma + Waveform Features   | Combines RMS, ZCR with HG |
| HG + WE     | High Gamma + Wavelet Energy      | Multi-resolution frequency features |
| HG, PE      | High Gamma and Power Envelope    | Complementary slow and fast features |

---

## ✅ Model Options

- CNN: For spatial filtering of time-series EMG inputs.
- LSTM: For capturing temporal dependencies.

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

---

## 🚀 Export & Deployment

- ✅ Export to **ONNX** and **TFLite**
- ✅ Compatible with edge devices

---

## 📦 Google Colab Integration

- 📂 Mounts Google Drive automatically
- 📝 TensorBoard logs saved to `/logs`
- 📁 Models saved in `/models` on Drive

---

## 📥 Getting Started

1. Upload the zip to Google Colab.
2. Unzip and open `EMG_CapgMyo_Training.ipynb` (or import the scripts).
3. Run step-by-step:
   - Load CapgMyo data (sample code included).
   - Preprocess signals & extract features.
   - Train CNN/LSTM.
   - Evaluate with metrics.
   - Export model (ONNX / TFLite).
   - Save everything to Google Drive.

---

## 📚 Reference

- Dataset: [CapgMyo](https://doi.org/10.1109/TNSRE.2016.2528160)
- Example subset: Subject 1, Session 1 for quick training

---

Enjoy exploring muscle signals with deep learning! 💪✨

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/aysha2016/emg_gesture_classification.git
cd emg_gesture_classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset
The project uses the CapgMyo dataset, which contains EMG signals from 8 different hand gestures.

### Dataset Structure
- 8 subjects
- 2 sessions per subject
- 8 gestures per session
- 8 EMG channels
- 1000 Hz sampling rate

### Download Instructions
1. Visit the [CapgMyo Dataset website](https://example.com/capgmyo)
2. Download the dataset
3. Extract to `data/raw/` directory

## Usage

### 1. Data Preprocessing
```python
from data.data_loader import CapgMyoDataLoader
from data.preprocessing import EMGPreprocessor

# Initialize data loader
data_loader = CapgMyoDataLoader(data_dir="data/raw")
X, y = data_loader.load_data()

# Preprocess data
preprocessor = EMGPreprocessor()
X_processed, y_processed = preprocessor.preprocess_pipeline(X, y)
```

### 2. Feature Extraction
```python
from features.feature_extractor import EMGFeatureExtractor

# Extract features
feature_extractor = EMGFeatureExtractor()
features = feature_extractor.extract_batch(X_processed)
```

### 3. Model Training
```python
from models.cnn_model import CNNModel
from training.trainer import ModelTrainer

# Create and train model
model = CNNModel(input_shape=features.shape[1:], num_classes=8)
trainer = ModelTrainer(model)
history = trainer.train(X_train, y_train, X_val, y_val)
```

### 4. Model Evaluation
```python
# Evaluate model
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

### 5. Model Export
```python
from utils.export import export_model

# Export model to different formats
export_model(model, export_dir="models/exported")
```

### Command Line Interface
```bash
# Train model
python scripts/train.py --config config/config.json --output_dir results

# Evaluate model
python scripts/evaluate.py --model_path models/saved/best_model.h5

# Export model
python scripts/export_model.py --model_path models/saved/best_model.h5

# Visualize results
python scripts/visualize_results.py --results_dir results
```

 

## Real-time Processing
The system is designed for real-time EMG signal processing:
- Efficient feature extraction
- Optimized model architectures
- TFLite export for edge devices
- Low latency inference

 
  
