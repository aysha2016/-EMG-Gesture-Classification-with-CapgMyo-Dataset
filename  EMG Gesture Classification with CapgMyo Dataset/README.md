
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
