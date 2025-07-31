## Site Address
https://huggingface.co/spaces/PushpakRaj/Ecg_model_interface

#  ECG + Metadata Multi-Class Classifier

This Gradio-based web app performs **multi-label classification** on ECG signal data and patient metadata using a pre-trained 1D CNN model. It predicts common cardiovascular conditions like:

- `NORM` (Normal)
- `MI` (Myocardial Infarction)
- `STTC` (ST/T Changes)
- `CD` (Conduction Disturbances)
- `HYP` (Hypertrophy)

---
##  Features

- Upload **metadata CSV** and **ECG signal `.npy` array**
- Preprocessing using trained `StandardScaler` for both inputs
- Multi-label prediction using a 1D CNN model
- Thresholding at `0.5` to convert probabilities into binary classes
- **Downloadable CSV** file with predictions

---

## Model Architecture

- Built using `TensorFlow` and `Keras`
- Multi-input 1D CNN for:
  - ECG signals: `(5000, 12)` input shape
  - Metadata: numerical tabular features
- Trained on the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.1/) dataset (or similar)

---

## File Structure
.

├── app.py # Main Gradio interface 

├── ecg_1DCNN_model.h5 # Trained Keras model

├── scaler_metadata.joblib # Scaler for metadata

├── scaler_ecg.joblib # Scaler for ECG signal

├── requirements.txt # Python dependencies

└── README.md # This file


---

##  Input Format

### Metadata (`.csv`)
- Each row = one patient
- Columns = numerical metadata features used during training

### ECG Signal (`.npy`)
- Shape: `(n_samples, 5000, 12)`
- Format: 32-bit floating point NumPy array

---

##  How to Run Locally

```bash
pip install -r requirements.txt
python app.py 
