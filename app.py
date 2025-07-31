import numpy as np
import pandas as pd
import tensorflow as tf
import gradio as  gr
from tensorflow import keras 
import joblib
import io
import tempfile


model = tf.keras.models.load_model("ecg_1DCNN_model.h5")
label = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
def preprocess(data , flag):
    if flag == 1:
        scaler = joblib.load("scaler_metadata.joblib")  # Load trained scaler
        scaled_ = pd.DataFrame(scaler.transform(data) , columns=data.columns)
    elif flag == 2:
        scaler = joblib.load("scaler_ecg.joblib")  # Load trained scaler
        scaled_ = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    else:
        raise ValueError("unsupported flag. Use 1 for tabular data 2 for array")

    return scaled_ 

def main(metadata , ecg):
    
    metadata_df = pd.read_csv(metadata)
    ecg_np = np.load(ecg)

    scaled_metadata = preprocess(metadata_df , 1)
    scaled_ecg = preprocess(ecg_np , 2)

    preds = model.predict([scaled_metadata , scaled_ecg])
    
    final_pred = []
    for row in preds:
        tmpp = []
        for p in row:
            if(p>0.5):
                tmpp.append(1)
            else:
                tmpp.append(0)
        final_pred.append(tmpp)
                  

    preds_df = pd.DataFrame(final_pred, columns=label)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="") as tmp:
            preds_df.to_csv(tmp.name, index=False)
            return tmp.name



iface = gr.Interface(
    fn=main,
    inputs=[
        gr.File(label="Upload Metadata CSV"),
        gr.File(label="Upload ECG Signal array")
    ],
    outputs=gr.File(label="Download the predictions"),
    title="ECG + Metadata Multi-Class Classifier",
    description="Upload patient metadata and ECG signals (both CSV files) to get predictions."
)

if __name__ == "__main__":
    iface.launch()
