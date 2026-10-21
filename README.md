# Machine Fault Detection from Sound

A Streamlit application that analyzes machine audio and predicts whether it sounds normal or anomalous. This is an experimental project built using a small subset of the MIMII industrial machine sound dataset.

## Live Demo

[Open the Machine Fault Detection app](https://fault-detection-c2gq.streamlit.app/)

## Features

- Upload a WAV file or record audio using a microphone.
- View a log-Mel spectrogram of the recording.
- Analyze audio with a Random Forest model, a CNN, or RF-CNN review.
- In RF-CNN review mode, both models analyze the recording. If they disagree, the app flags it for human review.

## Models

**Random Forest** uses extracted audio features, including MFCCs, RMS energy, spectral centroid, and zero-crossing rate.

**CNN** classifies a log-Mel spectrogram. Its inference preprocessing uses the normalization values saved with the trained checkpoint.

**RF-CNN review** displays each model’s prediction and anomaly score. Agreement is not proof that the result is correct, and disagreement does not identify which model is right.

## Dataset and Results

The models were trained using a small MIMII subset containing fan machine `id_06`: 50 normal recordings and 20 anomalous recordings. The held-out test set contained 15 normal and 6 anomalous recordings.

| Model | Accuracy | Anomaly precision | Anomaly recall | Anomaly F1 |
|---|---:|---:|---:|---:|
| Random Forest | 86% | 1.00 | 0.50 | 0.67 |
| CNN | 90% | 1.00 | 0.67 | 0.80 |

Confusion matrices use true labels as rows and predicted labels as columns, ordered `[normal, anomaly]`:

- Random Forest: `[[15, 0], [3, 3]]`
- CNN: `[[15, 0], [2, 4]]`

These results are preliminary because the test set is small. The models can miss faults. Scores are experimental and should not be treated as calibrated probabilities or industrial diagnoses.

## Run Locally

### Requirements

- Python and pip
- The project files and trained model files listed below

### Windows PowerShell

From the project directory:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
