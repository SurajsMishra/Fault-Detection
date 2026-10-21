import io

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from src.inference import predict_cnn_wav, predict_rf_cnn_review, predict_wav



st.set_page_config(page_title="Machine Fault Detection")
st.title("Machine Fault Detection Using Deep Learning")
st.write("Upload a WAV file or record a short sound with your microphone.")

upload_col, record_col = st.columns(2)

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload a WAV file", type=["wav"], key="wav_upload"
    )

with record_col:
    recorded_audio = st.audio_input(
        "Record with your microphone", sample_rate=16_000, key="microphone_recording"
    )

# If both are present, let the user choose which one to analyze.
audio_input = None
if uploaded_file is not None and recorded_audio is not None:
    source = st.radio(
        "Choose audio to analyze",
        ["Uploaded file", "Microphone recording"],
        horizontal=True,
    )
    audio_input = uploaded_file if source == "Uploaded file" else recorded_audio
elif uploaded_file is not None:
    audio_input = uploaded_file
elif recorded_audio is not None:
    audio_input = recorded_audio

if audio_input is not None:
    audio_bytes = audio_input.getvalue()

    st.audio(audio_bytes, format="audio/wav")

    try:
        audio, sample_rate = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16_000,
            mono=True,
        )

        duration = len(audio) / sample_rate
        st.write(
            f"Sample rate: {sample_rate:,} Hz  |  Duration: {duration:.2f} seconds"
        )

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=64
        )
        mel_db = librosa.power_to_db(mel+1e-10, ref=np.max)
        fig, ax = plt.subplots(figsize=(12,4))
        image = librosa.display.specshow(
            mel_db,
            sr=sample_rate,
            x_axis = "time",
            y_axis = "mel",
            ax = ax
        )
        ax.set_title("Sound energy over time")
        ax.set_ylabel("Frequency band")
        fig.colorbar(image, ax=ax, label="Relative sound level (db)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Brighter areas indicate stronger sound energy in that time and frequency band. "
            "Use this plot as a visual aid; the model produces the fault prediction."
        )
        selected_model = st.selectbox("Choose model", ["CNN", "RandomForest", "RF-CNN review"])
        if st.button("Analyze selected audio"):
            if selected_model == "RF-CNN review":
                result = predict_rf_cnn_review(audio_bytes)

                col1, col2 = st.columns(2)
                col1.metric("RF anomaly score", f"{result['rf']['anomaly_score']:.1%}")
                col2.metric("CNN anomaly score", f"{result['cnn']['anomaly_score']:.1%}")

                if result["agreed"]:
                    if result["label"] == "Anomaly":
                        st.error("Both models predict Anomaly")
                    else:
                        st.success("Both models predict Normal")
                else:
                    st.warning(
                        f"Models disagree: RF says {result['rf']['label']}; "
                        f"CNN says {result['cnn']['label']}. Review this recording."
                    )

            elif selected_model == "CNN":
                result = predict_cnn_wav(audio_bytes)
                st.write(f"CNN result: {result['label']}")
                st.metric("CNN anomaly score", f"{result['anomaly_score']:.1%}")
                if result["label"] == "Anomaly":
                    st.error("CNN predicts Anomaly")
                else:
                    st.success("CNN predicts Normal")
            else:
                result = predict_wav(audio_bytes)
                st.write(f"Random Forest result: {result['label']}")
                st.metric("RF anomaly score", f"{result['anomaly_score']:.1%}")
                if result["label"] == "Anomaly":
                    st.error("Random Forest predicts Anomaly")
                else:
                    st.success("Random Forest predicts Normal")

            st.caption(
                "This model was trained on a small fan id_06 subset. "
                "Treat the result as an experiment, not an industrial diagnosis."
            )
    except Exception as error:
        st.error(f"Could not process this recording: {error}")