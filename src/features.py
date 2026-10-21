import librosa
import numpy as np


def extract_features(audio, sample_rate):
    """Extract the same feature vector used by the Colab Random Forest."""
    target_rate = 16000
    target_length = target_rate * 10

    audio = np.asarray(audio, dtype=np.float32)

    if sample_rate != target_rate:
        audio = librosa.resample(
            audio, orig_sr=sample_rate, target_sr=target_rate
        )
        sample_rate = target_rate

    # MIMII clips used for training are 10 seconds long.
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    rms = librosa.feature.rms(y=audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(audio)

    return np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        [rms.mean(), rms.std()],
        [centroid.mean(), centroid.std()],
        [zcr.mean(), zcr.std()],
    ]).astype(np.float32)