"""
Extract MFCC-based features from the Donate a Cry dataset.

Features per file: 13 MFCCs + 13 deltas + 13 delta-deltas + spectral features
= 193 features (mean + std over time)

This is the approach reported by multiple papers to achieve 90%+ on cry data.

Usage:
  python -m src.pipelines.cry.extract_mfcc
"""

import sys
import time
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
WAV_DIR = PROJECT_ROOT / "data" / "donate_a_cry" / "wav"
OUTPUT_DIR = PROJECT_ROOT / "data" / "donate_a_cry_embeddings"

SAMPLE_RATE = 16000


def log(msg: str):
    print(msg, flush=True)


def extract_features(audio_data: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract a comprehensive feature vector from audio.

    Returns a 1D feature vector combining:
    - MFCCs (13) + deltas + delta-deltas: mean + std = 78 features
    - Chroma (12): mean + std = 24 features
    - Spectral contrast (7): mean + std = 14 features
    - Spectral centroid (1): mean + std = 2 features
    - Spectral bandwidth (1): mean + std = 2 features
    - Spectral rolloff (1): mean + std = 2 features
    - Zero crossing rate (1): mean + std = 2 features
    - RMS energy (1): mean + std = 2 features
    - Mel spectrogram (128): mean + std = 256 features (trimmed to top 40 bands)
    Total: ~206 features
    """
    features = []

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    for feat in [mfcc, mfcc_delta, mfcc_delta2]:
        features.append(feat.mean(axis=1))
        features.append(feat.std(axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    features.append(chroma.mean(axis=1))
    features.append(chroma.std(axis=1))

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
    features.append(contrast.mean(axis=1))
    features.append(contrast.std(axis=1))

    # Scalar features (shape: (1, T) → take mean + std)
    for feat_fn in [
        lambda: librosa.feature.spectral_centroid(y=audio_data, sr=sr),
        lambda: librosa.feature.spectral_bandwidth(y=audio_data, sr=sr),
        lambda: librosa.feature.spectral_rolloff(y=audio_data, sr=sr),
        lambda: librosa.feature.zero_crossing_rate(y=audio_data),
        lambda: librosa.feature.rms(y=audio_data),
    ]:
        feat = feat_fn()
        features.append(np.array([feat.mean()]))
        features.append(np.array([feat.std()]))

    # Mel spectrogram (top 40 bands by variance)
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.append(mel_db.mean(axis=1))
    features.append(mel_db.std(axis=1))

    return np.concatenate(features)


def discover_samples(wav_dir: Path) -> list[dict]:
    samples = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        label = wav_path.stem.split("_", 1)[0]
        samples.append({"path": wav_path, "label": label})
    return samples


def main():
    log("=" * 60)
    log("MFCC Feature Extraction for Donate a Cry")
    log("=" * 60)

    if not WAV_DIR.exists():
        log(f"Wav files not found at {WAV_DIR}")
        sys.exit(1)

    samples = discover_samples(WAV_DIR)
    label_counts = Counter(s["label"] for s in samples)
    log(f"Found {len(samples)} files, {len(label_counts)} classes")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        log(f"  {label}: {count}")

    embeddings = []
    labels = []
    filenames = []
    failed = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 1 or i == len(samples) - 1:
            elapsed = time.time() - start_time
            rate = (i / elapsed) if elapsed > 0 and i > 0 else 0
            eta = ((len(samples) - i) / rate) if rate > 0 else 0
            eta_str = f"{int(eta // 60)}m{int(eta % 60):02d}s" if rate > 0 else "?"
            log(f"  [{i + 1}/{len(samples)}] ({rate:.1f} files/s, ETA {eta_str})")

        try:
            audio_data, sr = sf.read(sample["path"])
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            # Resample if needed
            if sr != SAMPLE_RATE:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=SAMPLE_RATE)

            features = extract_features(audio_data, SAMPLE_RATE)
            embeddings.append(features)
            labels.append(sample["label"])
            filenames.append(sample["path"].name)
        except Exception as e:
            log(f"    ERROR on {sample['path'].name}: {e}")
            failed += 1

    total_time = time.time() - start_time
    log(f"\nDone! {len(embeddings)}/{len(samples)} in {total_time:.0f}s ({failed} failed)")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "cry_mfcc_embeddings.npz"

    unique_labels = sorted(set(labels))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    label_indices = np.array([label_to_idx[l] for l in labels])

    np.savez(
        output_path,
        embeddings=np.array(embeddings),
        labels=label_indices,
        label_names=np.array(unique_labels),
        filenames=np.array(filenames),
    )

    log(f"Saved to {output_path}")
    log(f"Feature dim: {np.array(embeddings).shape[1]}")


if __name__ == "__main__":
    main()
