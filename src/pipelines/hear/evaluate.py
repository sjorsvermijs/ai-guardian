"""
Evaluate the trained SPRSound classifier.

Usage:
  python -m src.pipelines.hear.evaluate info
  python -m src.pipelines.hear.evaluate classify <audio.wav>
  python -m src.pipelines.hear.evaluate classify <audio.wav> --binary
"""

import os
import sys
import json
import pickle
import numpy as np
import soundfile as sf
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

MODEL_DIR = PROJECT_ROOT / "models" / "sprsound_classifier"


def load_classifier(task="multiclass"):
    """Load trained classifier and metadata."""
    with open(MODEL_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)

    task_meta = metadata[task]
    model_file = MODEL_DIR / task_meta["model_file"]

    with open(model_file, "rb") as f:
        classifier = pickle.load(f)

    return classifier, task_meta["label_names"], metadata


def classify_audio(audio_path: str, task: str = "multiclass"):
    """Classify a single audio file using HeAR + trained classifier."""
    from src.pipelines.hear.pipeline import HeARPipeline

    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    # Load classifier
    classifier, label_names, metadata = load_classifier(task)
    print(f"Task: {task}")
    print(f"Classes: {label_names}")

    # Load HeAR
    print("Loading HeAR model...")
    pipeline = HeARPipeline({"sample_rate": 16000})
    pipeline.initialize()

    # Read audio
    audio_data, sr = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    print(f"Audio: {audio_path.name} ({len(audio_data)/sr:.1f}s, {sr}Hz)")

    # Extract embedding
    result = pipeline.process(audio_data)
    if not result.data.get("embeddings"):
        print("Failed to extract embeddings!")
        sys.exit(1)

    embedding = np.mean(result.data["embeddings"], axis=0).reshape(1, -1)

    # Predict
    prediction = classifier.predict(embedding)[0]
    probabilities = classifier.predict_proba(embedding)[0]

    print(f"\nPrediction: {label_names[prediction]}")
    print(f"\nClass probabilities:")
    for name, prob in sorted(zip(label_names, probabilities), key=lambda x: -x[1]):
        bar = "=" * int(prob * 40)
        print(f"  {name:>15}: {prob:.3f} {bar}")

    pipeline.cleanup()
    return label_names[prediction], dict(zip(label_names, probabilities.tolist()))


def show_info():
    """Show saved model performance metrics."""
    with open(MODEL_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)

    print("=" * 60)
    print("SPRSound Respiratory Sound Classifier")
    print("=" * 60)
    print(f"Feature extractor: {metadata['feature_extractor']}")
    print(f"Dataset: {metadata['dataset']}")
    print(f"Embedding dim: {metadata['embedding_dim']}")

    for task in ["multiclass", "binary"]:
        info = metadata[task]
        print(f"\n--- {task.upper()} ---")
        print(f"  Labels: {info['label_names']}")
        print(f"  Accuracy: {info['accuracy']:.4f}")
        print(f"  F1 (macro): {info['f1_macro']:.4f}")
        print(f"  Model: {info['model_file']}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.pipelines.hear.evaluate info")
        print("  python -m src.pipelines.hear.evaluate classify <audio.wav>")
        print("  python -m src.pipelines.hear.evaluate classify <audio.wav> --binary")
        sys.exit(1)

    command = sys.argv[1]

    if command == "info":
        show_info()
    elif command == "classify":
        if len(sys.argv) < 3:
            print("Please provide an audio file path.")
            sys.exit(1)
        audio_path = sys.argv[2]
        task = "binary" if "--binary" in sys.argv else "multiclass"
        classify_audio(audio_path, task)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
