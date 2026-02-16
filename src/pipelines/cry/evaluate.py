"""
Evaluate the trained cry classifier or classify a new audio file.

Usage:
  python -m src.pipelines.cry.evaluate info
  python -m src.pipelines.cry.evaluate classify <audio.wav>
"""

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

MODEL_DIR = PROJECT_ROOT / "models" / "cry_classifier"


def load_classifier():
    with open(MODEL_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)
    with open(MODEL_DIR / metadata["model_file"], "rb") as f:
        classifier = pickle.load(f)
    return classifier, metadata["label_names"], metadata


def classify_audio(audio_path: str):
    """Classify a baby cry audio file."""
    from src.pipelines.cry.pipeline import CryPipeline

    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    classifier, label_names, metadata = load_classifier()
    print(f"Classes: {label_names}")

    print("Loading AST model...")
    pipeline = CryPipeline({"sample_rate": 16000})
    pipeline.initialize()

    audio_data, sr = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    print(f"Audio: {audio_path.name} ({len(audio_data)/sr:.1f}s, {sr}Hz)")

    result = pipeline.process(audio_data)
    if result.data.get("embedding") is None:
        print("Failed to extract embedding!")
        sys.exit(1)

    embedding = result.data["embedding"].reshape(1, -1)

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
    with open(MODEL_DIR / "metadata.json", "r") as f:
        metadata = json.load(f)

    print("=" * 60)
    print("Baby Cry Classifier")
    print("=" * 60)
    for key, value in metadata.items():
        if key != "model_file":
            print(f"  {key}: {value}")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.pipelines.cry.evaluate info")
        print("  python -m src.pipelines.cry.evaluate classify <audio.wav>")
        sys.exit(1)

    command = sys.argv[1]
    if command == "info":
        show_info()
    elif command == "classify":
        if len(sys.argv) < 3:
            print("Please provide an audio file path.")
            sys.exit(1)
        classify_audio(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
