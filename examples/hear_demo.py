"""
Minimal HeAR Pipeline Demo
Processes a sample audio file using Google's Health Acoustic Representations model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import soundfile as sf
from src.pipelines.hear import HeARPipeline


def main():
    # 1. Load sample audio file
    audio_path = project_root / "data" / "sample" / "donate_a_cry_audio_1_belly_pain.wav"
    print(f"Loading audio: {audio_path.name}")

    audio_data, sample_rate = sf.read(audio_path)
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {len(audio_data) / sample_rate:.2f} seconds")

    # 2. Initialize HeAR pipeline (downloads model on first run)
    print("\nInitializing HeAR model...")
    pipeline = HeARPipeline({"sample_rate": sample_rate})
    pipeline.initialize()

    # 3. Process audio
    print("\nProcessing audio...")
    result = pipeline.process(audio_data)

    # 4. Display results
    print(f"\nResults:")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Chunks processed: {result.data.get('num_chunks', 0)}")
    print(f"  Embedding dimension: {result.data.get('embedding_dim', 0)}")

    if result.data.get("embeddings"):
        emb = result.data["embeddings"][0]
        print(f"  First embedding (shape {emb.shape}): [{emb[0]:.4f}, {emb[1]:.4f}, ..., {emb[-1]:.4f}]")

    # 5. Cleanup
    pipeline.cleanup()


if __name__ == "__main__":
    main()
