"""
Extract audio embeddings from the Donate a Cry dataset.

Supports multiple backends (ast, hubert).
Reads wav files from data/donate_a_cry/wav/ (label is filename prefix).

Usage:
  python -m src.pipelines.cry.extract_embeddings              # default: ast
  python -m src.pipelines.cry.extract_embeddings --backend hubert
"""

import sys
import time
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.pipelines.cry.pipeline import CryPipeline

WAV_DIR = PROJECT_ROOT / "data" / "donate_a_cry" / "wav"
OUTPUT_DIR = PROJECT_ROOT / "data" / "donate_a_cry_embeddings"


def log(msg: str):
    print(msg, flush=True)


def discover_samples(wav_dir: Path) -> list[dict]:
    """Find wav files and parse labels from filenames.

    Files are named: reason_originalname.wav
    """
    samples = []
    for wav_path in sorted(wav_dir.glob("*.wav")):
        label = wav_path.stem.split("_", 1)[0]
        samples.append({"path": wav_path, "label": label})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", default="ast", choices=["ast", "hubert"],
        help="Embedding model backend (default: ast)",
    )
    args = parser.parse_args()
    backend = args.backend

    log("=" * 60)
    log(f"Embedding Extraction for Donate a Cry  [backend={backend}]")
    log("=" * 60)

    if not WAV_DIR.exists():
        log(f"Wav files not found at {WAV_DIR}")
        log("Run first: python -m src.pipelines.cry.download_donate_a_cry")
        sys.exit(1)

    # Discover
    log("\n1. Discovering dataset...")
    samples = discover_samples(WAV_DIR)
    if not samples:
        log("No wav files found!")
        sys.exit(1)

    label_counts = Counter(s["label"] for s in samples)
    log(f"   Found {len(samples)} files, {len(label_counts)} classes")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        log(f"     {label}: {count}")

    # Initialize model
    log(f"\n2. Loading {backend} model...")
    pipeline = CryPipeline({"sample_rate": 16000, "model_backend": backend})
    if not pipeline.initialize():
        log("Failed to initialize pipeline!")
        sys.exit(1)

    # Extract
    log(f"\n3. Extracting embeddings from {len(samples)} files...")
    embeddings = []
    labels = []
    filenames = []
    failed = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        elapsed = time.time() - start_time
        rate = (i / elapsed) if elapsed > 0 and i > 0 else 0
        eta = ((len(samples) - i) / rate) if rate > 0 else 0
        eta_str = f"{int(eta // 60)}m{int(eta % 60):02d}s" if rate > 0 else "?"

        log(f"  [{i + 1}/{len(samples)}] {sample['path'].name}  "
            f"({rate:.1f} files/s, ETA {eta_str})")

        try:
            audio_data, sr = sf.read(sample["path"])
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            result = pipeline.process(audio_data)

            if result.data.get("embedding") is not None:
                embeddings.append(result.data["embedding"])
                labels.append(sample["label"])
                filenames.append(sample["path"].name)
            else:
                log(f"    SKIP: no embedding returned")
                failed += 1
        except Exception as e:
            log(f"    ERROR: {e}")
            failed += 1

    total_time = time.time() - start_time
    log(f"\n  Done! {len(embeddings)}/{len(samples)} in {total_time:.0f}s "
        f"({total_time / max(len(samples), 1):.2f}s/file, {failed} failed)")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"cry_{backend}_embeddings.npz"

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

    log(f"\n4. Saved to {output_path}")
    log(f"   Embeddings shape: {np.array(embeddings).shape}")
    log(f"   Labels: {unique_labels}")

    pipeline.cleanup()
    log("Done!")


if __name__ == "__main__":
    main()
