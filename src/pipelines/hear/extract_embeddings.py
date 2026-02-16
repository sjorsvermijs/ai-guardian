"""
Extract HeAR embeddings from the SPRSound dataset.

SPRSound structure (BioCAS2022):
  train2022_wav/   - 1,949 .wav files
  train2022_json/  - 1,949 .json annotations with record_annotation + event_annotation
  test2022_wav/    - 734 .wav files
  test2022_json/   - 734 .json annotations

Each JSON has:
  {
    "record_annotation": "Normal" | "CAS" | "DAS" | "CAS & DAS" | "Poor Quality",
    "event_annotation": [{"start": ms, "end": ms, "type": "Normal|Wheeze|..."}]
  }

This script extracts one HeAR embedding per audio file (averaged over 2s chunks)
and saves everything to a .npz file for fast classifier training.

Usage:
  python -m src.pipelines.hear.extract_embeddings
"""

import os
import sys
import json
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.pipelines.hear.pipeline import HeARPipeline


SPRSOUND_DIR = PROJECT_ROOT / "data" / "sprsound"
BIOCAS_DIR = SPRSOUND_DIR / "BioCAS2022"
OUTPUT_DIR = PROJECT_ROOT / "data" / "sprsound_embeddings"


def log(msg: str):
    """Print with flush so progress is visible in real-time."""
    print(msg, flush=True)


def load_annotation(json_path: Path) -> dict:
    """Load a SPRSound JSON annotation file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_samples(biocas_dir: Path) -> list[dict]:
    """Find all audio files and their labels from BioCAS2022."""
    samples = []

    splits = [
        ("train", biocas_dir / "train2022_wav", biocas_dir / "train2022_json"),
        ("test", biocas_dir / "test2022_wav", biocas_dir / "test2022_json"),
    ]

    for split_name, wav_dir, json_dir in splits:
        if not wav_dir.exists():
            log(f"  WARNING: {wav_dir} not found, skipping.")
            continue

        wav_files = sorted(wav_dir.glob("*.wav"))
        log(f"  [{split_name}] Found {len(wav_files)} wav files")

        # For test set, annotations might be in sub-folders
        json_dirs = [json_dir]
        for sub in json_dir.iterdir() if json_dir.exists() else []:
            if sub.is_dir():
                json_dirs.append(sub)

        # Build a lookup of all available annotations
        annotation_map = {}
        for jdir in json_dirs:
            if not jdir.exists():
                continue
            for jf in jdir.glob("*.json"):
                annotation_map[jf.stem] = jf

        for wav_path in wav_files:
            sample = {
                "path": wav_path,
                "split": split_name,
                "record_label": None,
                "event_labels": [],
            }

            # Try to find matching annotation
            json_path = annotation_map.get(wav_path.stem)
            if json_path:
                try:
                    ann = load_annotation(json_path)
                    sample["record_label"] = ann.get("record_annotation")
                    events = ann.get("event_annotation", [])
                    sample["event_labels"] = [e.get("type") for e in events if e.get("type")]
                except Exception as e:
                    log(f"  WARNING: Could not parse {json_path.name}: {e}")

            samples.append(sample)

    return samples


def extract_embeddings(samples: list[dict], pipeline: HeARPipeline):
    """Extract HeAR embeddings for all samples, reporting progress per file."""
    embeddings = []
    record_labels = []
    event_labels_list = []
    splits = []
    filenames = []
    failed = 0
    total = len(samples)
    start_time = time.time()

    for i, sample in enumerate(samples):
        elapsed = time.time() - start_time
        rate = (i / elapsed) if elapsed > 0 and i > 0 else 0
        eta = ((total - i) / rate) if rate > 0 else 0
        eta_str = f"{int(eta // 60)}m{int(eta % 60):02d}s" if rate > 0 else "?"

        log(f"  [{i + 1}/{total}] {sample['path'].name}  "
            f"({rate:.1f} files/s, ETA {eta_str})")

        try:
            audio_data, sample_rate = sf.read(sample["path"])

            # Convert stereo to mono
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            result = pipeline.process(audio_data)

            if result.data.get("embeddings"):
                # Average all chunk embeddings into a single vector per file
                file_embedding = np.mean(result.data["embeddings"], axis=0)
                embeddings.append(file_embedding)
                record_labels.append(sample["record_label"] or "Unknown")
                event_labels_list.append(",".join(sample["event_labels"]) if sample["event_labels"] else "Unknown")
                splits.append(sample["split"])
                filenames.append(sample["path"].name)
            else:
                log(f"    SKIP: no embeddings returned")
                failed += 1
        except Exception as e:
            log(f"    ERROR: {e}")
            failed += 1

    total_time = time.time() - start_time
    log(f"\n  Done! Extracted {len(embeddings)}/{total} in {total_time:.0f}s "
        f"({total_time / total:.2f}s/file, {failed} failed)")
    return np.array(embeddings), record_labels, event_labels_list, splits, filenames


def main():
    log("=" * 60)
    log("HeAR Embedding Extraction for SPRSound")
    log("=" * 60)

    if not BIOCAS_DIR.exists():
        log(f"SPRSound BioCAS2022 not found at {BIOCAS_DIR}")
        log("Run first: python -m src.pipelines.hear.download_sprsound")
        sys.exit(1)

    # Discover dataset
    log("\n1. Discovering dataset structure...")
    samples = discover_samples(BIOCAS_DIR)
    if not samples:
        log("No audio samples found!")
        sys.exit(1)

    # Show label distributions
    record_counts = Counter(s["record_label"] for s in samples if s["record_label"])
    log(f"\n  Record-level labels:")
    for label, count in sorted(record_counts.items(), key=lambda x: -x[1]):
        log(f"    {label}: {count}")

    event_counts = Counter(e for s in samples for e in s["event_labels"])
    log(f"\n  Event-level labels:")
    for label, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        log(f"    {label}: {count}")

    # Initialize HeAR
    log("\n2. Loading HeAR model...")
    pipeline = HeARPipeline({"sample_rate": 16000})
    if not pipeline.initialize():
        log("Failed to initialize HeAR pipeline!")
        sys.exit(1)

    # Extract embeddings
    log(f"\n3. Extracting embeddings from {len(samples)} files...")
    embeddings, record_labels, event_labels, splits, filenames = extract_embeddings(samples, pipeline)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "sprsound_hear_embeddings.npz"

    # Encode record labels as integers
    unique_record_labels = sorted(set(record_labels))
    record_label_to_idx = {l: i for i, l in enumerate(unique_record_labels)}
    record_label_indices = np.array([record_label_to_idx[l] for l in record_labels])

    np.savez(
        output_path,
        embeddings=embeddings,
        record_labels=record_label_indices,
        record_label_names=np.array(unique_record_labels),
        event_labels=np.array(event_labels),
        splits=np.array(splits),
        filenames=np.array(filenames),
    )

    log(f"\n4. Saved to {output_path}")
    log(f"   Embeddings shape: {embeddings.shape}")
    log(f"   Record labels: {unique_record_labels}")

    pipeline.cleanup()
    log("Done!")


if __name__ == "__main__":
    main()
