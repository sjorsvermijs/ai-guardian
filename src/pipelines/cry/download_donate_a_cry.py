"""
Download the Donate a Cry corpus and convert audio to wav format.
Source: https://github.com/gveres/donateacry-corpus

Filename convention: UUID-timestamp-version-gender-age-reason.extension
Reason codes: hu=hungry, bu=burping, bp=belly_pain, dc=discomfort,
              ti=tired, lo=lonely, ch=cold_hot, sc=scared, dk=unknown

Usage:
  python -m src.pipelines.cry.download_donate_a_cry
"""

import subprocess
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "donate_a_cry"
RAW_DIR = DATA_DIR / "raw"
WAV_DIR = DATA_DIR / "wav"

REASON_CODES = {
    "hu": "hungry",
    "bu": "burping",
    "bp": "belly_pain",
    "dc": "discomfort",
    "ti": "tired",
    "lo": "lonely",
    "ch": "cold_hot",
    "sc": "scared",
    "dk": "unknown",
}


def parse_filename(filename: str) -> dict:
    """Parse Donate a Cry filename to extract metadata.

    Format: UUID-timestamp-version-gender-age-reason.ext
    Example: 0D1AD73E-...-1.0-m-04-hu.caf
    """
    stem = Path(filename).stem
    parts = stem.rsplit("-", 3)  # Split from right to get gender-age-reason

    if len(parts) < 4:
        return {"gender": "unknown", "age": "unknown", "reason": "unknown"}

    reason_code = parts[-1]
    age = parts[-2]
    gender = parts[-3]

    return {
        "gender": "male" if gender == "m" else "female" if gender == "f" else gender,
        "age": age,
        "reason_code": reason_code,
        "reason": REASON_CODES.get(reason_code, "unknown"),
    }


def convert_to_wav(input_path: Path, output_path: Path) -> bool:
    """Convert audio file to 16kHz mono wav using ffmpeg or soundfile."""
    try:
        # Try ffmpeg first (handles .caf and .3gp)
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(input_path),
                "-ar", "16000", "-ac", "1",
                str(output_path),
            ],
            capture_output=True,
            timeout=30,
        )
        return result.returncode == 0
    except FileNotFoundError:
        # ffmpeg not installed, try soundfile/pydub as fallback
        try:
            import soundfile as sf
            import numpy as np
            from scipy import signal

            data, sr = sf.read(input_path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            if sr != 16000:
                new_length = int(len(data) * 16000 / sr)
                data = signal.resample(data, new_length)
            sf.write(output_path, data, 16000)
            return True
        except Exception as e:
            print(f"  Could not convert {input_path.name}: {e}", flush=True)
            return False


def main():
    # Clone the corpus
    if (RAW_DIR / "README.md").exists() or (RAW_DIR / "donateacry_corpus_cleaned_and_updated_data").exists():
        print(f"Donate a Cry already downloaded at {RAW_DIR}")
    else:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print("Cloning Donate a Cry corpus...")
        subprocess.run(
            [
                "git", "clone",
                "https://github.com/gveres/donateacry-corpus.git",
                str(RAW_DIR),
            ],
            check=True,
        )
        print(f"Downloaded to {RAW_DIR}")

    # Find all audio files (cleaned dataset preferred)
    cleaned_dir = RAW_DIR / "donateacry_corpus_cleaned_and_updated_data"
    if cleaned_dir.exists():
        source_dir = cleaned_dir
    else:
        source_dir = RAW_DIR

    audio_files = []
    for ext in ["*.caf", "*.3gp", "*.wav", "*.ogg", "*.mp3"]:
        audio_files.extend(source_dir.rglob(ext))

    print(f"\nFound {len(audio_files)} audio files")

    # Parse labels
    label_counts = Counter()
    for f in audio_files:
        meta = parse_filename(f.name)
        label_counts[meta["reason"]] += 1

    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    # Convert to wav
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    converted = 0
    failed = 0

    print(f"\nConverting to 16kHz mono wav...")
    for i, audio_path in enumerate(audio_files):
        meta = parse_filename(audio_path.name)
        # Save as: reason_originalname.wav
        wav_name = f"{meta['reason']}_{audio_path.stem}.wav"
        wav_path = WAV_DIR / wav_name

        if wav_path.exists():
            converted += 1
            continue

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i + 1}/{len(audio_files)}] {audio_path.name}", flush=True)

        if convert_to_wav(audio_path, wav_path):
            converted += 1
        else:
            failed += 1

    print(f"\nDone! Converted: {converted}, Failed: {failed}")
    print(f"Wav files saved to {WAV_DIR}")


if __name__ == "__main__":
    main()
