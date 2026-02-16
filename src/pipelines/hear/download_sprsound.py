"""
Download and organize the SPRSound dataset.
Clones from: https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound

Usage:
  python -m src.pipelines.hear.download_sprsound
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "sprsound"


def main():
    if (DATA_DIR / "README.md").exists():
        print(f"SPRSound already exists at {DATA_DIR}")
        print("Delete the folder and re-run to re-download.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Cloning SPRSound dataset (this may take a while)...")
    subprocess.run(
        [
            "git", "clone",
            "https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound.git",
            str(DATA_DIR),
        ],
        check=True,
    )
    print(f"Dataset downloaded to {DATA_DIR}")

    # Summarize what we got
    wav_files = list(DATA_DIR.rglob("*.wav"))
    json_files = list(DATA_DIR.rglob("*.json"))
    print(f"\nFound {len(wav_files)} audio files")
    print(f"Found {len(json_files)} annotation files")

    # Show per-split counts
    for split_dir in sorted(DATA_DIR.rglob("*_wav")):
        count = len(list(split_dir.glob("*.wav")))
        print(f"  {split_dir.relative_to(DATA_DIR)}: {count} files")


if __name__ == "__main__":
    main()
