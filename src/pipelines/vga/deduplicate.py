"""
VGA — Perceptual Hash Deduplication
====================================
Scans data/vga/<class>/raw/**, groups near-duplicates, keeps one per group,
copies survivors to data/vga/<class>/deduped/.

Usage (from project root)
-------------------------
    python -m src.pipelines.vga.deduplicate
    python -m src.pipelines.vga.deduplicate --class eczema
    python -m src.pipelines.vga.deduplicate --threshold 6
"""

import argparse
import logging
import shutil
from pathlib import Path

import imagehash
import yaml
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_DIR = Path(__file__).parent
_PROJECT_ROOT = _DIR.parent.parent.parent
_DEFAULT_CONFIG = _DIR / "config.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def hash_fn(algo: str, size: int):
    algos = {
        "phash": imagehash.phash,
        "dhash": imagehash.dhash,
        "ahash": imagehash.average_hash,
        "whash": imagehash.whash,
    }
    fn = algos.get(algo, imagehash.phash)
    return lambda img: fn(img, hash_size=size)


def collect_images(raw_dir: Path) -> list[Path]:
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(raw_dir.rglob(f"*{ext}"))
        paths.extend(raw_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def deduplicate(
    raw_dir: Path,
    out_dir: Path,
    algo: str,
    hash_size: int,
    hamming_threshold: int,
) -> tuple[int, int]:
    images = collect_images(raw_dir)
    if not images:
        log.warning("No images found in %s", raw_dir)
        return 0, 0

    log.info("Hashing %d images...", len(images))
    get_hash = hash_fn(algo, hash_size)

    hashes: list[tuple[Path, object]] = []
    for p in tqdm(images, desc="Hashing", unit="img"):
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                h = get_hash(img)
            hashes.append((p, h))
        except (UnidentifiedImageError, Exception) as exc:
            log.debug("Skip %s: %s", p.name, exc)

    log.info("  Valid images: %d", len(hashes))

    kept: list[Path] = []
    seen_hashes: list[object] = []

    for path, h in tqdm(hashes, desc="Deduping", unit="img"):
        is_dup = any((h - sh) <= hamming_threshold for sh in seen_hashes)
        if not is_dup:
            kept.append(path)
            seen_hashes.append(h)

    log.info(
        "  Kept %d / %d  (removed %d duplicates)",
        len(kept), len(hashes), len(hashes) - len(kept),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(tqdm(kept, desc="Copying", unit="img")):
        suffix = src.suffix.lower()
        if suffix == ".webp":
            suffix = ".jpg"
        dst = out_dir / f"{i:06d}{suffix}"
        if src.suffix.lower() == ".webp":
            with Image.open(src) as img:
                img.convert("RGB").save(dst, "JPEG", quality=92)
        else:
            shutil.copy2(src, dst)

    return len(hashes), len(kept)


def main():
    p = argparse.ArgumentParser(description="VGA perceptual-hash deduplication")
    p.add_argument("--config", default=str(_DEFAULT_CONFIG))
    p.add_argument("--class", dest="only_class", default=None)
    p.add_argument("--threshold", type=int, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    dedup_cfg = cfg["dedup"]
    algo = dedup_cfg["hash_algorithm"]
    hash_size = dedup_cfg["hash_size"]
    threshold = args.threshold if args.threshold is not None else dedup_cfg["hamming_threshold"]
    output_root = _PROJECT_ROOT / cfg["output_dir"]

    classes = cfg["classes"]
    if args.only_class:
        classes = {args.only_class: classes[args.only_class]}

    summary = {}
    for class_name, class_cfg in classes.items():
        folder = class_cfg["folder"]
        raw_dir = output_root / folder / "raw"
        out_dir = output_root / folder / "deduped"

        log.info("=" * 60)
        log.info("Deduplicating class: %s", class_name.upper())
        total, kept = deduplicate(raw_dir, out_dir, algo, hash_size, threshold)
        summary[class_name] = {"total": total, "kept": kept}

    log.info("=" * 60)
    log.info("SUMMARY")
    for cls, s in summary.items():
        pct = 100 * s["kept"] / max(s["total"], 1)
        log.info("  %-12s  %4d / %4d  (%.0f%% kept)", cls, s["kept"], s["total"], pct)

    log.info("Next: python -m src.pipelines.vga.quality_filter")


if __name__ == "__main__":
    main()
