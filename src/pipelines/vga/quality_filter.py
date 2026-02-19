"""
VGA — Quality Filter
=====================
Scans data/vga/<class>/deduped/, applies resolution and blur checks,
copies passing images to data/vga/<class>/final/.

Usage (from project root)
-------------------------
    python -m src.pipelines.vga.quality_filter
    python -m src.pipelines.vga.quality_filter --class chickenpox
    python -m src.pipelines.vga.quality_filter --no-blur-check
"""

import argparse
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def laplacian_variance(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_image(
    path: Path,
    min_w: int, min_h: int, max_w: int, max_h: int,
    min_kb: float, max_kb: float,
    blur_threshold: float, check_blur: bool,
) -> tuple[bool, str]:
    size_kb = path.stat().st_size / 1024
    if size_kb < min_kb:
        return False, f"too small ({size_kb:.1f} KB)"
    if size_kb > max_kb:
        return False, f"too large ({size_kb:.1f} KB)"

    try:
        with Image.open(path) as img:
            w, h = img.size
    except (UnidentifiedImageError, Exception) as exc:
        return False, f"cannot open: {exc}"

    if w < min_w or h < min_h:
        return False, f"resolution too low ({w}x{h})"
    if w > max_w or h > max_h:
        return False, f"resolution too high ({w}x{h})"

    if check_blur:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            return False, "OpenCV cannot read"
        lv = laplacian_variance(img_bgr)
        if lv < blur_threshold:
            return False, f"too blurry (Laplacian={lv:.1f})"

    return True, ""


def filter_class(
    deduped_dir: Path,
    out_dir: Path,
    quality_cfg: dict,
    check_blur: bool,
) -> tuple[int, int]:
    images = []
    for ext in IMAGE_EXTS:
        images.extend(deduped_dir.glob(f"*{ext}"))
        images.extend(deduped_dir.glob(f"*{ext.upper()}"))
    images = sorted(set(images))

    if not images:
        log.warning("No images in %s", deduped_dir)
        return 0, 0

    out_dir.mkdir(parents=True, exist_ok=True)
    passed = 0
    reasons: dict[str, int] = {}

    for src in tqdm(images, desc="Filtering", unit="img"):
        ok, reason = check_image(
            src,
            min_w=quality_cfg["min_width"],
            min_h=quality_cfg["min_height"],
            max_w=quality_cfg["max_width"],
            max_h=quality_cfg["max_height"],
            min_kb=quality_cfg["min_file_kb"],
            max_kb=quality_cfg["max_file_kb"],
            blur_threshold=quality_cfg["blur_threshold"],
            check_blur=check_blur,
        )
        if ok:
            dst = out_dir / f"{passed:06d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            passed += 1
        else:
            key = reason.split("(")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
            log.debug("Rejected %s: %s", src.name, reason)

    if reasons:
        log.info("  Rejection reasons:")
        for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            log.info("    %-30s %d", r, cnt)

    return len(images), passed


def main():
    p = argparse.ArgumentParser(description="VGA quality filter for scraped images")
    p.add_argument("--config", default=str(_DEFAULT_CONFIG))
    p.add_argument("--class", dest="only_class", default=None)
    p.add_argument("--no-blur-check", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    quality_cfg = cfg["quality"]
    output_root = _PROJECT_ROOT / cfg["output_dir"]
    check_blur = not args.no_blur_check

    classes = cfg["classes"]
    if args.only_class:
        classes = {args.only_class: classes[args.only_class]}

    summary = {}
    for class_name, class_cfg in classes.items():
        folder = class_cfg["folder"]
        deduped_dir = output_root / folder / "deduped"
        final_dir = output_root / folder / "final"

        log.info("=" * 60)
        log.info("Filtering class: %s", class_name.upper())
        total, kept = filter_class(deduped_dir, final_dir, quality_cfg, check_blur)
        summary[class_name] = {"total": total, "kept": kept}

    log.info("=" * 60)
    log.info("FINAL DATASET SUMMARY")
    grand_total = 0
    for cls, s in summary.items():
        pct = 100 * s["kept"] / max(s["total"], 1)
        log.info("  %-12s  %4d final  (%.0f%% of deduped)", cls, s["kept"], pct)
        grand_total += s["kept"]
    log.info("  %-12s  %4d total", "TOTAL", grand_total)
    log.info("Final images: data/vga/<class>/final/")
    log.info("Next: python -m src.pipelines.vga.finetune")


if __name__ == "__main__":
    main()
