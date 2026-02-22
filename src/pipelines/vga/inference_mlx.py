"""
VGA — Standalone MLX Inference & Evaluation
=============================================
Load the MLX LoRA adapter and classify baby skin images on Apple Silicon.
The PyTorch version (inference.py) is kept for CUDA users.

Usage (from project root)
-------------------------
    python -m src.pipelines.vga.inference_mlx --image path/to/image.jpg
    python -m src.pipelines.vga.inference_mlx --dir path/to/folder/
    python -m src.pipelines.vga.inference_mlx --eval
"""

import argparse
import logging
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError

_DIR = Path(__file__).parent
_PROJECT_ROOT = _DIR.parent.parent.parent

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Use 4-bit quantized model for inference to fit in 8GB unified memory.
# Training uses bf16 (in finetune_mlx.py) since mlx-vlm's LoRA trainer
# handles memory more efficiently during training.
MODEL_ID = "mlx-community/medgemma-1.5-4b-it-4bit"

LABELS = ["healthy", "eczema", "chickenpox"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

USER_QUESTION = (
    "This is a photo of a baby's skin. "
    "Classify it as exactly one of: healthy, eczema, chickenpox. "
    "Reply with ONLY one word: healthy, eczema, or chickenpox."
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(adapter_path: str):
    """Load MedGemma 1.5 4B via mlx-vlm with optional LoRA adapter."""
    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    adapter_file = Path(adapter_path) / "adapters.safetensors"

    if adapter_file.exists():
        log.info("Loading model with MLX adapter from %s ...", adapter_path)
        model, processor = load(
            MODEL_ID,
            adapter_path=adapter_path,
            tokenizer_config={"trust_remote_code": True},
        )
    else:
        log.info("No adapter found at %s — loading base model only.", adapter_path)
        model, processor = load(
            MODEL_ID,
            tokenizer_config={"trust_remote_code": True},
        )

    config = load_config(MODEL_ID)
    log.info("Model ready (MLX / Apple Silicon)")
    return model, processor, config


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model, processor, config, image_path: str) -> tuple[str, str]:
    """Predict skin condition for a single image."""
    import tempfile
    from mlx_vlm import generate

    # Resize image to 224x224 to reduce GPU memory usage
    img = Image.open(image_path).convert("RGB")
    if img.width > 224 or img.height > 224:
        img = img.resize((224, 224), Image.BILINEAR)
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        img.save(tmp.name, quality=95)
        use_path = tmp.name
    else:
        use_path = image_path

    # Build prompt matching training format: image token BEFORE text
    # (processor.apply_chat_template puts image first, prompt_utils puts it last)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": USER_QUESTION},
        ]},
    ]
    formatted_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    result = generate(
        model, processor, formatted_prompt,
        [use_path], max_tokens=10, verbose=False,
    )
    # mlx-vlm may return a GenerationResult object or a string
    raw = (result.text if hasattr(result, 'text') else str(result)).strip().lower()

    # Clean up temp file
    if use_path != image_path:
        os.unlink(use_path)

    pred = "unknown"
    for label in LABELS:
        if label in raw:
            pred = label
            break

    return pred, raw


def predict_dir(model, processor, config, img_dir: Path):
    """Classify all images in a directory."""
    from collections import Counter

    paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not paths:
        log.warning("No images found in %s", img_dir)
        return

    log.info("Classifying %d images in %s ...", len(paths), img_dir)
    results = {}
    for path in paths:
        try:
            Image.open(path).convert("RGB")
        except (UnidentifiedImageError, Exception) as e:
            log.warning("  Skipping %s: %s", path.name, e)
            continue

        pred, raw = predict(model, processor, config, str(path))
        results[path.name] = pred
        print(f"  {path.name:<40}  -> {pred}  (raw: '{raw}')")

    counts = Counter(results.values())
    print("\nSummary:")
    for label in LABELS + ["unknown"]:
        if counts[label]:
            print(f"  {label}: {counts[label]}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, processor, config, data_dir: Path):
    """Evaluate on the test split of the dataset."""
    all_samples = []
    for label in LABELS:
        final_dir = data_dir / label / "final"
        if not final_dir.exists():
            log.warning("Missing: %s", final_dir)
            continue
        paths = [p for p in final_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        for p in paths:
            all_samples.append({"path": str(p), "label": label})

    random.seed(42)
    random.shuffle(all_samples)
    n = len(all_samples)
    test_samples = all_samples[int(n * 0.8) + int(n * 0.1):]

    log.info("Evaluating on %d test samples ...", len(test_samples))

    correct = 0
    total = 0
    per_class = {l: {"correct": 0, "total": 0} for l in LABELS}
    errors = []

    import gc
    import mlx.core as mx

    for i, sample in enumerate(test_samples):
        label = sample["label"]
        try:
            Image.open(sample["path"]).convert("RGB")
        except Exception:
            continue

        pred, raw = predict(model, processor, config, sample["path"])
        is_correct = pred == label
        correct += int(is_correct)
        total += 1
        per_class[label]["total"] += 1
        per_class[label]["correct"] += int(is_correct)

        if not is_correct:
            errors.append((Path(sample["path"]).name, label, pred, raw))

        # Free GPU memory between predictions to avoid OOM
        mx.metal.clear_cache()
        if i % 5 == 0:
            gc.collect()
            log.info("  [%d/%d] acc so far: %.1f%%", total, len(test_samples),
                     correct / max(total, 1) * 100)

    acc = correct / max(total, 1) * 100
    print("\n" + "=" * 55)
    print(f"Test accuracy: {acc:.1f}%  ({correct}/{total})")
    print("=" * 55)
    for l in LABELS:
        c = per_class[l]
        pct = c["correct"] / max(c["total"], 1) * 100
        bar = "#" * int(pct / 5)
        print(f"  {l:<12}  {pct:5.1f}%  ({c['correct']:>2}/{c['total']:>2})  {bar}")
    print("=" * 55)

    if errors:
        print(f"\nMisclassified ({len(errors)}):")
        for name, true, pred, raw in errors[:20]:
            print(f"  {name:<40}  true={true:<12} pred={pred} raw='{raw}'")

    return acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="VGA — MLX inference with fine-tuned MedGemma"
    )
    p.add_argument("--adapter", default=str(
        _PROJECT_ROOT / "models" / "vga_skin_adapter_mlx"
    ))
    p.add_argument("--image", default=None)
    p.add_argument("--dir", default=None)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--data-dir", default="data/vga")
    return p.parse_args()


def main():
    args = parse_args()

    model, processor, config = load_model(args.adapter)

    if args.image:
        path = Path(args.image)
        try:
            Image.open(path).convert("RGB")
        except Exception as e:
            log.error("Cannot open image %s: %s", path, e)
            return
        pred, raw = predict(model, processor, config, str(path))
        print(f"\nPrediction: {pred.upper()}")
        print(f"Raw output: '{raw}'")

    elif args.dir:
        predict_dir(model, processor, config, Path(args.dir))

    elif args.eval:
        evaluate(model, processor, config, _PROJECT_ROOT / args.data_dir)

    else:
        print("Error: specify --image, --dir, or --eval")


if __name__ == "__main__":
    main()
