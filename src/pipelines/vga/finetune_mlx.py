"""
VGA — LoRA Fine-tuning on Apple Silicon with MLX
==================================================
Trains a LoRA adapter on MedGemma 1.5 4B using mlx-vlm to classify
baby skin images as:  healthy | eczema | chickenpox

Designed for Apple Silicon Macs (M1/M2/M3/M4, unified memory).
The PyTorch version (finetune.py) is kept for CUDA users.

Replicates the same training configuration as finetune.py:
- Same data split (80/10/10, seed=42)
- Same LoRA config (rank=8, alpha=16, dropout=0.05)
- Same augmentations (flip, rotation, brightness, contrast, blur)
- Same effective batch size (1 x 8 grad accum = 8)
- Same learning rate (2e-4) and epochs (3)

Usage (from project root)
-------------------------
    python -m src.pipelines.vga.finetune_mlx
    python -m src.pipelines.vga.finetune_mlx --epochs 5 --lora-rank 16
    python -m src.pipelines.vga.finetune_mlx --max-steps 10   # smoke test
    python -m src.pipelines.vga.finetune_mlx --eval-only --adapter-path runs/vga_mlx/lora_adapter
    python -m src.pipelines.vga.finetune_mlx --no-augment      # skip augmentation
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError

_DIR = Path(__file__).parent
_PROJECT_ROOT = _DIR.parent.parent.parent

load_dotenv(_PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Use 4-bit quantized model — fits in 8GB unified memory for both training
# and inference. The bf16 version requires >10GB and causes OOM during inference.
MODEL_ID = "mlx-community/medgemma-1.5-4b-it-4bit"

LABELS = ["healthy", "eczema", "chickenpox"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

USER_QUESTION = (
    "This is a photo of a baby's skin. "
    "Classify it as exactly one of: healthy, eczema, chickenpox. "
    "Reply with ONLY one word: healthy, eczema, or chickenpox."
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Pre-resize all training images to this size before feeding to mlx-vlm.
# This avoids the slow per-step 896×896→224×224 resize inside the trainer
# and works around the --image-resize-shape bug in mlx-vlm 0.3.12.
TRAIN_IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------------------------
# Augmentation (identical to finetune.py)
# ---------------------------------------------------------------------------

def augment_image(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply the same augmentation pipeline as the CUDA finetune.py."""
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    angle = rng.uniform(-20, 20)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.10:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.0)))
    return img


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_split(data_dir: Path, split_ratios=(0.8, 0.1, 0.1), seed=42):
    """Load image paths and labels, split into train/val/test."""
    all_samples = []
    for label in LABELS:
        final_dir = data_dir / label / "final"
        if not final_dir.exists():
            log.warning("Missing: %s — skipping", final_dir)
            continue
        paths = [p for p in final_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        log.info("  %s: %d images", label, len(paths))
        for p in paths:
            all_samples.append({"path": str(p), "label": label})

    random.seed(seed)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    train = all_samples[:n_train]
    val = all_samples[n_train: n_train + n_val]
    test = all_samples[n_train + n_val:]

    log.info("Split -> train: %d  val: %d  test: %d", len(train), len(val), len(test))
    return train, val, test


def generate_augmented_images(
    samples: list, aug_dir: Path, copies_per_image: int = 2, seed: int = 42
) -> list:
    """
    Pre-generate augmented copies of training images on disk.

    The CUDA version augments on-the-fly each epoch, so each image gets
    ~epochs different augmentations. We replicate this by generating
    `copies_per_image` augmented variants per original image, saved to disk
    so mlx-vlm can load them by path.

    All images (originals + augmented) are resized to TRAIN_IMAGE_SIZE to
    avoid the slow per-step resize inside mlx-vlm and to work around the
    --image-resize-shape bug in mlx-vlm 0.3.12.

    Returns the expanded sample list (resized originals + augmented copies).
    """
    aug_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    expanded = []
    generated = 0
    resized_originals = 0

    for sample in samples:
        src_path = Path(sample["path"])
        try:
            img = Image.open(src_path).convert("RGB")
        except (UnidentifiedImageError, Exception) as e:
            log.warning("Skipping augmentation for %s: %s", src_path.name, e)
            continue

        # Save resized original
        resized_name = f"{src_path.stem}_resized{src_path.suffix}"
        resized_path = aug_dir / sample["label"] / resized_name
        resized_path.parent.mkdir(parents=True, exist_ok=True)
        img.resize(TRAIN_IMAGE_SIZE, Image.BILINEAR).save(resized_path, quality=95)
        expanded.append({"path": str(resized_path), "label": sample["label"]})
        resized_originals += 1

        for i in range(copies_per_image):
            aug_img = augment_image(img, rng)
            aug_img = aug_img.resize(TRAIN_IMAGE_SIZE, Image.BILINEAR)
            aug_name = f"{src_path.stem}_aug{i}{src_path.suffix}"
            aug_path = aug_dir / sample["label"] / aug_name
            aug_path.parent.mkdir(parents=True, exist_ok=True)
            aug_img.save(aug_path, quality=95)
            expanded.append({"path": str(aug_path), "label": sample["label"]})
            generated += 1

    log.info(
        "Augmentation: %d resized originals + %d augmented = %d total training samples",
        resized_originals, generated, len(expanded),
    )
    return expanded


def prepare_jsonl(samples: list, output_path: Path):
    """
    Write samples to JSONL in the chat-message format expected by mlx-vlm.

    Gemma3 requires multimodal content format so processor.apply_chat_template
    inserts the <start_of_image> token where images should be placed.

    Each line: {"messages": [...], "images": ["<path>"]}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in samples:
            entry = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": USER_QUESTION},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": sample["label"]},
                        ],
                    },
                ],
                "images": [sample["path"]],
            }
            f.write(json.dumps(entry) + "\n")

    log.info("Wrote %d samples to %s", len(samples), output_path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    """Run LoRA fine-tuning via mlx-vlm."""
    import shutil
    import subprocess
    import sys

    data_dir = _PROJECT_ROOT / args.data_dir
    output_dir = _PROJECT_ROOT / args.output_dir
    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Final install location (where pipeline_mlx.py loads from)
    install_dir = _PROJECT_ROOT / "models" / "vga_skin_adapter_mlx"

    # Prepare dataset splits as JSONL
    log.info("Loading dataset from %s", data_dir)
    train_samples, val_samples, test_samples = load_split(data_dir)

    # Augment training images (replicates on-the-fly augmentation from CUDA version)
    if args.augment:
        aug_dir = output_dir / "augmented"
        train_samples = generate_augmented_images(
            train_samples, aug_dir,
            copies_per_image=args.aug_copies,
            seed=42,
        )

    jsonl_dir = output_dir / "data"
    prepare_jsonl(train_samples, jsonl_dir / "train.jsonl")
    prepare_jsonl(val_samples, jsonl_dir / "valid.jsonl")

    # Save test samples for later evaluation (outside jsonl_dir to avoid
    # HuggingFace datasets picking it up as a split with mismatched schema)
    test_path = output_dir / "test_samples.json"
    with open(test_path, "w") as f:
        json.dump(test_samples, f)
    log.info("Saved %d test samples to %s", len(test_samples), test_path)

    # Build mlx-vlm lora command
    # Matches CUDA finetune.py training config as closely as possible
    cmd = [
        sys.executable, "-m", "mlx_vlm.lora",
        "--model-path", MODEL_ID,
        "--dataset", str(jsonl_dir),
        "--output-path", str(adapter_dir / "adapters.safetensors"),
        # LoRA config (same as CUDA: rank=8, alpha=16, dropout=0.05)
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        # Training config
        "--learning-rate", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--print-every", "10",
        # Pass --apply-chat-template to DISABLE lora.py-level template processing.
        # Despite the name, action="store_false" means the flag disables it.
        # This is correct: the trainer's get_prompt() already applies the template
        # via processor.apply_chat_template(). Not passing this flag causes
        # double-application and errors like "0 image tokens but received 1 images".
        "--apply-chat-template",
    ]

    if args.max_steps > 0:
        cmd.extend(["--steps", str(args.max_steps)])
    else:
        cmd.extend(["--epochs", str(args.epochs)])

    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))

    if result.returncode != 0:
        log.error("Fine-tuning failed with exit code %d", result.returncode)
        return None, test_samples

    log.info("LoRA adapter saved to %s", adapter_dir)

    # Copy adapter to models/ so the backend pipeline picks it up automatically
    install_dir.mkdir(parents=True, exist_ok=True)
    for f in adapter_dir.iterdir():
        shutil.copy2(f, install_dir / f.name)
    log.info("Adapter installed to %s (used by backend pipeline)", install_dir)

    return str(adapter_dir), test_samples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(adapter_path: str, test_samples: list):
    """Evaluate the fine-tuned model on the test split."""
    from mlx_vlm import load, generate
    from mlx_vlm.utils import load_config

    log.info("Loading model with adapter from %s ...", adapter_path)
    model, processor = load(
        MODEL_ID,
        adapter_path=adapter_path,
        tokenizer_config={"trust_remote_code": True},
    )
    config = load_config(MODEL_ID)

    correct = 0
    total = 0
    per_class = {l: {"correct": 0, "total": 0} for l in LABELS}
    errors = []

    for sample in test_samples:
        label = sample["label"]
        img_path = sample["path"]

        try:
            Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, Exception):
            continue

        # Use processor.apply_chat_template to match training format
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
            [img_path], max_tokens=10, verbose=False,
        )
        raw = (result.text if hasattr(result, 'text') else str(result)).strip().lower()

        pred = "unknown"
        for l in LABELS:
            if l in raw:
                pred = l
                break

        is_correct = pred == label
        correct += int(is_correct)
        total += 1
        per_class[label]["total"] += 1
        per_class[label]["correct"] += int(is_correct)

        if not is_correct:
            errors.append((Path(img_path).name, label, pred, raw))

    acc = correct / max(total, 1) * 100
    log.info("=" * 55)
    log.info("Test accuracy: %.1f%%  (%d/%d)", acc, correct, total)
    for l in LABELS:
        c = per_class[l]
        pct = c["correct"] / max(c["total"], 1) * 100
        log.info("  %-12s  %.1f%%  (%d/%d)", l, pct, c["correct"], c["total"])
    log.info("=" * 55)

    if errors:
        log.info("Misclassified (%d):", len(errors))
        for name, true, pred, raw in errors[:20]:
            log.info("  %-40s  true=%-12s pred=%s raw='%s'", name, true, pred, raw)

    return acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="VGA — LoRA fine-tuning on Apple Silicon (MLX)"
    )
    p.add_argument("--data-dir", default="data/vga")
    p.add_argument("--output-dir", default="runs/vga_mlx")
    # Training — defaults match CUDA finetune.py where supported
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Max training steps (-1 = full run, use 10 for smoke test)")
    # LoRA — defaults match CUDA finetune.py
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05,
                   help="LoRA dropout (CUDA default: 0.05)")
    # Augmentation — replicates CUDA on-the-fly augmentation
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="Skip data augmentation")
    p.add_argument("--aug-copies", type=int, default=2,
                   help="Augmented copies per image (default: 2, ~3x dataset)")
    p.set_defaults(augment=True)
    # Eval
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--adapter-path", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    import platform
    log.info("Platform: %s %s", platform.machine(), platform.platform())

    if args.eval_only:
        adapter = args.adapter_path or str(
            _PROJECT_ROOT / "runs" / "vga_mlx" / "lora_adapter"
        )
        if not Path(adapter).exists():
            log.error("Adapter not found: %s", adapter)
            log.error("Run finetune_mlx.py first, or pass --adapter-path <path>")
            return

        # Load test samples
        test_path = _PROJECT_ROOT / args.output_dir / "test_samples.json"
        if test_path.exists():
            with open(test_path) as f:
                test_samples = json.load(f)
        else:
            _, _, test_samples = load_split(_PROJECT_ROOT / args.data_dir)

        evaluate(adapter, test_samples)
    else:
        adapter_path, test_samples = train(args)
        if adapter_path:
            log.info("Running evaluation on test split...")
            evaluate(adapter_path, test_samples)


if __name__ == "__main__":
    main()
