"""
VGA — Standalone Inference & Evaluation
========================================
Load the fine-tuned LoRA adapter and classify baby skin images.

Usage (from project root)
-------------------------
    python -m src.pipelines.vga.inference --image path/to/image.jpg
    python -m src.pipelines.vga.inference --dir path/to/folder/
    python -m src.pipelines.vga.inference --eval
"""

import argparse
import logging
import os
from pathlib import Path

import torch
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

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = "google/medgemma-1.5-4b-it"

LABELS = ["healthy", "eczema", "chickenpox"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

USER_QUESTION = (
    "Look at this photo of a baby's skin. "
    "Classify the skin condition as exactly one of: healthy, eczema, chickenpox. "
    "Respond with only that single word."
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(adapter_path: str, device: str = "cuda"):
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration
    from peft import PeftModel

    log.info("Loading processor from %s ...", adapter_path)
    processor = AutoProcessor.from_pretrained(adapter_path, token=HF_TOKEN)

    log.info("Loading base model in 4-bit ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.bfloat16,
        token=HF_TOKEN,
    )

    log.info("Loading LoRA adapter from %s ...", adapter_path)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    vram_gb = torch.cuda.memory_allocated() / 1e9
    log.info("Model ready. VRAM used: %.2f GB", vram_gb)
    return model, processor


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(model, processor, image: Image.Image, device: str = "cuda") -> tuple[str, str]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_QUESTION},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    raw = processor.decode(new_tokens, skip_special_tokens=True).strip().lower()

    pred = "unknown"
    for label in LABELS:
        if label in raw:
            pred = label
            break

    return pred, raw


def predict_dir(model, processor, img_dir: Path, device: str = "cuda"):
    paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if not paths:
        log.warning("No images found in %s", img_dir)
        return

    log.info("Classifying %d images in %s ...", len(paths), img_dir)
    results = {}
    for path in paths:
        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, Exception) as e:
            log.warning("  Skipping %s: %s", path.name, e)
            continue

        pred, raw = predict(model, processor, image, device)
        results[path.name] = pred
        print(f"  {path.name:<40}  -> {pred}  (raw: '{raw}')")

    from collections import Counter
    counts = Counter(results.values())
    print("\nSummary:")
    for label in LABELS + ["unknown"]:
        if counts[label]:
            print(f"  {label}: {counts[label]}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, processor, data_dir: Path, device: str = "cuda"):
    import random

    all_samples = []
    for label in LABELS:
        final_dir = data_dir / label / "final"
        if not final_dir.exists():
            log.warning("Missing: %s", final_dir)
            continue
        paths = [p for p in final_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        for p in paths:
            all_samples.append({"path": p, "label": label})

    random.seed(42)
    random.shuffle(all_samples)
    n = len(all_samples)
    test_samples = all_samples[int(n * 0.8) + int(n * 0.1):]

    log.info("Evaluating on %d test samples ...", len(test_samples))

    correct = 0
    total = 0
    per_class = {l: {"correct": 0, "total": 0} for l in LABELS}
    errors = []

    for sample in test_samples:
        label = sample["label"]
        try:
            image = Image.open(sample["path"]).convert("RGB")
        except Exception:
            continue

        pred, raw = predict(model, processor, image, device)
        is_correct = pred == label
        correct += int(is_correct)
        total += 1
        per_class[label]["total"] += 1
        per_class[label]["correct"] += int(is_correct)

        if not is_correct:
            errors.append((sample["path"].name, label, pred, raw))

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
    p = argparse.ArgumentParser(description="VGA — Inference with fine-tuned MedGemma")
    p.add_argument("--adapter", default=str(_PROJECT_ROOT / "models" / "vga_skin_adapter"))
    p.add_argument("--image", default=None)
    p.add_argument("--dir", default=None)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--data-dir", default="data/vga")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    adapter_path = args.adapter
    if not Path(adapter_path).exists():
        log.error("Adapter not found: %s", adapter_path)
        log.error("Run finetune first, or pass --adapter <path>")
        return

    model, processor = load_model(adapter_path, device)

    if args.image:
        path = Path(args.image)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            log.error("Cannot open image %s: %s", path, e)
            return
        pred, raw = predict(model, processor, image, device)
        print(f"\nPrediction: {pred.upper()}")
        print(f"Raw output: '{raw}'")

    elif args.dir:
        predict_dir(model, processor, Path(args.dir), device)

    elif args.eval:
        evaluate(model, processor, _PROJECT_ROOT / args.data_dir, device)

    else:
        print("Error: specify --image, --dir, or --eval")


if __name__ == "__main__":
    main()
