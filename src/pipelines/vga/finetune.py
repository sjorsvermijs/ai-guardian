"""
VGA — QLoRA Finetuning on Baby Skin Images
============================================
Trains a LoRA adapter on top of the 4-bit quantized MedGemma model to classify
baby skin images as:  healthy | eczema | chickenpox

Designed for an NVIDIA RTX 3070 (8 GB VRAM).

Usage (from project root)
-------------------------
    python -m src.pipelines.vga.finetune
    python -m src.pipelines.vga.finetune --epochs 5 --lora-rank 16
    python -m src.pipelines.vga.finetune --max-steps 10   # smoke test
"""

import argparse
import logging
import os
import random
from pathlib import Path

import torch
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
MODEL_ID = "google/medgemma-1.5-4b-it"

LABELS = ["healthy", "eczema", "chickenpox"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

USER_QUESTION = (
    "Look at this photo of a baby's skin. "
    "Classify the skin condition as exactly one of: healthy, eczema, chickenpox. "
    "Respond with only that single word."
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_image(img: Image.Image, rng: random.Random) -> Image.Image:
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
# Dataset
# ---------------------------------------------------------------------------

def load_split(data_dir: Path, split_ratios=(0.8, 0.1, 0.1), seed=42):
    all_samples = []
    for label in LABELS:
        final_dir = data_dir / label / "final"
        if not final_dir.exists():
            log.warning("Missing: %s — skipping", final_dir)
            continue
        paths = [p for p in final_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
        log.info("  %s: %d images", label, len(paths))
        for p in paths:
            all_samples.append({"path": p, "label": label, "label_id": LABEL2ID[label]})

    random.seed(seed)
    random.shuffle(all_samples)

    n = len(all_samples)
    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    train = all_samples[:n_train]
    val = all_samples[n_train : n_train + n_val]
    test = all_samples[n_train + n_val :]

    log.info("Split -> train: %d  val: %d  test: %d", len(train), len(val), len(test))
    return train, val, test


class SkinDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list, processor, max_length: int = 128, augment: bool = False):
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
        self._rng = random.Random()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample["label"]

        try:
            image = Image.open(sample["path"]).convert("RGB")
        except (UnidentifiedImageError, Exception):
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.augment:
            image = augment_image(image, self._rng)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_QUESTION},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        inputs = self.processor(
            text=text, images=image, return_tensors="pt",
            padding="max_length", truncation=True, max_length=self.max_length,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs["pixel_values"].squeeze(0)
        token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids)).squeeze(0)

        labels = input_ids.clone()
        try:
            model_token_id = self.processor.tokenizer.encode("model\n", add_special_tokens=False)
            seq = input_ids.tolist()
            answer_start = len(seq)
            for i in range(len(seq) - len(model_token_id), -1, -1):
                if seq[i : i + len(model_token_id)] == model_token_id:
                    answer_start = i + len(model_token_id)
                    break
            labels[:answer_start] = -100
        except Exception:
            labels[:int(len(input_ids) * 0.9)] = -100

        labels[input_ids == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "token_type_ids": torch.stack([x["token_type_ids"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def load_model_and_processor(lora_rank: int, lora_alpha: int):
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    log.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)

    log.info("Loading model in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="cuda",
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    )

    vram_gb = torch.cuda.memory_allocated() / 1e9
    log.info("Model loaded. VRAM used: %.2f GB", vram_gb)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    from transformers import TrainingArguments, Trainer

    data_dir = _PROJECT_ROOT / args.data_dir
    output_dir = _PROJECT_ROOT / args.output_dir

    log.info("Loading dataset from %s", data_dir)
    train_samples, val_samples, test_samples = load_split(data_dir)

    model, processor = load_model_and_processor(args.lora_rank, args.lora_alpha)

    train_ds = SkinDataset(train_samples, processor, max_length=args.max_length, augment=True)
    val_ds = SkinDataset(val_samples, processor, max_length=args.max_length, augment=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    log.info("Starting training...")
    trainer.train()

    adapter_dir = output_dir / "lora_adapter"
    log.info("Saving LoRA adapter to %s", adapter_dir)
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    log.info("Done. LoRA adapter saved.")
    return trainer, test_samples, processor, model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, processor, test_samples, max_length, device="cuda"):
    from transformers import GenerationConfig
    model.eval()

    correct = 0
    total = 0
    per_class = {l: {"correct": 0, "total": 0} for l in LABELS}

    gen_config = GenerationConfig(
        max_new_tokens=5,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )

    for sample in test_samples:
        label = sample["label"]
        try:
            image = Image.open(sample["path"]).convert("RGB")
        except Exception:
            continue

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
            output_ids = model.generate(**inputs, generation_config=gen_config)

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        pred_text = processor.decode(new_tokens, skip_special_tokens=True).strip().lower()

        pred = None
        for l in LABELS:
            if l in pred_text:
                pred = l
                break

        correct += int(pred == label)
        total += 1
        per_class[label]["total"] += 1
        per_class[label]["correct"] += int(pred == label)

    acc = correct / max(total, 1) * 100
    log.info("=" * 50)
    log.info("Test accuracy: %.1f%% (%d / %d)", acc, correct, total)
    for l in LABELS:
        c = per_class[l]
        pct = c["correct"] / max(c["total"], 1) * 100
        log.info("  %-12s  %.1f%%  (%d/%d)", l, pct, c["correct"], c["total"])
    log.info("=" * 50)
    return acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VGA — Finetune MedGemma on baby skin images")
    p.add_argument("--data-dir", default="data/vga")
    p.add_argument("--output-dir", default="runs/vga")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Max training steps (-1 = full run, use 10 for smoke test)")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--adapter-path", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    log.info("Device: %s", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    log.info("VRAM total: %.1f GB", torch.cuda.get_device_properties(0).total_memory / 1e9)

    if args.eval_only:
        from peft import PeftModel
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        adapter = args.adapter_path or str(_PROJECT_ROOT / "models" / "vga_skin_adapter")
        log.info("Eval-only mode. Loading adapter from %s", adapter)
        processor = AutoProcessor.from_pretrained(adapter, token=HF_TOKEN)
        base = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID, device_map="cuda", torch_dtype=torch.bfloat16, token=HF_TOKEN
        )
        model = PeftModel.from_pretrained(base, adapter)
        _, _, test_samples = load_split(_PROJECT_ROOT / args.data_dir)
        evaluate(model, processor, test_samples, args.max_length)
    else:
        trainer, test_samples, processor, model = train(args)
        log.info("Running evaluation on test split...")
        evaluate(model, processor, test_samples, args.max_length)


if __name__ == "__main__":
    main()
