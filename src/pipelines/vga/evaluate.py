"""
Evaluate the VGA skin classification model.

Usage:
  python -m src.pipelines.vga.evaluate info
  python -m src.pipelines.vga.evaluate classify <image.jpg>
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "vga_skin_adapter"
RESULTS_FILE = Path(__file__).parent / "evaluation_results.json"


def show_info():
    """Show saved model performance metrics."""
    meta_path = RESULTS_FILE
    if not meta_path.exists():
        print(f"Metadata not found: {meta_path}")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    print("=" * 60)
    print("VGA Skin Condition Classifier — Evaluation Results")
    print("=" * 60)
    print(f"Base model:       {meta['model']}")
    print(f"Adapter:          {meta['adapter']}")
    print(f"Method:           {meta['method']}")
    print(f"Trainable params: {meta['trainable_params']}")

    ds = meta["dataset"]
    print(f"\nDataset:          {ds['total_images']} images ({ds['split']})")
    print(f"Classes:          {', '.join(ds['classes'])}")
    print(f"Test set:         {ds['test']} samples")

    ev = meta["evaluation"]
    print(f"\n{'— Test Results —':^60}")
    print(f"  Accuracy:       {ev['accuracy']:.1%}")
    print(f"  F1 (macro):     {ev['f1_macro']:.3f}")
    print(f"  F1 (weighted):  {ev['f1_weighted']:.3f}")

    print(f"\n  {'Class':<12} {'Precision':>9} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*48}")
    for cls, m in ev["per_class"].items():
        print(f"  {cls:<12} {m['precision']:>9.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {m['support']:>9}")

    print(f"\n  Misclassified:  {ev['misclassified']} ({ev['confusion_notes']})")
    print("=" * 60)


def classify_image(image_path: str):
    """Classify a single image using the VGA pipeline."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    from src.pipelines.vga.pipeline import VGAPipeline
    from src.core.config import Config

    image_path = Path(image_path)
    if not image_path.exists():
        print(f"File not found: {image_path}")
        sys.exit(1)

    config = Config()
    pipeline = VGAPipeline(config.vga_config)
    print("Loading model...")
    pipeline.initialize()

    from PIL import Image
    img = Image.open(image_path).convert("RGB")

    pred, confidence = pipeline._predict_single(img)
    print(f"\nImage:      {image_path.name}")
    print(f"Prediction: {pred}")
    print(f"Confidence: {confidence:.1%}")

    pipeline.cleanup()
    return pred, confidence


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.pipelines.vga.evaluate info")
        print("  python -m src.pipelines.vga.evaluate classify <image.jpg>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "info":
        show_info()
    elif command == "classify":
        if len(sys.argv) < 3:
            print("Please provide an image file path.")
            sys.exit(1)
        classify_image(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
