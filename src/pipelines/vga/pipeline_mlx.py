"""
VGA (Visual Grading Assessment) Pipeline — MLX-Optimized Skin Condition Classifier

Apple Silicon version using mlx-vlm for native Metal acceleration.
Analyzes video screenshots using a fine-tuned LoRA adapter on MedGemma 1.5 4B
to classify infant skin conditions as: healthy | eczema | chickenpox

The PyTorch version (pipeline.py) is kept for CUDA users.
"""

import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from dotenv import load_dotenv

from src.core.base_pipeline import BasePipeline, PipelineResult

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

LABELS = ["healthy", "eczema", "chickenpox"]

USER_QUESTION = (
    "This is a photo of a baby's skin. "
    "Classify it as exactly one of: healthy, eczema, chickenpox. "
    "Reply with ONLY one word: healthy, eczema, or chickenpox."
)


class VGAPipelineMLX(BasePipeline):
    """
    VGA Pipeline (MLX) — classifies infant skin conditions from video screenshots.

    Uses mlx-vlm with a LoRA adapter on MedGemma 1.5 4B for native Apple Silicon
    inference. Processes 3 evenly-spaced screenshots, aggregates via majority vote.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.processor = None
        self.model_config = None
        self.screenshot_count = config.get('screenshot_count', 3)
        self.base_model_id = config.get(
            'base_model_mlx', 'mlx-community/medgemma-1.5-4b-it-4bit'
        )
        self.adapter_path: Optional[Path] = None
        self.hf_token = config.get('hf_token', '') or os.environ.get('HF_TOKEN', '')
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.labels = config.get('labels', LABELS)

    def initialize(self) -> bool:
        """
        Load the MedGemma 1.5 4B model via mlx-vlm and optionally apply
        a LoRA adapter trained with finetune_mlx.py.
        """
        try:
            logger.info("Initializing VGA Pipeline (MLX / Apple Silicon)...")

            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            # Check for MLX LoRA adapter
            adapter_dir = Path(
                self.config.get('model_path_mlx', 'models/vga_skin_adapter_mlx')
            )
            adapter_file = adapter_dir / "adapters.safetensors"

            if adapter_file.exists():
                logger.info("Loading base model + MLX LoRA adapter from %s", adapter_dir)
                self.model, self.processor = load(
                    self.base_model_id,
                    adapter_path=str(adapter_dir),
                    tokenizer_config={"trust_remote_code": True},
                )
            else:
                logger.info(
                    "No MLX adapter found at %s — loading base model only. "
                    "Run finetune_mlx.py to train an adapter.",
                    adapter_dir,
                )
                self.model, self.processor = load(
                    self.base_model_id,
                    tokenizer_config={"trust_remote_code": True},
                )

            self.model_config = load_config(self.base_model_id)
            self.adapter_path = adapter_dir

            logger.info("VGA Pipeline (MLX) ready — model: %s", self.base_model_id)
            self.is_initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize VGA MLX Pipeline: %s", e, exc_info=True)
            self.is_initialized = False
            return False

    def _predict_single(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a single PIL image via mlx-vlm.

        Returns dict with keys: prediction, confidence, raw
        """
        from mlx_vlm import generate

        # Resize to 224x224 to reduce GPU memory usage
        if image.width > 224 or image.height > 224:
            image = image.resize((224, 224), Image.BILINEAR)

        # Save image to a temporary file (mlx-vlm expects file paths)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            image.save(tmp, format="JPEG", quality=95)
            tmp_path = tmp.name

        try:
            # Use processor.apply_chat_template to match training format
            # (image token BEFORE text, matching how mlx-vlm's trainer formats prompts)
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_QUESTION},
                ]},
            ]
            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            result = generate(
                self.model,
                self.processor,
                formatted_prompt,
                [tmp_path],
                max_tokens=10,
                verbose=False,
            )
        finally:
            os.unlink(tmp_path)

        # mlx-vlm may return a GenerationResult object or a string
        raw = (result.text if hasattr(result, 'text') else str(result)).strip().lower()

        # Match to known labels
        pred = "unknown"
        for label in self.labels:
            if label in raw:
                pred = label
                break

        # Confidence: 1.0 for a clean single-word match, 0.7 otherwise
        confidence = 1.0 if raw in self.labels else 0.7

        return {"prediction": pred, "confidence": confidence, "raw": raw}

    def process(self, input_data: Any) -> PipelineResult:
        """Process a single image (numpy array H,W,C)."""
        if not self.is_initialized:
            raise RuntimeError("VGA MLX Pipeline not initialized. Call initialize() first.")

        if not self.validate_input(input_data):
            return PipelineResult(
                pipeline_name="VGA",
                timestamp=datetime.now(),
                confidence=0.0,
                data={},
                warnings=[],
                errors=["Invalid input format. Expected numpy array image (H, W, C)"],
                metadata={}
            )

        try:
            image = Image.fromarray(np.uint8(input_data)).convert("RGB")
            result = self._predict_single(image)

            return PipelineResult(
                pipeline_name="VGA",
                timestamp=datetime.now(),
                confidence=result["confidence"],
                data={
                    "skin_assessment": {
                        "classification": result["prediction"],
                        "confidence": result["confidence"],
                        "severity": "none" if result["prediction"] == "healthy" else "detected",
                        "overall_status": "normal" if result["prediction"] == "healthy" else "concerning",
                    },
                    "raw_output": result["raw"],
                },
                warnings=[],
                errors=[],
                metadata={"labels": self.labels, "backend": "mlx"}
            )
        except Exception as e:
            logger.error("VGA MLX process error: %s", e, exc_info=True)
            return PipelineResult(
                pipeline_name="VGA",
                timestamp=datetime.now(),
                confidence=0.0,
                data={},
                warnings=[],
                errors=[f"Error processing image: {str(e)}"],
                metadata={}
            )

    def process_batch(self, images: List[np.ndarray]) -> PipelineResult:
        """
        Process multiple screenshots and aggregate results.

        Classifies each screenshot independently, then uses majority vote
        for the final classification and averages confidence scores.
        """
        if not self.is_initialized:
            raise RuntimeError("VGA MLX Pipeline not initialized. Call initialize() first.")

        if not images or len(images) == 0:
            return PipelineResult(
                pipeline_name="VGA_Batch",
                timestamp=datetime.now(),
                confidence=0.0,
                data={},
                warnings=[],
                errors=["No images provided for batch processing"],
                metadata={}
            )
        
        logger.info("Processing batch of %d images through VGA MLX Pipeline...", len(images))

        try:
            per_screenshot = []
            predictions = []
            confidences = []

            for i, img_array in enumerate(images):
                image = Image.fromarray(np.uint8(img_array)).convert("RGB")
                result = self._predict_single(image)
                per_screenshot.append({
                    "index": i,
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "raw": result["raw"],
                })
                if result["prediction"] != "unknown":
                    predictions.append(result["prediction"])
                    confidences.append(result["confidence"])

            # Majority vote
            if predictions:
                vote_counts = Counter(predictions)
                classification = vote_counts.most_common(1)[0][0]
                avg_confidence = sum(confidences) / len(confidences)
            else:
                classification = "unknown"
                avg_confidence = 0.0

            is_healthy = classification == "healthy"

            return PipelineResult(
                pipeline_name="VGA_Batch",
                timestamp=datetime.now(),
                confidence=avg_confidence,
                data={
                    "skin_assessment": {
                        "classification": classification,
                        "confidence": avg_confidence,
                        "severity": "none" if is_healthy else "detected",
                        "overall_status": "normal" if is_healthy else "concerning",
                    },
                    "per_screenshot": per_screenshot,
                    "labels": self.labels,
                    "num_screenshots_analyzed": len(images),
                },
                warnings=[],
                errors=[],
                metadata={
                    "batch_size": len(images),
                    "vote_counts": dict(Counter(predictions)) if predictions else {},
                    "model": self.base_model_id,
                    "backend": "mlx",
                }
            )

        except Exception as e:
            logger.error("VGA MLX batch error: %s", e, exc_info=True)
            return PipelineResult(
                pipeline_name="VGA_Batch",
                timestamp=datetime.now(),
                confidence=0.0,
                data={},
                warnings=[],
                errors=[f"Error in batch processing: {str(e)}"],
                metadata={}
            )

    def validate_input(self, input_data: Any) -> bool:
        """Validate input image format."""
        if not isinstance(input_data, np.ndarray):
            return False
        if len(input_data.shape) not in [2, 3]:
            return False
        if len(input_data.shape) == 3 and input_data.shape[2] not in [1, 3, 4]:
            return False
        return True

    def cleanup(self) -> None:
        """Release model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.is_initialized = False
        logger.info("VGA MLX Pipeline cleaned up")
