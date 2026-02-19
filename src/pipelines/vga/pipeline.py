"""
VGA (Visual Grading Assessment) Pipeline — Skin Condition Classifier

Analyzes video screenshots using a fine-tuned MedGemma LoRA adapter to classify
infant skin conditions as: healthy | eczema | chickenpox

The adapter is loaded from HuggingFace (private repo, requires HF_TOKEN) on top
of the 4-bit quantized MedGemma 1.5 4B base model.
"""

import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv

from src.core.base_pipeline import BasePipeline, PipelineResult

# Load environment variables from .env file in project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

LABELS = ["healthy", "eczema", "chickenpox"]

USER_QUESTION = (
    "Look at this photo of a baby's skin. "
    "Classify the skin condition as exactly one of: healthy, eczema, chickenpox. "
    "Respond with only that single word."
)


class VGAPipeline(BasePipeline):
    """
    VGA Pipeline — classifies infant skin conditions from video screenshots.

    Uses a QLoRA fine-tuned MedGemma 1.5 4B model (4-bit NF4 quantization).
    Processes 3 evenly-spaced screenshots, aggregates via majority vote.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.processor = None
        self.screenshot_count = config.get('screenshot_count', 3)
        self.base_model_id = config.get('base_model', 'google/medgemma-1.5-4b-it')
        self.adapter_repo = config.get('adapter_repo', 'dinopasic/medgemma-skin-v2')
        self.adapter_path: Optional[Path] = None
        self.hf_token = config.get('hf_token', '') or os.environ.get('HF_TOKEN', '')
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.labels = config.get('labels', LABELS)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def initialize(self) -> bool:
        """
        Download the LoRA adapter from HuggingFace (if not cached),
        then load the 4-bit base model + adapter.
        """
        try:
            logger.info("Initializing VGA Pipeline (MedGemma skin classifier)...")

            # Download adapter from HuggingFace if not already cached
            model_path = Path(self.config.get('model_path', 'models/vga_skin_adapter'))
            if not (model_path / 'adapter_model.safetensors').exists():
                logger.info("Downloading adapter from %s ...", self.adapter_repo)
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id=self.adapter_repo,
                    local_dir=str(model_path),
                    token=self.hf_token or None,
                )
            self.adapter_path = model_path

            # Load processor
            from transformers import AutoProcessor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(
                str(self.adapter_path), token=self.hf_token or None
            )

            # Load base model in 4-bit
            from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration
            logger.info("Loading base model in 4-bit NF4...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = Gemma3ForConditionalGeneration.from_pretrained(
                self.base_model_id,
                quantization_config=bnb_config,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                token=self.hf_token or None,
            )

            # Apply LoRA adapter
            from peft import PeftModel
            logger.info("Loading LoRA adapter from %s ...", self.adapter_path)
            self.model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
            self.model.eval()

            vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            logger.info("VGA Pipeline ready. VRAM: %.2f GB", vram_gb)
            self.is_initialized = True
            return True

        except Exception as e:
            logger.error("Failed to initialize VGA Pipeline: %s", e, exc_info=True)
            self.is_initialized = False
            return False

    def _predict_single(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on a single PIL image.

        Returns dict with keys: prediction, confidence, raw
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_QUESTION},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=text, images=image, return_tensors="pt"
        ).to(self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        raw = self.processor.decode(new_tokens, skip_special_tokens=True).strip().lower()

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
            raise RuntimeError("VGA Pipeline not initialized. Call initialize() first.")

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
                metadata={"labels": self.labels}
            )
        except Exception as e:
            logger.error("VGA process error: %s", e, exc_info=True)
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
            raise RuntimeError("VGA Pipeline not initialized. Call initialize() first.")

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
                    "model": self.adapter_repo,
                }
            )

        except Exception as e:
            logger.error("VGA batch error: %s", e, exc_info=True)
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_initialized = False
        logger.info("VGA Pipeline cleaned up")
