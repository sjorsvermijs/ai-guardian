"""
Cry Classification Pipeline — supports multiple audio embedding models.

Supported backends:
  - "ast"    : Audio Spectrogram Transformer (MIT/ast-finetuned-audioset)  768-dim
  - "hubert" : HuBERT Base (facebook/hubert-base-ls960)                   768-dim

Usage:
  pipeline = CryPipeline({"model_backend": "hubert"})
  pipeline.initialize()
  result = pipeline.process(audio_data)
"""

from typing import Any, Dict
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import pickle

from ...core.base_pipeline import BasePipeline, PipelineResult

DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "models"
SAMPLE_RATE = 16000

BACKENDS = {
    "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "hubert": "facebook/hubert-base-ls960",
}

CRY_LABELS = ["belly", "burping", "cold", "discomfort", "hungry", "tired"]


class CryPipeline(BasePipeline):
    """
    Baby Cry Classification Pipeline.
    Extracts embeddings from audio for downstream cry-reason classification.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.processor = None
        self.classifier = None
        self.device = None
        self.sample_rate = config.get("sample_rate", SAMPLE_RATE)
        self.backend = config.get("model_backend", "ast")
        self.model_name = config.get(
            "model_name", BACKENDS.get(self.backend, BACKENDS["ast"])
        )
        self.cache_dir = Path(config.get("cache_dir", DEFAULT_CACHE_DIR))

    def initialize(self) -> bool:
        """Load the selected model from HuggingFace."""
        try:
            # Check for MPS (Apple Silicon), CUDA (NVIDIA), then fallback to CPU
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            self.cache_dir.mkdir(parents=True, exist_ok=True)

            if self.backend == "hubert":
                self._init_hubert()
            else:
                self._init_ast()

            self.model.to(self.device)
            self.model.eval()

            # Load cry classifier
            classifier_path = self.cache_dir / "cry_classifier" / "svm_ast.pkl"
            if classifier_path.exists():
                with open(classifier_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                print(f"✓ Loaded cry classifier from {classifier_path}")
            else:
                print(f"⚠️ Cry classifier not found at {classifier_path}, will return embeddings only")

            self.is_initialized = True
            print(
                f"Cry Pipeline ({self.backend}) initialized on {self.device}",
                flush=True,
            )
            return True
        except Exception as e:
            print(f"Cry Pipeline initialization failed: {e}", flush=True)
            return False

    def _init_ast(self):
        from transformers import ASTModel, ASTFeatureExtractor

        self.processor = ASTFeatureExtractor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = ASTModel.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )

    def _init_hubert(self):
        from transformers import HubertModel, Wav2Vec2FeatureExtractor

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        self.model = HubertModel.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )

    def process(self, audio_data: np.ndarray) -> PipelineResult:
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        if not self.validate_input(audio_data):
            return PipelineResult(
                pipeline_name="Cry",
                timestamp=datetime.now(),
                confidence=0.0,
                data={},
                warnings=[],
                errors=["Invalid audio input"],
                metadata={},
            )

        try:
            if self.sample_rate != SAMPLE_RATE:
                from scipy import signal

                new_length = int(
                    len(audio_data) * SAMPLE_RATE / self.sample_rate
                )
                audio_data = signal.resample(audio_data, new_length)

            embedding = self._extract_embedding(audio_data)

            # If classifier is loaded, make predictions
            result_data = {
                "embedding_dim": len(embedding),
            }

            if self.classifier is not None:
                try:
                    # Predict probabilities for each class
                    probabilities = self.classifier.predict_proba(embedding.reshape(1, -1))[0]
                    predicted_class_idx = np.argmax(probabilities)
                    predicted_class = CRY_LABELS[predicted_class_idx]
                    confidence = float(probabilities[predicted_class_idx])

                    result_data.update({
                        "prediction": predicted_class,
                        "confidence": confidence,
                        "probabilities": {
                            label: float(prob)
                            for label, prob in zip(CRY_LABELS, probabilities)
                        }
                    })
                except Exception as e:
                    print(f"⚠️ Classifier prediction failed: {e}")
                    result_data["embedding"] = embedding  # Fallback to embedding
            else:
                result_data["embedding"] = embedding  # No classifier, return embedding

            return PipelineResult(
                pipeline_name="Cry",
                timestamp=datetime.now(),
                confidence=result_data.get("confidence", 0.95),
                data=result_data,
                warnings=[],
                errors=[],
                metadata={
                    "sample_rate": SAMPLE_RATE,
                    "duration_seconds": len(audio_data) / SAMPLE_RATE,
                    "model": self.model_name,
                    "backend": self.backend,
                    "classifier_available": self.classifier is not None,
                },
            )
        except Exception as e:
            return PipelineResult(
                pipeline_name="Cry",
                timestamp=datetime.now(),
                confidence=0.0,
                data={},
                warnings=[],
                errors=[str(e)],
                metadata={},
            )

    def _extract_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        if self.backend == "hubert":
            return self._extract_hubert(audio_data)
        return self._extract_ast(audio_data)

    def _extract_ast(self, audio_data: np.ndarray) -> np.ndarray:
        inputs = self.processor(
            audio_data, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            # CLS token (first token of last hidden state)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return embedding

    def _extract_hubert(self, audio_data: np.ndarray) -> np.ndarray:
        inputs = self.processor(
            audio_data,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)
            # Mean-pool over time dimension for a fixed-size embedding
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
        return embedding

    def validate_input(self, audio_data: Any) -> bool:
        if not isinstance(audio_data, np.ndarray):
            return False
        if len(audio_data.shape) > 2:
            return False
        return True

    def cleanup(self) -> None:
        self.model = None
        self.processor = None
        self.is_initialized = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("Cry Pipeline cleaned up", flush=True)
