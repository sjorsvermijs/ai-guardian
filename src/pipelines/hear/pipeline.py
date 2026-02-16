"""
HeAR Pipeline Implementation
Processes audio data using Google's Health Acoustic Representations model.
https://huggingface.co/google/hear-pytorch
"""

from typing import Any, Dict, List
from datetime import datetime
import os
from pathlib import Path
import numpy as np
import torch

from ...core.base_pipeline import BasePipeline, PipelineResult

# Default model cache directory (project_root/models)
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "models"
from .audio_utils import (
    preprocess_audio,
    resample_audio_and_convert_to_mono,
    SAMPLE_RATE,
    CLIP_LENGTH,
)


class HeARPipeline(BasePipeline):
    """
    Health Acoustic Representations (HeAR) Pipeline.
    Generates embeddings from audio for health acoustic analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.device = None
        self.sample_rate = config.get("sample_rate", SAMPLE_RATE)
        self.model_name = config.get("model_name", "google/hear-pytorch")
        self.cache_dir = Path(config.get("cache_dir", DEFAULT_CACHE_DIR))

    def initialize(self) -> bool:
        """
        Load HeAR model from HuggingFace.

        Note: The model is gated. You must:
        1. Create a HuggingFace account at https://huggingface.co
        2. Accept terms at https://huggingface.co/google/hear-pytorch
        3. Run: huggingface-cli login
        """
        try:
            from transformers import AutoModel

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            token = os.environ.get("HF_TOKEN")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                token=token,
                cache_dir=self.cache_dir,
            )
            self.model.to(self.device)
            self.model.eval()

            self.is_initialized = True
            print(f"HeAR Pipeline initialized on {self.device}")
            return True
        except Exception as e:
            error_msg = str(e)
            if "gated repo" in error_msg.lower() or "401" in error_msg:
                print("HeAR model access denied. To fix this:")
                print("  1. Go to https://huggingface.co/google/hear-pytorch")
                print("  2. Accept the terms of use")
                print("  3. Run: huggingface-cli login")
            else:
                print(f"HeAR Pipeline initialization failed: {e}")
            return False

    def process(self, audio_data: np.ndarray) -> PipelineResult:
        """
        Process audio data to extract HeAR embeddings.

        Args:
            audio_data: Audio samples as numpy array.

        Returns:
            PipelineResult with embeddings and metadata.
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        if not self.validate_input(audio_data):
            return PipelineResult(
                pipeline_name="HeAR",
                timestamp=datetime.now(),
                confidence=0.0,
                data={},
                warnings=[],
                errors=["Invalid audio input"],
                metadata={},
            )

        try:
            # Resample if needed
            if self.sample_rate != SAMPLE_RATE:
                audio_data = resample_audio_and_convert_to_mono(
                    audio_data, self.sample_rate, SAMPLE_RATE
                )

            # Process in 2-second chunks
            embeddings = self._extract_embeddings(audio_data)

            return PipelineResult(
                pipeline_name="HeAR",
                timestamp=datetime.now(),
                confidence=0.95,
                data={
                    "embeddings": embeddings,
                    "embedding_dim": 512,
                    "num_chunks": len(embeddings),
                },
                warnings=[],
                errors=[],
                metadata={
                    "sample_rate": SAMPLE_RATE,
                    "duration_seconds": len(audio_data) / SAMPLE_RATE,
                    "model": self.model_name,
                },
            )
        except Exception as e:
            return PipelineResult(
                pipeline_name="HeAR",
                timestamp=datetime.now(),
                confidence=0.0,
                data={},
                warnings=[],
                errors=[str(e)],
                metadata={},
            )

    def _extract_embeddings(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """Extract embeddings from audio in 2-second chunks."""
        embeddings = []

        # Split into 2-second chunks (32000 samples at 16kHz)
        for start in range(0, len(audio_data), CLIP_LENGTH):
            chunk = audio_data[start : start + CLIP_LENGTH]

            # Prepare tensor (will be padded if < 2s)
            audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
            spectrogram = preprocess_audio(audio_tensor).to(self.device)

            # Get embedding
            with torch.no_grad():
                output = self.model(spectrogram, return_dict=True, output_hidden_states=True)
                embedding = output.pooler_output.cpu().numpy().flatten()
                embeddings.append(embedding)

        return embeddings

    def validate_input(self, audio_data: Any) -> bool:
        """Validate audio input format."""
        if not isinstance(audio_data, np.ndarray):
            return False
        if len(audio_data.shape) > 2:
            return False
        return True

    def cleanup(self) -> None:
        """Release model resources."""
        self.model = None
        self.is_initialized = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("HeAR Pipeline cleaned up")
