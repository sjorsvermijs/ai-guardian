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
import pickle
from dotenv import load_dotenv

from ...core.base_pipeline import BasePipeline, PipelineResult

# Default model cache directory (project_root/models)
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "models"

# Load environment variables from .env file in project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
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
        self.classifier_binary = None
        self.classifier_multiclass = None
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

            # Check for MPS (Apple Silicon), CUDA (NVIDIA), then fallback to CPU
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

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

            # Load SPRSound classifiers
            binary_path = self.cache_dir / "sprsound_classifier" / "mlp_binary.pkl"
            multiclass_path = self.cache_dir / "sprsound_classifier" / "mlp_multiclass.pkl"

            if binary_path.exists():
                with open(binary_path, 'rb') as f:
                    self.classifier_binary = pickle.load(f)
                print(f"✓ Loaded SPRSound binary classifier (Normal vs Adventitious)")
            else:
                print(f"⚠️ Binary classifier not found at {binary_path}")

            if multiclass_path.exists():
                with open(multiclass_path, 'rb') as f:
                    self.classifier_multiclass = pickle.load(f)
                print(f"✓ Loaded SPRSound multiclass classifier (CAS/DAS/Normal/Poor)")
            else:
                print(f"⚠️ Multiclass classifier not found at {multiclass_path}")

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

            result_data = {
                "embedding_dim": 512,
                "num_chunks": len(embeddings),
            }

            # If classifiers are loaded, make predictions
            if self.classifier_binary is not None or self.classifier_multiclass is not None:
                try:
                    # Make predictions for each chunk
                    binary_predictions = []
                    multiclass_predictions = []

                    for embedding in embeddings:
                        emb_reshaped = np.array(embedding).reshape(1, -1)

                        if self.classifier_binary is not None:
                            binary_prob = self.classifier_binary.predict_proba(emb_reshaped)[0]
                            binary_predictions.append(binary_prob)

                        if self.classifier_multiclass is not None:
                            multi_prob = self.classifier_multiclass.predict_proba(emb_reshaped)[0]
                            multiclass_predictions.append(multi_prob)

                    # Aggregate predictions across chunks (average probabilities)
                    # Process multiclass FIRST so we can use "Poor Quality" to
                    # override the binary classifier (which has no quality awareness).
                    is_poor_quality = False
                    if multiclass_predictions:
                        avg_multi_prob = np.mean(multiclass_predictions, axis=0)
                        multi_labels = ["CAS", "CAS & DAS", "DAS", "Normal", "Poor Quality"]
                        multi_pred_idx = np.argmax(avg_multi_prob)
                        is_poor_quality = multi_labels[multi_pred_idx] == "Poor Quality"

                        result_data["multiclass_classification"] = {
                            "prediction": multi_labels[multi_pred_idx],
                            "confidence": float(avg_multi_prob[multi_pred_idx]),
                            "probabilities": {
                                label: float(prob)
                                for label, prob in zip(multi_labels, avg_multi_prob)
                            },
                            "note": "CAS=Crackles, DAS=Wheezes"
                        }

                    if binary_predictions:
                        avg_binary_prob = np.mean(binary_predictions, axis=0)
                        binary_labels = ["Normal", "Adventitious"]
                        binary_pred_idx = np.argmax(avg_binary_prob)
                        binary_confidence = float(
                            avg_binary_prob[binary_pred_idx])

                        # Override binary prediction when audio quality is poor or
                        # confidence is too low. HeAR was trained on stethoscope
                        # recordings; phone/ambient audio is out-of-distribution.
                        min_confidence = self.config.get(
                            'min_confidence', 0.75)
                        if is_poor_quality:
                            binary_prediction = "Inconclusive"
                        elif binary_confidence < min_confidence:
                            binary_prediction = "Inconclusive"
                        else:
                            binary_prediction = binary_labels[binary_pred_idx]

                        result_data["binary_classification"] = {
                            "prediction": binary_prediction,
                            "confidence": binary_confidence,
                            "raw_prediction": binary_labels[binary_pred_idx],
                            "probabilities": {
                                label: float(prob)
                                for label, prob in zip(binary_labels, avg_binary_prob)
                            },
                            "audio_source": "ambient",
                        }

                    confidence = result_data.get("binary_classification", {}).get("confidence", 0.95)

                except Exception as e:
                    print(f"⚠️ Classifier prediction failed: {e}")
                    import traceback
                    traceback.print_exc()
                    result_data["embeddings"] = embeddings  # Fallback
                    confidence = 0.95
            else:
                result_data["embeddings"] = embeddings  # No classifiers, return embeddings
                confidence = 0.95

            return PipelineResult(
                pipeline_name="HeAR",
                timestamp=datetime.now(),
                confidence=confidence,
                data=result_data,
                warnings=[],
                errors=[],
                metadata={
                    "sample_rate": SAMPLE_RATE,
                    "duration_seconds": len(audio_data) / SAMPLE_RATE,
                    "model": self.model_name,
                    "binary_classifier_available": self.classifier_binary is not None,
                    "multiclass_classifier_available": self.classifier_multiclass is not None,
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
        """Extract embeddings from audio in 2-second chunks (batched)."""
        # Preprocess all chunks into spectrograms
        spectrograms = []
        for start in range(0, len(audio_data), CLIP_LENGTH):
            chunk = audio_data[start : start + CLIP_LENGTH]
            audio_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
            spectrograms.append(preprocess_audio(audio_tensor))

        # Batch all spectrograms into a single forward pass
        batch = torch.cat(spectrograms, dim=0).to(self.device)

        with torch.no_grad():
            output = self.model(batch, return_dict=True, output_hidden_states=True)
            # pooler_output shape: (num_chunks, 512)
            all_embeddings = output.pooler_output.cpu().numpy()

        return [all_embeddings[i] for i in range(all_embeddings.shape[0])]

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
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("HeAR Pipeline cleaned up")
