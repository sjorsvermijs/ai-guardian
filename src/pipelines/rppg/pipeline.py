"""
rPPG Pipeline Implementation
Extracts vital signs (heart rate, SpO2, respiratory rate) from video frames
"""

from typing import Any, Dict, Union
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import tempfile

from ...core.base_pipeline import BasePipeline, PipelineResult


class RPPGPipeline(BasePipeline):
    """
    Remote Photoplethysmography (rPPG) Pipeline
    Extracts vital signs from video frames using subtle color changes in skin
    Uses the open-rppg library for rPPG signal extraction
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.model_name = config.get('model_name', 'PhysNet.pure')
        self.fps = config.get('fps', 30)
        
    def initialize(self) -> bool:
        """
        Initialize rPPG model using open-rppg library
        
        Returns:
            bool: True if successful
        """
        try:
            import rppg
            
            # Validate model name
            if self.model_name not in rppg.supported_models:
                print(f"Warning: Model '{self.model_name}' not in supported models.")
                print(f"Available models: {rppg.supported_models}")
                print(f"Falling back to 'PhysNet.pure'")
                self.model_name = 'PhysNet.pure'
            
            # Initialize the model
            self.model = rppg.Model(self.model_name)
            self.is_initialized = True
            print(f"rPPG Pipeline initialized successfully with model: {self.model_name}")
            return True
        except ImportError as e:
            print(f"rPPG Pipeline initialization failed: open-rppg not installed. {e}")
            return False
        except Exception as e:
            print(f"rPPG Pipeline initialization failed: {e}")
            return False
    
    def process(self, video_input: Union[str, Path, np.ndarray]) -> PipelineResult:
        """
        Process video to extract vital signs using open-rppg
        
        Args:
            video_input: Can be:
                - String/Path: Path to video file
                - np.ndarray: Video frames of shape (num_frames, height, width, channels)
        
        Why NumPy Array Input is Supported:
            1. Real-time Processing: Process live camera feeds without disk I/O
            2. Pipeline Integration: Use with existing CV workflows (OpenCV, etc.)
            3. Custom Preprocessing: Apply filters/adjustments before rPPG analysis
            4. Memory Efficiency: Process video segments without full file load
            5. Testing: Create synthetic test data easily
        
        Note:
            NumPy arrays are internally converted to temporary video files
            (required by open-rppg library). Temporary files are automatically
            cleaned up after processing.
            
        Returns:
            PipelineResult with vital signs data
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        temp_file = None
        try:
            # Handle different input types
            if isinstance(video_input, (str, Path)):
                video_path = str(video_input)
            elif isinstance(video_input, np.ndarray):
                # Convert numpy array to temporary video file
                if not self.validate_input(video_input):
                    return PipelineResult(
                        pipeline_name="rPPG",
                        timestamp=datetime.now(),
                        confidence=0.0,
                        data={},
                        warnings=[],
                        errors=["Invalid video input format"],
                        metadata={}
                    )
                video_path = self._frames_to_video(video_input)
                temp_file = video_path
            else:
                return PipelineResult(
                    pipeline_name="rPPG",
                    timestamp=datetime.now(),
                    confidence=0.0,
                    data={},
                    warnings=[],
                    errors=["Unsupported input type. Use video path or numpy array."],
                    metadata={}
                )
            
            # Process video using open-rppg
            result = self.model.process_video(video_path)
            
            # Extract results
            hr = float(result.get('hr', 0.0))
            signal_quality = float(result.get('SQI', 0.0))
            hrv_metrics = result.get('hrv', {})
            latency = result.get('latency', 0.0)
            
            # Convert breathing rate from Hz to breaths/min
            if 'breathingrate' in hrv_metrics:
                hrv_metrics['breathingrate'] = hrv_metrics['breathingrate'] * 60
            
            # Get BVP signal if available
            bvp_signal = None
            try:
                bvp_data = self.model.bvp()
                if isinstance(bvp_data, tuple) and len(bvp_data) == 2:
                    bvp_signal = bvp_data[0]  # First element is the signal
            except:
                pass
            
            # Build results dictionary
            results = {
                "heart_rate": hr,
                "signal_quality": signal_quality,
                "heart_rate_variability": hrv_metrics,
                "respiratory_rate": hrv_metrics.get('breathingrate') if hrv_metrics else None,
                "latency_ms": latency * 1000 if latency else 0.0,
            }
            
            if bvp_signal is not None:
                results["bvp_signal_length"] = len(bvp_signal) if hasattr(bvp_signal, '__len__') else 0
            
            # Determine warnings
            warnings = []
            if signal_quality < 0.3:
                warnings.append("Very low signal quality - results may be unreliable")
            elif signal_quality < 0.5:
                warnings.append("Low signal quality - results should be interpreted with caution")
            
            if hr < 40 or hr > 180:
                warnings.append(f"Heart rate {hr:.1f} BPM is outside normal range (40-180)")
            
            # Calculate confidence based on signal quality
            confidence = min(1.0, max(0.0, signal_quality))
            
            # Get video metadata
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            return PipelineResult(
                pipeline_name="rPPG",
                timestamp=datetime.now(),
                confidence=confidence,
                data=results,
                warnings=warnings,
                errors=[],
                metadata={
                    "model_name": self.model_name,
                    "fps": fps,
                    "num_frames": frame_count,
                    "duration_seconds": frame_count / fps if fps > 0 else 0
                }
            )
            
        except Exception as e:
            return PipelineResult(
                pipeline_name="rPPG",
                timestamp=datetime.now(),
                confidence=0.0,
                data={},
                warnings=[],
                errors=[f"Processing error: {str(e)}"],
                metadata={}
            )
        finally:
            # Clean up temporary file if created
            if temp_file and Path(temp_file).exists():
                try:
                    Path(temp_file).unlink()
                except:
                    pass
    
    def _frames_to_video(self, frames: np.ndarray) -> str:
        """
        Convert numpy array of frames to a temporary video file
        
        This method enables flexible input formats while working with the
        open-rppg library which requires video files. The conversion allows:
        - Real-time camera feed processing
        - Integration with existing video processing pipelines
        - Custom preprocessing before rPPG analysis
        - Partial video processing without full file I/O
        
        Args:
            frames: Array of shape (num_frames, height, width, channels)
            
        Returns:
            Path to temporary video file (automatically cleaned up after use)
            
        Performance Note:
            Small overhead for conversion, but benefits of flexibility usually
            outweigh the cost for real-time and custom preprocessing workflows.
        """
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        
        # Get frame properties
        height, width = frames[0].shape[:2]
        fps = self.fps
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            # Ensure frame is in BGR format (OpenCV default)
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            # If already BGR, use as is
            
            out.write(frame.astype(np.uint8))
        
        out.release()
        return temp_path
    
    def validate_input(self, video_frames: Any) -> bool:
        """Validate video input format"""
        if not isinstance(video_frames, np.ndarray):
            return False
        # Accept both (frames, height, width, channels) and (frames, height, width)
        if len(video_frames.shape) not in [3, 4]:
            return False
        if len(video_frames) == 0:
            return False
        return True
    
    def cleanup(self) -> None:
        """Release model resources"""
        if self.model is not None:
            try:
                self.model.stop()
            except:
                pass
        self.model = None
        self.is_initialized = False
        print("rPPG Pipeline cleaned up")
