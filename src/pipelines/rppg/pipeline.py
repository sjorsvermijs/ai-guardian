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
        used_tensor = False
        try:
            # Handle different input types
            if isinstance(video_input, (str, Path)):
                video_path = str(video_input)
                # Process video file using open-rppg
                result = self.model.process_video(video_path)
                
            elif isinstance(video_input, np.ndarray):
                # Process numpy array directly (no video encoding!)
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
                
                print(f"✓ Processing {len(video_input)} frames directly with tensor method (no video conversion)")
                # Use process_video_tensor to avoid video encoding/decoding
                result = self.model.process_video_tensor(video_input, fps=self.fps)
                used_tensor = True
                
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
            
            # Debug: print raw result
            print(f"🔍 Raw rPPG result: {result}")
            
            # Check if we got valid results
            if not result or result.get('hr') is None:
                return PipelineResult(
                    pipeline_name="rPPG",
                    timestamp=datetime.now(),
                    confidence=0.0,
                    data={},
                    warnings=["No heart rate detected - ensure face is clearly visible"],
                    errors=[],
                    metadata={}
                )
            
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
            if used_tensor:
                # For tensor input, use the input array and configured fps
                fps = self.fps
                frame_count = len(video_input)
            else:
                # For video file input, read metadata from the file
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
        
        # Use rawvideo or huffyuv for completely lossless, all-keyframe encoding
        # AVI format with MJPEG also works (each frame is independent JPEG)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Motion JPEG - every frame is a keyframe
        temp_path_avi = temp_path.replace('.mp4', '.avi')
        out = cv2.VideoWriter(temp_path_avi, fourcc, fps, (width, height))
        temp_path = temp_path_avi
        
        if not out.isOpened():
            # Fallback to uncompressed
            fourcc = 0
            out = cv2.VideoWriter(temp_path_avi, fourcc, fps, (width, height))
        
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
    
    def process_webcam(self, camera_index: int = 0, duration: float = 10.0) -> PipelineResult:
        """
        Process video from webcam for a specified duration
        
        Args:
            camera_index: Camera index (0 for default webcam)
            duration: Duration in seconds to capture video
            
        Returns:
            PipelineResult with vital signs from captured video
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        import time
        
        print(f"Starting webcam capture (camera {camera_index}) for {duration}s...")
        print("Position yourself in front of camera, ensure good lighting...")
        
        try:
            # Use open-rppg's video_capture context manager
            with self.model.video_capture(camera_index):
                start_time = time.time()
                
                # Wait for specified duration to collect data
                while time.time() - start_time < duration:
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    
                    # Show progress every second
                    if int(elapsed) != int(elapsed - 0.1):
                        print(f"  Capturing... {remaining:.1f}s remaining", end='\r')
                    
                    time.sleep(0.1)
                
                print(f"\n  ✓ Capture complete ({duration}s)")
                
                # Get results
                result = self.model.hr(start=0, end=duration)
                
                if not result or result['hr'] is None:
                    return PipelineResult(
                        pipeline_name="rPPG",
                        timestamp=datetime.now(),
                        confidence=0.0,
                        data={},
                        warnings=["No heart rate detected - ensure face is visible"],
                        errors=[],
                        metadata={'camera_index': camera_index, 'duration': duration}
                    )
                
                # Extract results
                hr = float(result.get('hr', 0.0))
                signal_quality = float(result.get('SQI', 0.0))
                hrv_metrics = result.get('hrv', {})
                latency = result.get('latency', 0.0)
                
                # Convert breathing rate from Hz to breaths/min
                if 'breathingrate' in hrv_metrics:
                    hrv_metrics['breathingrate'] = hrv_metrics['breathingrate'] * 60
                
                # Build results dictionary
                results = {
                    "heart_rate": hr,
                    "signal_quality": signal_quality,
                    "heart_rate_variability": hrv_metrics,
                    "respiratory_rate": hrv_metrics.get('breathingrate') if hrv_metrics else None,
                    "latency_ms": latency * 1000 if latency else 0.0,
                }
                
                # Determine warnings
                warnings = []
                if signal_quality < 0.3:
                    warnings.append("Very low signal quality - improve lighting or camera position")
                elif signal_quality < 0.5:
                    warnings.append("Low signal quality - results should be interpreted with caution")
                
                if hr < 40 or hr > 180:
                    warnings.append(f"Heart rate {hr:.1f} BPM is outside normal range (40-180)")
                
                # Calculate confidence
                confidence = min(1.0, max(0.0, signal_quality))
                
                return PipelineResult(
                    pipeline_name="rPPG",
                    timestamp=datetime.now(),
                    confidence=confidence,
                    data=results,
                    warnings=warnings,
                    errors=[],
                    metadata={
                        "model_name": self.model_name,
                        "camera_index": camera_index,
                        "duration_seconds": duration,
                        "capture_mode": "webcam"
                    }
                )
                
        except Exception as e:
            return PipelineResult(
                pipeline_name="rPPG",
                timestamp=datetime.now(),
                confidence=0.0,
                data={},
                warnings=[],
                errors=[f"Webcam processing failed: {str(e)}"],
                metadata={'camera_index': camera_index}
            )
    
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
