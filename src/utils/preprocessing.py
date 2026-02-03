"""
Utility functions for data preprocessing and validation
"""

import numpy as np
from typing import Tuple, Optional
import cv2


def preprocess_audio(audio_data: np.ndarray, 
                     target_sample_rate: int = 16000,
                     normalize: bool = True) -> np.ndarray:
    """
    Preprocess audio data for HeAR pipeline
    
    Args:
        audio_data: Raw audio samples
        target_sample_rate: Target sampling rate
        normalize: Whether to normalize audio
        
    Returns:
        Preprocessed audio array
    """
    # TODO: Implement resampling if needed
    
    if normalize:
        # Normalize to [-1, 1]
        audio_data = audio_data.astype(np.float32)
        max_val = np.abs(audio_data).max()
        if max_val > 0:
            audio_data = audio_data / max_val
    
    return audio_data


def preprocess_video(video_frames: np.ndarray,
                     target_size: Tuple[int, int] = (640, 480),
                     normalize: bool = True) -> np.ndarray:
    """
    Preprocess video frames for rPPG pipeline
    
    Args:
        video_frames: Array of video frames (N, H, W, C)
        target_size: Target frame size (width, height)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed video frames
    """
    processed_frames = []
    
    for frame in video_frames:
        # Resize if needed
        if frame.shape[:2] != target_size[::-1]:
            frame = cv2.resize(frame, target_size)
        
        if normalize:
            frame = frame.astype(np.float32) / 255.0
        
        processed_frames.append(frame)
    
    return np.array(processed_frames)


def preprocess_image(image: np.ndarray,
                     target_size: Tuple[int, int] = (512, 512),
                     normalize: bool = True) -> np.ndarray:
    """
    Preprocess image for MedGemma VQA pipeline
    
    Args:
        image: Input image array
        target_size: Target image size (width, height)
        normalize: Whether to normalize pixel values
        
    Returns:
        Preprocessed image
    """
    # Resize if needed
    if image.shape[:2] != target_size[::-1]:
        image = cv2.resize(image, target_size)
    
    if normalize:
        image = image.astype(np.float32) / 255.0
    
    return image


def validate_vital_signs(heart_rate: float,
                         respiratory_rate: float,
                         spo2: float) -> Tuple[bool, list]:
    """
    Validate vital signs are within reasonable ranges
    
    Args:
        heart_rate: Heart rate in BPM
        respiratory_rate: Respiratory rate in breaths/min
        spo2: Oxygen saturation percentage
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Heart rate validation (40-200 BPM reasonable range)
    if not (40 <= heart_rate <= 200):
        issues.append(f"Heart rate {heart_rate} BPM out of range")
    
    # Respiratory rate validation (5-60 breaths/min reasonable range)
    if not (5 <= respiratory_rate <= 60):
        issues.append(f"Respiratory rate {respiratory_rate} out of range")
    
    # SpO2 validation (70-100% reasonable range)
    if not (70 <= spo2 <= 100):
        issues.append(f"SpO2 {spo2}% out of range")
    
    return len(issues) == 0, issues
