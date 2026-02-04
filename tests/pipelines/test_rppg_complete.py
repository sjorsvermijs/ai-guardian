#!/usr/bin/env python
"""
Standalone test script for rPPG Pipeline
Tests both video file and numpy array inputs
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.rppg import RPPGPipeline
from src.core.config import config


def test_with_video_file():
    """Test pipeline with video file path"""
    print("="*70)
    print("TEST 1: Video File Input")
    print("="*70)
    
    # Initialize pipeline
    print("\n📦 Initializing rPPG Pipeline...")
    pipeline = RPPGPipeline(config.get_pipeline_config('rppg'))
    
    if not pipeline.initialize():
        print("❌ Failed to initialize pipeline")
        return None
    
    print(f"✅ Initialized with model: {pipeline.model_name}")
    
    # Process sample video
    video_path = Path("data/sample/video/sample_video.mp4")
    
    if not video_path.exists():
        print(f"\n❌ Video file not found: {video_path}")
        pipeline.cleanup()
        return None
    
    print(f"\n🎬 Processing: {video_path.name}")
    result = pipeline.process(video_path)
    
    # Display results
    display_results(result)
    
    # Cleanup
    pipeline.cleanup()
    return result


def test_with_numpy_array():
    """Test pipeline with numpy array of frames"""
    print("\n" + "="*70)
    print("TEST 2: NumPy Array Input")
    print("="*70)
    
    # Load video into numpy array
    video_path = Path("data/sample/video/sample_video.mp4")
    
    if not video_path.exists():
        print(f"\n❌ Video file not found: {video_path}")
        return None
    
    print("\n📹 Loading video frames into numpy array...")
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    # Load first 300 frames (~10 seconds at 30fps)
    max_frames = 300
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1
    
    cap.release()
    
    frames_array = np.array(frames)
    print(f"✅ Loaded {len(frames)} frames")
    print(f"   Shape: {frames_array.shape}")
    
    # Initialize pipeline
    print("\n📦 Initializing rPPG Pipeline...")
    pipeline = RPPGPipeline(config.get_pipeline_config('rppg'))
    
    if not pipeline.initialize():
        print("❌ Failed to initialize pipeline")
        return None
    
    print(f"✅ Initialized with model: {pipeline.model_name}")
    
    # Process frames
    print(f"\n🎬 Processing numpy array...")
    result = pipeline.process(frames_array)
    
    # Display results
    display_results(result)
    
    # Cleanup
    pipeline.cleanup()
    return result


def display_results(result):
    """Display pipeline results in a formatted way"""
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    
    if result.errors:
        print(f"\n❌ Errors:")
        for error in result.errors:
            print(f"   • {error}")
        return
    
    print(f"\n✅ Processing successful!")
    print(f"   Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Confidence: {result.confidence:.1%}")
    
    print(f"\n💓 Vital Signs:")
    hr = result.data.get('heart_rate', 0)
    sq = result.data.get('signal_quality', 0)
    
    print(f"   Heart Rate:     {hr:.1f} BPM")
    print(f"   Signal Quality: {sq:.1%}")
    
    if 'latency_ms' in result.data:
        print(f"   Latency:        {result.data['latency_ms']:.1f} ms")
    
    if 'bvp_signal_length' in result.data:
        print(f"   BVP Samples:    {result.data['bvp_signal_length']}")
    
    if result.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in result.warnings:
            print(f"   • {warning}")
    
    print(f"\n📈 Video Info:")
    print(f"   Model:    {result.metadata.get('model_name', 'N/A')}")
    print(f"   FPS:      {result.metadata.get('fps', 0):.2f}")
    print(f"   Frames:   {result.metadata.get('num_frames', 0)}")
    print(f"   Duration: {result.metadata.get('duration_seconds', 0):.1f} sec")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🏥 AI Guardian - rPPG Pipeline Test Suite")
    print("="*70)
    
    # Test 1: Video file input
    result1 = test_with_video_file()
    
    # Test 2: NumPy array input
    result2 = test_with_numpy_array()
    
    # Summary
    print("\n" + "="*70)
    print("📝 SUMMARY")
    print("="*70)
    
    if result1 and not result1.errors:
        print(f"✅ Video File Test:   PASSED (HR: {result1.data.get('heart_rate', 0):.1f} BPM)")
    else:
        print("❌ Video File Test:   FAILED")
    
    if result2 and not result2.errors:
        print(f"✅ NumPy Array Test:  PASSED (HR: {result2.data.get('heart_rate', 0):.1f} BPM)")
    else:
        print("❌ NumPy Array Test:  FAILED")
    
    print("\n" + "="*70)
    print("✨ All tests completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
