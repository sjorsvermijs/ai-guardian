#!/usr/bin/env python
"""
Test script for rPPG Pipeline
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.rppg import RPPGPipeline
from src.core.config import config

def main():
    print("="*70)
    print("AI Guardian - rPPG Pipeline Test")
    print("="*70)
    
    # Initialize pipeline
    print("\n1. Initializing rPPG Pipeline...")
    pipeline = RPPGPipeline(config.get_pipeline_config('rppg'))
    
    if not pipeline.initialize():
        print("❌ Failed to initialize pipeline")
        return
    
    print("✅ Pipeline initialized successfully!")
    print(f"   Model: {pipeline.model_name}")
    
    # Process sample video
    video_path = Path("data/sample/video/sample_video.mp4")
    
    if not video_path.exists():
        print(f"\n❌ Video file not found: {video_path}")
        return
    
    print(f"\n2. Processing video: {video_path.name}")
    print(f"   This may take a moment...")
    
    result = pipeline.process(video_path)
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\nPipeline: {result.pipeline_name}")
    print(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Confidence: {result.confidence:.2%}")
    
    if result.errors:
        print(f"\n❌ Errors:")
        for error in result.errors:
            print(f"   • {error}")
    else:
        print(f"\n✅ Processing successful!")
        
        print(f"\n📊 Vital Signs:")
        print(f"   Heart Rate: {result.data.get('heart_rate', 'N/A'):.1f} BPM")
        print(f"   Signal Quality: {result.data.get('signal_quality', 0):.2%}")
        
        if 'heart_rate_variability' in result.data:
            hrv = result.data['heart_rate_variability']
            if hrv:
                print(f"   HRV Metrics: {len(hrv)} parameters")
        
        if 'bvp_signal_length' in result.data:
            print(f"   BVP Signal Length: {result.data['bvp_signal_length']} samples")
        
        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   • {warning}")
        
        print(f"\n📈 Metadata:")
        print(f"   Model: {result.metadata.get('model_name', 'N/A')}")
        print(f"   FPS: {result.metadata.get('fps', 0):.2f}")
        print(f"   Frames: {result.metadata.get('num_frames', 0)}")
        print(f"   Duration: {result.metadata.get('duration_seconds', 0):.2f} seconds")
    
    # Cleanup
    print("\n3. Cleaning up...")
    pipeline.cleanup()
    print("✅ Done!")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
