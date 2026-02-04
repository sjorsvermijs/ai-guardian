#!/usr/bin/env python
"""
Quick test of rPPG pipeline - Run this to test the pipeline!
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.rppg import RPPGPipeline
from src.core.config import config


def main():
    # Initialize
    pipeline = RPPGPipeline(config.get_pipeline_config('rppg'))
    pipeline.initialize()
    
    # Process video
    video_path = "data/sample/video/sample_video.mp4"
    result = pipeline.process(video_path)
    
    # Show results
    if result.errors:
        print(f"❌ Error: {result.errors[0]}")
    else:
        print(f"✅ Heart Rate: {result.data['heart_rate']:.1f} BPM")
        print(f"📊 Quality: {result.data['signal_quality']:.1%}")
        print(f"⏱️  Processed: {result.metadata['duration_seconds']:.1f} sec video")
    
    # Cleanup
    pipeline.cleanup()


if __name__ == "__main__":
    main()
