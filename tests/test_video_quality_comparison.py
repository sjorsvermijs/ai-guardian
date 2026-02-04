#!/usr/bin/env python
"""
Test rPPG pipeline with uncompressed video for better accuracy
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.rppg import RPPGPipeline
from src.core.config import config


def test_video(video_path, video_name):
    """Test pipeline with a specific video"""
    print(f"\n{'='*70}")
    print(f"Testing: {video_name}")
    print(f"{'='*70}")
    
    # Initialize pipeline
    pipeline = RPPGPipeline(config.get_pipeline_config('rppg'))
    pipeline.initialize()
    
    # Process video
    print(f"📹 Processing {video_path}...")
    result = pipeline.process(video_path)
    
    # Display results
    if result.errors:
        print(f"\n❌ Error: {result.errors[0]}")
    else:
        print(f"\n✅ Processing successful!")
        print(f"\n💓 Vital Signs:")
        print(f"   Heart Rate:     {result.data['heart_rate']:.1f} BPM")
        print(f"   Signal Quality: {result.data['signal_quality']:.1%}")
        print(f"   Confidence:     {result.confidence:.1%}")
        
        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   • {warning}")
        
        print(f"\n📊 Video Info:")
        print(f"   Model:    {result.metadata.get('model_name', 'N/A')}")
        print(f"   FPS:      {result.metadata.get('fps', 0):.2f}")
        print(f"   Frames:   {result.metadata.get('num_frames', 0)}")
        print(f"   Duration: {result.metadata.get('duration_seconds', 0):.1f} sec")
    
    pipeline.cleanup()
    return result


def main():
    print("\n" + "="*70)
    print("🏥 rPPG Pipeline - Video Quality Comparison Test")
    print("="*70)
    
    # Test compressed video
    compressed = "data/sample/video/sample_video.mp4"
    result_compressed = test_video(compressed, "Compressed Video (MP4)")
    
    # Test uncompressed video
    uncompressed = "data/sample/video/uncompressed_vid.avi"
    if Path(uncompressed).exists():
        result_uncompressed = test_video(uncompressed, "Uncompressed Video (AVI)")
        
        # Comparison
        print(f"\n{'='*70}")
        print("📈 COMPARISON")
        print(f"{'='*70}")
        
        if not result_compressed.errors and not result_uncompressed.errors:
            hr_comp = result_compressed.data['heart_rate']
            hr_uncomp = result_uncompressed.data['heart_rate']
            sq_comp = result_compressed.data['signal_quality']
            sq_uncomp = result_uncompressed.data['signal_quality']
            
            print(f"\nCompressed (MP4):")
            print(f"   Heart Rate:     {hr_comp:.1f} BPM")
            print(f"   Signal Quality: {sq_comp:.1%}")
            
            print(f"\nUncompressed (AVI):")
            print(f"   Heart Rate:     {hr_uncomp:.1f} BPM")
            print(f"   Signal Quality: {sq_uncomp:.1%}")
            
            # Calculate improvements
            sq_improvement = ((sq_uncomp - sq_comp) / sq_comp * 100) if sq_comp > 0 else 0
            
            print(f"\n📊 Results:")
            print(f"   Signal Quality Improvement: {sq_improvement:+.1f}%")
            
            if sq_uncomp > sq_comp:
                print(f"   ✅ Uncompressed video has BETTER signal quality!")
            elif sq_uncomp < sq_comp:
                print(f"   ⚠️  Compressed video performed better (unexpected)")
            else:
                print(f"   ➖ Similar signal quality")
    else:
        print(f"\n❌ Uncompressed video not found at: {uncompressed}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
