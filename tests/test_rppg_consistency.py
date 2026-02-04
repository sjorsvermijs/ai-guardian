#!/usr/bin/env python
"""
Test consistency of rPPG pipeline - check if same video gives same results
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.rppg import RPPGPipeline
from src.core.config import config


def test_consistency(video_path, video_name, num_runs=5):
    """Test pipeline multiple times with same video"""
    print(f"\n{'='*70}")
    print(f"Testing Consistency: {video_name}")
    print(f"Running {num_runs} times on same video")
    print(f"{'='*70}")
    
    results = []
    
    for i in range(num_runs):
        # Initialize fresh pipeline each time
        pipeline = RPPGPipeline(config.get_pipeline_config('rppg'))
        pipeline.initialize()
        
        # Process video
        result = pipeline.process(video_path)
        
        if not result.errors:
            hr = result.data['heart_rate']
            sq = result.data['signal_quality']
            results.append({'hr': hr, 'sq': sq, 'run': i+1})
            print(f"Run {i+1}: HR={hr:.2f} BPM, SQ={sq:.4f}")
        else:
            print(f"Run {i+1}: ERROR - {result.errors[0]}")
        
        pipeline.cleanup()
    
    if results:
        # Calculate statistics
        hrs = [r['hr'] for r in results]
        sqs = [r['sq'] for r in results]
        
        hr_mean = sum(hrs) / len(hrs)
        hr_min = min(hrs)
        hr_max = max(hrs)
        hr_std = (sum((x - hr_mean)**2 for x in hrs) / len(hrs)) ** 0.5
        
        sq_mean = sum(sqs) / len(sqs)
        sq_std = (sum((x - sq_mean)**2 for x in sqs) / len(sqs)) ** 0.5
        
        print(f"\n📊 Statistics:")
        print(f"   Heart Rate:")
        print(f"      Mean:  {hr_mean:.2f} BPM")
        print(f"      Range: {hr_min:.2f} - {hr_max:.2f} BPM")
        print(f"      StdDev: {hr_std:.2f} BPM")
        print(f"      Variance: {hr_max - hr_min:.2f} BPM")
        
        print(f"\n   Signal Quality:")
        print(f"      Mean:  {sq_mean:.4f}")
        print(f"      StdDev: {sq_std:.4f}")
        
        # Assess consistency
        if hr_std < 1.0:
            print(f"\n   ✅ Very consistent results (StdDev < 1 BPM)")
        elif hr_std < 5.0:
            print(f"\n   ✅ Consistent results (StdDev < 5 BPM)")
        elif hr_std < 10.0:
            print(f"\n   ⚠️  Moderate variance (StdDev < 10 BPM)")
        else:
            print(f"\n   ❌ High variance - results unreliable!")
        
        return hr_std, sq_std
    
    return None, None


def main():
    print("\n" + "="*70)
    print("🔬 rPPG Pipeline Consistency Test")
    print("="*70)
    
    # Test compressed video
    compressed = "data/sample/video/sample_video.mp4"
    hr_std_comp, sq_std_comp = test_consistency(compressed, "Compressed Video (MP4)", num_runs=5)
    
    # Test uncompressed video
    uncompressed = "data/sample/video/uncompressed_vid.avi"
    if Path(uncompressed).exists():
        hr_std_uncomp, sq_std_uncomp = test_consistency(uncompressed, "Uncompressed Video (AVI)", num_runs=5)
        
        # Compare consistency
        if hr_std_comp and hr_std_uncomp:
            print(f"\n{'='*70}")
            print("🔍 CONSISTENCY COMPARISON")
            print(f"{'='*70}")
            
            print(f"\nCompressed (MP4):")
            print(f"   HR Std Dev:  {hr_std_comp:.2f} BPM")
            print(f"   SQ Std Dev:  {sq_std_comp:.4f}")
            
            print(f"\nUncompressed (AVI):")
            print(f"   HR Std Dev:  {hr_std_uncomp:.2f} BPM")
            print(f"   SQ Std Dev:  {sq_std_uncomp:.4f}")
            
            if hr_std_uncomp < hr_std_comp:
                improvement = ((hr_std_comp - hr_std_uncomp) / hr_std_comp * 100)
                print(f"\n   ✅ Uncompressed video is {improvement:.1f}% more consistent!")
            else:
                print(f"\n   ⚠️  Compressed video is more consistent (unexpected)")
    
    print("\n" + "="*70)
    print("\n💡 Explanation:")
    print("   Variance in results can come from:")
    print("   • Low signal quality causing unstable readings")
    print("   • Neural network stochasticity (if model has dropout/randomness)")
    print("   • Floating point precision differences")
    print("   • Frame processing order variations")
    print("\n   Better signal quality → More consistent results")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
