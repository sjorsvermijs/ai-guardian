"""
Test all available rPPG models on uncompressed video to find the best performer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.rppg.pipeline import RPPGPipeline
from src.core.config import config
import time

# Slow Mamba models only
MODELS = [
    'PhysMamba.pure', 'PhysMamba.rlap', 'RhythmMamba.rlap', 'RhythmMamba.pure'
]

VIDEO_PATH = "data/sample/video/uncompressed_vid.avi"

def test_model(model_name):
    """Test a single model and return results."""
    try:
        # Create config for this model
        test_config = {
            'model_name': model_name,
            'fps': config.rppg_config['fps'],
            'window_size': config.rppg_config['window_size'],
            'roi_method': config.rppg_config['roi_method'],
            'signal_processing': config.rppg_config['signal_processing']
        }
        
        # Initialize pipeline
        pipeline = RPPGPipeline(test_config)
        pipeline.initialize()
        
        # Process video
        start_time = time.time()
        result = pipeline.process(VIDEO_PATH)
        processing_time = time.time() - start_time
        
        # Extract metrics
        signal_quality = result.data.get('signal_quality', 0)
        hr = result.data.get('heart_rate', 0)
        hrv_available = bool(result.data.get('hrv', {}))
        confidence = result.confidence
        
        return {
            'model': model_name,
            'success': True,
            'signal_quality': signal_quality,
            'heart_rate': hr,
            'hrv_available': hrv_available,
            'confidence': confidence,
            'processing_time': processing_time,
            'warnings': len(result.warnings),
            'errors': len(result.errors)
        }
    except Exception as e:
        return {
            'model': model_name,
            'success': False,
            'error': str(e),
            'signal_quality': 0,
            'heart_rate': 0,
            'hrv_available': False,
            'confidence': 0,
            'processing_time': 0
        }

def main():
    print("=" * 80)
    print("TESTING ALL rPPG MODELS ON UNCOMPRESSED VIDEO")
    print("=" * 80)
    print(f"\nVideo: {VIDEO_PATH}")
    print(f"Total models to test: {len(MODELS)}\n")
    
    results = []
    
    for i, model in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] Testing {model}...", end=" ", flush=True)
        result = test_model(model)
        results.append(result)
        
        if result['success']:
            print(f"✓ SQI: {result['signal_quality']:.1%}, HR: {result['heart_rate']:.1f} BPM, Time: {result['processing_time']:.1f}s")
        else:
            print(f"✗ Error: {result['error']}")
    
    # Sort by signal quality
    results.sort(key=lambda x: x['signal_quality'], reverse=True)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (Ranked by Signal Quality)")
    print("=" * 80)
    print(f"\n{'Rank':<6}{'Model':<22}{'SQI':<10}{'HR (BPM)':<12}{'HRV':<8}{'Confidence':<12}{'Time (s)':<10}")
    print("-" * 80)
    
    for i, r in enumerate(results, 1):
        if r['success']:
            hrv_status = "✓" if r['hrv_available'] else "✗"
            print(f"{i:<6}{r['model']:<22}{r['signal_quality']:<10.1%}{r['heart_rate']:<12.1f}{hrv_status:<8}{r['confidence']:<12.1%}{r['processing_time']:<10.1f}")
        else:
            print(f"{i:<6}{r['model']:<22}FAILED")
    
    # Best model
    best = results[0]
    if best['success']:
        print("\n" + "=" * 80)
        print("BEST PERFORMING MODEL")
        print("=" * 80)
        print(f"Model: {best['model']}")
        print(f"Signal Quality: {best['signal_quality']:.1%}")
        print(f"Heart Rate: {best['heart_rate']:.1f} BPM")
        print(f"HRV Available: {'Yes' if best['hrv_available'] else 'No'}")
        print(f"Confidence: {best['confidence']:.1%}")
        print(f"Processing Time: {best['processing_time']:.1f}s")
    
    # Success rate
    successful = sum(1 for r in results if r['success'])
    print(f"\nSuccess Rate: {successful}/{len(MODELS)} ({successful/len(MODELS):.1%})")
    
    # HRV capable models
    hrv_models = [r['model'] for r in results if r['success'] and r['hrv_available']]
    if hrv_models:
        print(f"\nModels with HRV (SQI > 0.5): {len(hrv_models)}")
        for model in hrv_models[:5]:  # Top 5
            print(f"  - {model}")

if __name__ == "__main__":
    main()
