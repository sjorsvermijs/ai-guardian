#!/usr/bin/env python3
"""
Webcam rPPG Demo - Monitor vital signs from laptop webcam
"""
import os
import sys
import warnings

# Set environment variable for macOS camera permissions
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL')
warnings.filterwarnings('ignore', message='Class AVF')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.rppg.pipeline import RPPGPipeline
from src.core.config import config

def display_results(result):
    """Display results in a formatted way."""
    print("\n" + "=" * 80)
    print("💓 VITAL SIGNS RESULTS")
    print("=" * 80)
    
    hr = result.data.get('heart_rate')
    rr = result.data.get('respiratory_rate')
    sqi = result.data.get('signal_quality', 0)
    hrv = result.data.get('heart_rate_variability', {})
    
    print(f"\n📊 Measurements:")
    print(f"  Heart Rate:       {hr:.1f} BPM" if hr else "  Heart Rate:       N/A")
    if rr:
        print(f"  Respiratory Rate: {rr:.1f} breaths/min")
        
        # Clinical interpretation
        if rr < 12:
            status = "⚠️  Bradypnea (low)"
        elif 12 <= rr <= 20:
            status = "✅ Normal"
        elif 20 < rr <= 30:
            status = "⚠️  Tachypnea (high)"
        else:
            status = "🚨 Severe tachypnea"
        print(f"                    {status}")
    else:
        print(f"  Respiratory Rate: Not available (requires SQI > 50%)")
    
    print(f"  Signal Quality:   {sqi:.1%}")
    print(f"  Confidence:       {result.confidence:.1%}")
    
    if hrv and 'LF/HF' in hrv:
        print(f"\n🫀 HRV Metrics:")
        print(f"  SDNN:             {hrv.get('sdnn', 'N/A'):.1f} ms")
        print(f"  RMSSD:            {hrv.get('rmssd', 'N/A'):.1f} ms")
        print(f"  LF/HF Ratio:      {hrv.get('LF/HF', 'N/A'):.2f}")
        
        lf_hf = hrv.get('LF/HF', 0)
        if lf_hf > 2.5:
            balance = "Sympathetic dominant (stressed)"
        elif lf_hf < 1.5:
            balance = "Parasympathetic dominant (relaxed)"
        else:
            balance = "Balanced"
        print(f"  Autonomic:        {balance}")
    
    if result.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in result.warnings:
            print(f"  • {warning}")
    
    print("\n" + "=" * 80)

def main():
    """Main webcam monitoring function."""
    print("=" * 80)
    print("🎥 AI GUARDIAN - WEBCAM rPPG MONITOR")
    print("=" * 80)
    print()
    print("This tool measures your vital signs using your laptop webcam.")
    print("Uses remote photoplethysmography (rPPG) to detect subtle color")
    print("changes in your face caused by blood flow.")
    print()
    print("Tips for best results:")
    print("  • Sit in a well-lit area (natural light is best)")
    print("  • Face the camera directly")
    print("  • Stay as still as possible")
    print("  • Remove glasses if possible")
    print("  • Avoid heavy makeup")
    print()
    
    # Get user input
    try:
        duration = float(input("Enter capture duration in seconds (recommended: 30-60): ") or "30")
        camera_index = int(input("Enter camera index (0 for default, 1 for external): ") or "0")
    except ValueError:
        print("Invalid input, using defaults: 30s on camera 0")
        duration = 30
        camera_index = 0
    
    if duration < 10:
        print("\n⚠️  Warning: Duration < 10s may give unreliable results")
        print("   Recommended minimum: 30 seconds")
    
    print(f"\n{'=' * 80}")
    print("Initializing rPPG pipeline...")
    
    # Initialize pipeline
    pipeline = RPPGPipeline(config.rppg_config)
    pipeline.initialize()
    
    print(f"✓ Pipeline ready (model: {config.rppg_config['model_name']})")
    print()
    
    # Process webcam
    result = pipeline.process_webcam(camera_index=camera_index, duration=duration)
    
    # Display results
    if result.errors:
        print("\n❌ ERROR:")
        for error in result.errors:
            print(f"  {error}")
    else:
        display_results(result)
    
    # Cleanup
    pipeline.cleanup()
    
    print("\nMonitoring complete!")
    
    # Ask to repeat
    repeat = input("\nRun another measurement? (y/n): ").lower()
    if repeat == 'y':
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user.")
        sys.exit(0)
