"""
Test respiratory rate extraction from rPPG pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.rppg.pipeline import RPPGPipeline
from src.core.config import config

def test_respiratory_extraction():
    """Test breathing rate extraction with high-quality video."""
    
    print("=" * 80)
    print("🫁 RESPIRATORY RATE EXTRACTION TEST")
    print("=" * 80)
    print(f"\nModel: {config.rppg_config['model_name']}")
    print("Video: data/sample/video/uncompressed_vid.avi (High quality)")
    print()
    
    # Initialize pipeline
    pipeline = RPPGPipeline(config.rppg_config)
    pipeline.initialize()
    
    # Process video
    result = pipeline.process("data/sample/video/uncompressed_vid.avi")
    
    # Extract all metrics
    hr = result.data.get('heart_rate', 'N/A')
    sqi = result.data.get('signal_quality', 0)
    hrv = result.data.get('heart_rate_variability', {})
    
    print("📊 RESULTS")
    print("=" * 80)
    print(f"Heart Rate:       {hr:.1f} BPM" if isinstance(hr, (int, float)) else f"Heart Rate:       {hr}")
    print(f"Signal Quality:   {sqi:.1%}")
    print(f"HRV Available:    {'✓ Yes' if hrv else '✗ No'}")
    print()
    
    if hrv:
        print("💓 HEART RATE VARIABILITY METRICS:")
        print("-" * 80)
        
        # Time domain
        print("\n⏱️  Time Domain:")
        print(f"  BPM (from peaks):  {hrv.get('bpm', 'N/A'):.1f}" if 'bpm' in hrv else "  BPM: N/A")
        print(f"  IBI (ms):          {hrv.get('ibi', 'N/A'):.1f}" if 'ibi' in hrv else "  IBI: N/A")
        print(f"  SDNN:              {hrv.get('sdnn', 'N/A'):.1f}" if 'sdnn' in hrv else "  SDNN: N/A")
        print(f"  RMSSD:             {hrv.get('rmssd', 'N/A'):.1f}" if 'rmssd' in hrv else "  RMSSD: N/A")
        print(f"  pNN50:             {hrv.get('pnn50', 'N/A'):.1f}" if 'pnn50' in hrv else "  pNN50: N/A")
        print(f"  pNN20:             {hrv.get('pnn20', 'N/A'):.1f}" if 'pnn20' in hrv else "  pNN20: N/A")
        
        # Frequency domain
        print("\n📡 Frequency Domain:")
        print(f"  VLF:               {hrv.get('VLF', 'N/A'):.4f}" if 'VLF' in hrv else "  VLF: N/A")
        print(f"  LF:                {hrv.get('LF', 'N/A'):.4f}" if 'LF' in hrv else "  LF: N/A")
        print(f"  HF:                {hrv.get('HF', 'N/A'):.4f}" if 'HF' in hrv else "  HF: N/A")
        print(f"  TP (Total Power):  {hrv.get('TP', 'N/A'):.4f}" if 'TP' in hrv else "  TP: N/A")
        print(f"  LF/HF Ratio:       {hrv.get('LF/HF', 'N/A'):.2f}" if 'LF/HF' in hrv else "  LF/HF: N/A")
        
        # RESPIRATORY RATE
        print("\n🫁 RESPIRATORY:")
        rr_main = result.data.get('respiratory_rate')
        if rr_main is not None:
            print(f"  Breathing Rate:    {rr_main:.1f} breaths/min")
            
            # Clinical interpretation
            if rr_main < 12:
                status = "⚠️  BRADYPNEA (Slow breathing)"
            elif 12 <= rr_main <= 20:
                status = "✅ NORMAL"
            elif 20 < rr_main <= 30:
                status = "⚠️  TACHYPNEA (Fast breathing)"
            else:
                status = "🚨 SEVERE TACHYPNEA"
            
            print(f"  Status:            {status}")
            print(f"\n  Reference: Normal adult = 12-20 breaths/min")
            print(f"  Method:            Respiratory Sinus Arrhythmia (RSA) from HRV")
        else:
            print("  Breathing Rate:    ✗ Not available")
            print("  (Requires SQI > 0.5 for HRV calculation)")
    else:
        print("⚠️  HRV metrics not available (SQI too low)")
        print(f"   Current SQI: {sqi:.1%}")
        print("   Required: > 50%")
    
    print("\n" + "=" * 80)
    
    # Cleanup
    pipeline.cleanup()

if __name__ == "__main__":
    test_respiratory_extraction()
