"""
Test the fusion engine with respiratory rate from rPPG pipeline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.rppg.pipeline import RPPGPipeline
from src.core.fusion_engine import FusionEngine
from src.core.config import config

def test_fusion_with_respiratory():
    """Test fusion engine with rPPG including respiratory rate."""
    
    print("=" * 80)
    print("🏥 FUSION ENGINE TEST - Respiratory Rate Integration")
    print("=" * 80)
    print()
    
    # Initialize rPPG pipeline
    print("Initializing rPPG pipeline...")
    rppg_pipeline = RPPGPipeline(config.rppg_config)
    rppg_pipeline.initialize()
    
    # Process high-quality video
    print("Processing uncompressed video...")
    rppg_result = rppg_pipeline.process("data/sample/video/uncompressed_vid.avi")
    
    print(f"✓ rPPG processing complete")
    print(f"  Heart Rate:       {rppg_result.data.get('heart_rate', 'N/A'):.1f} BPM")
    print(f"  Respiratory Rate: {rppg_result.data.get('respiratory_rate', 'N/A'):.1f} breaths/min")
    print(f"  Signal Quality:   {rppg_result.data.get('signal_quality', 0):.1%}")
    print()
    
    # Initialize fusion engine
    print("Initializing fusion engine...")
    fusion = FusionEngine(config.fusion_config if hasattr(config, 'fusion_config') else {})
    
    # Fuse results (only rPPG for now)
    print("Fusing pipeline results...")
    triage = fusion.fuse(rppg_result=rppg_result)
    
    # Display triage report
    print("\n" + "=" * 80)
    print("📋 TRIAGE REPORT")
    print("=" * 80)
    print(f"\nTimestamp:     {triage.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Priority:      {triage.priority.value}")
    print(f"Confidence:    {triage.confidence:.1%}")
    print(f"\nSummary:")
    print(f"  {triage.summary}")
    
    print("\n💓 VITAL SIGNS:")
    print("-" * 80)
    for key, value in triage.vital_signs.items():
        if isinstance(value, (int, float)):
            print(f"  {key:25} {value:.2f}")
        else:
            print(f"  {key:25} {value}")
    
    if triage.critical_alerts:
        print("\n🚨 CRITICAL ALERTS:")
        print("-" * 80)
        for alert in triage.critical_alerts:
            print(f"  • {alert}")
    
    if triage.recommendations:
        print("\n📝 RECOMMENDATIONS:")
        print("-" * 80)
        for rec in triage.recommendations:
            print(f"  • {rec}")
    
    print("\n📊 METADATA:")
    print("-" * 80)
    for key, value in triage.metadata.items():
        print(f"  {key:25} {value}")
    
    print("\n" + "=" * 80)
    
    # Cleanup
    rppg_pipeline.cleanup()

if __name__ == "__main__":
    test_fusion_with_respiratory()
