#!/usr/bin/env python3
"""
Test MedGemma integration in the fusion engine
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.fusion_engine import FusionEngine
from src.core.base_pipeline import PipelineResult


def test_medgemma_interpretation():
    """Test MedGemma clinical interpretation with sample vital signs"""
    
    print("\n" + "=" * 80)
    print("Testing MedGemma Integration in Fusion Engine")
    print("=" * 80)
    
    # Initialize fusion engine with MedGemma
    config = {
        'use_medgemma': True,
        'hear_weight': 0.25,
        'rppg_weight': 0.35,
        'vqa_weight': 0.40
    }
    
    print("\nInitializing fusion engine...")
    fusion = FusionEngine(config=config)
    
    if not fusion.use_medgemma:
        print("\n⚠️ MedGemma not available. Install dependencies:")
        print("   pip install transformers torch accelerate")
        return
    
    # Create mock rPPG result with vital signs
    print("\n" + "-" * 80)
    print("Test Case 1: Normal Vital Signs")
    print("-" * 80)
    
    from datetime import datetime
    
    rppg_result = PipelineResult(
        pipeline_name="rPPG",
        timestamp=datetime.now(),
        confidence=0.82,
        data={
            'heart_rate': 72.0,
            'respiratory_rate': 15.5,
            'signal_quality': 0.82,
            'hrv': {
                'sdnn': 45.2,
                'rmssd': 38.1,
                'LF/HF': 1.8
            }
        },
        warnings=[],
        errors=[],
        metadata={'model': 'ME-chunk.rlap'}
    )
    
    # Generate triage report
    report = fusion.fuse(rppg_result=rppg_result)
    
    print(f"\nPriority: {report.priority.value}")
    print(f"Confidence: {report.confidence:.1%}")
    print(f"\nVital Signs:")
    for key, value in report.vital_signs.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.1f}")
    
    print(f"\n{report.summary}")
    
    # Test Case 2: Abnormal vitals
    print("\n\n" + "-" * 80)
    print("Test Case 2: Tachycardia with Tachypnea")
    print("-" * 80)
    
    rppg_result_abnormal = PipelineResult(
        pipeline_name="rPPG",
        timestamp=datetime.now(),
        confidence=0.75,
        data={
            'heart_rate': 125.0,
            'respiratory_rate': 28.0,
            'signal_quality': 0.75
        },
        warnings=[],
        errors=[],
        metadata={'model': 'ME-chunk.rlap'}
    )
    
    report2 = fusion.fuse(rppg_result=rppg_result_abnormal)
    
    print(f"\nPriority: {report2.priority.value}")
    print(f"Confidence: {report2.confidence:.1%}")
    print(f"\nVital Signs:")
    for key, value in report2.vital_signs.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.1f}")
    
    if report2.critical_alerts:
        print(f"\n🚨 Critical Alerts:")
        for alert in report2.critical_alerts:
            print(f"  - {alert}")
    
    print(f"\n{report2.summary}")
    
    # Test Case 3: Bradypnea
    print("\n\n" + "-" * 80)
    print("Test Case 3: Low Respiratory Rate")
    print("-" * 80)
    
    rppg_result_brady = PipelineResult(
        pipeline_name="rPPG",
        timestamp=datetime.now(),
        confidence=0.79,
        data={
            'heart_rate': 64.5,
            'respiratory_rate': 9.2,
            'signal_quality': 0.79
        },
        warnings=[],
        errors=[],
        metadata={'model': 'ME-chunk.rlap'}
    )
    
    report3 = fusion.fuse(rppg_result=rppg_result_brady)
    
    print(f"\nPriority: {report3.priority.value}")
    print(f"Confidence: {report3.confidence:.1%}")
    print(f"\nVital Signs:")
    for key, value in report3.vital_signs.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.1f}")
    
    if report3.critical_alerts:
        print(f"\n⚠️ Alerts:")
        for alert in report3.critical_alerts:
            print(f"  - {alert}")
    
    if report3.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in report3.recommendations:
            print(f"  - {rec}")
    
    print(f"\n{report3.summary}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_medgemma_interpretation()
