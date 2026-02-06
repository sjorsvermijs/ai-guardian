#!/usr/bin/env python3
"""
Test Fusion Engine with Multi-Modal Data
Tests rPPG + HeAR + VQA integration with MedGemma interpretation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.fusion_engine import FusionEngine
from src.core.base_pipeline import PipelineResult
from datetime import datetime


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_report(report):
    """Print formatted triage report"""
    print(f"\n🚦 Triage Priority: {report.priority.value}")
    print(f"📊 Confidence: {report.confidence:.1%}")
    
    if report.vital_signs:
        print(f"\n💓 Vital Signs:")
        for key, value in report.vital_signs.items():
            if isinstance(value, (int, float)) and key != 'signal_quality':
                print(f"  • {key.replace('_', ' ').title()}: {value:.1f}")
    
    if report.acoustic_indicators:
        print(f"\n🎤 Acoustic Analysis:")
        for key, value in report.acoustic_indicators.items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    if report.visual_indicators:
        print(f"\n👁️ Visual Observations:")
        for key, value in report.visual_indicators.items():
            if isinstance(value, list):
                print(f"  • {key.replace('_', ' ').title()}:")
                for item in value:
                    print(f"    - {item}")
            else:
                print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    if report.critical_alerts:
        print(f"\n🚨 Critical Alerts:")
        for alert in report.critical_alerts:
            print(f"  • {alert}")
    
    if report.recommendations:
        print(f"\n💡 Recommendations:")
        for rec in report.recommendations:
            print(f"  • {rec}")
    
    # Extract MedGemma interpretation
    if "--- MedGemma Clinical Interpretation ---" in report.summary:
        medgemma_part = report.summary.split("--- MedGemma Clinical Interpretation ---")[1]
        print(f"\n🤖 MedGemma Clinical Interpretation:")
        print(medgemma_part.strip())
    else:
        print(f"\n📋 Summary:")
        print(f"  {report.summary}")


def test_case_1_healthy():
    """Test Case 1: Healthy individual - all pipelines agree"""
    print_section("TEST CASE 1: Healthy Individual")
    
    rppg_result = PipelineResult(
        pipeline_name="rPPG",
        timestamp=datetime.now(),
        confidence=0.85,
        data={
            'heart_rate': 72.0,
            'respiratory_rate': 14.5,
            'signal_quality': 0.85,
            'hrv': {
                'sdnn': 48.2,
                'rmssd': 42.1,
                'LF/HF': 1.5
            }
        },
        warnings=[],
        errors=[],
        metadata={'model': 'ME-chunk.rlap', 'duration': 15}
    )
    
    hear_result = PipelineResult(
        pipeline_name="HeAR",
        timestamp=datetime.now(),
        confidence=0.80,
        data={
            'cough_detected': False,
            'breathing_pattern': 'normal',
            'respiratory_sounds': 'clear'
        },
        warnings=[],
        errors=[],
        metadata={'model': 'HeAR-base'}
    )
    
    vqa_result = PipelineResult(
        pipeline_name="MedGemma-VQA",
        timestamp=datetime.now(),
        confidence=0.75,
        data={
            'skin_color': 'normal',
            'breathing_effort': 'normal',
            'critical_flags': []
        },
        warnings=[],
        errors=[],
        metadata={'model': 'medgemma-vqa'}
    )
    
    return rppg_result, hear_result, vqa_result


def test_case_2_respiratory_distress():
    """Test Case 2: Respiratory distress - multiple modalities confirm"""
    print_section("TEST CASE 2: Respiratory Distress (Multi-Modal)")
    
    rppg_result = PipelineResult(
        pipeline_name="rPPG",
        timestamp=datetime.now(),
        confidence=0.78,
        data={
            'heart_rate': 112.0,
            'respiratory_rate': 28.5,
            'signal_quality': 0.78,
            'spo2': 91.0
        },
        warnings=['Elevated heart rate', 'Tachypnea detected'],
        errors=[],
        metadata={'model': 'ME-chunk.rlap', 'duration': 15}
    )
    
    hear_result = PipelineResult(
        pipeline_name="HeAR",
        timestamp=datetime.now(),
        confidence=0.82,
        data={
            'cough_detected': True,
            'breathing_pattern': 'labored',
            'respiratory_sounds': 'wheezing',
            'cough_frequency': 'high'
        },
        warnings=['Abnormal respiratory sounds'],
        errors=[],
        metadata={'model': 'HeAR-base'}
    )
    
    vqa_result = PipelineResult(
        pipeline_name="MedGemma-VQA",
        timestamp=datetime.now(),
        confidence=0.85,
        data={
            'skin_color': 'normal',
            'breathing_effort': 'increased',
            'nasal_flaring': True,
            'critical_flags': ['Increased work of breathing', 'Nasal flaring observed']
        },
        warnings=['Visual signs of respiratory distress'],
        errors=[],
        metadata={'model': 'medgemma-vqa'}
    )
    
    return rppg_result, hear_result, vqa_result


def test_case_3_hypoxemia():
    """Test Case 3: Severe hypoxemia - critical multi-modal findings"""
    print_section("TEST CASE 3: Severe Hypoxemia (Critical)")
    
    rppg_result = PipelineResult(
        pipeline_name="rPPG",
        timestamp=datetime.now(),
        confidence=0.70,
        data={
            'heart_rate': 135.0,
            'respiratory_rate': 32.0,
            'signal_quality': 0.70,
            'spo2': 86.0
        },
        warnings=['Critical: Low oxygen saturation', 'Severe tachycardia', 'Severe tachypnea'],
        errors=[],
        metadata={'model': 'ME-chunk.rlap', 'duration': 15}
    )
    
    hear_result = PipelineResult(
        pipeline_name="HeAR",
        timestamp=datetime.now(),
        confidence=0.75,
        data={
            'cough_detected': True,
            'breathing_pattern': 'rapid_shallow',
            'respiratory_sounds': 'decreased',
            'vocal_quality': 'weak'
        },
        warnings=['Critically abnormal breathing pattern'],
        errors=[],
        metadata={'model': 'HeAR-base'}
    )
    
    vqa_result = PipelineResult(
        pipeline_name="MedGemma-VQA",
        timestamp=datetime.now(),
        confidence=0.88,
        data={
            'skin_color': 'cyanotic',
            'breathing_effort': 'severe',
            'nasal_flaring': True,
            'accessory_muscle_use': True,
            'critical_flags': [
                'Cyanosis detected (bluish skin discoloration)',
                'Severe respiratory distress',
                'Accessory muscle use observed'
            ]
        },
        warnings=['CRITICAL: Visual signs of severe hypoxemia'],
        errors=[],
        metadata={'model': 'medgemma-vqa'}
    )
    
    return rppg_result, hear_result, vqa_result


def test_case_4_bradypnea():
    """Test Case 4: Severe bradypnea - potential respiratory failure"""
    print_section("TEST CASE 4: Severe Bradypnea")
    
    rppg_result = PipelineResult(
        pipeline_name="rPPG",
        timestamp=datetime.now(),
        confidence=0.82,
        data={
            'heart_rate': 58.0,
            'respiratory_rate': 7.5,
            'signal_quality': 0.82
        },
        warnings=['Severe bradypnea - risk of respiratory failure'],
        errors=[],
        metadata={'model': 'ME-chunk.rlap', 'duration': 15}
    )
    
    hear_result = PipelineResult(
        pipeline_name="HeAR",
        timestamp=datetime.now(),
        confidence=0.68,
        data={
            'cough_detected': False,
            'breathing_pattern': 'very_slow',
            'respiratory_sounds': 'diminished'
        },
        warnings=['Critically slow breathing detected'],
        errors=[],
        metadata={'model': 'HeAR-base'}
    )
    
    vqa_result = PipelineResult(
        pipeline_name="MedGemma-VQA",
        timestamp=datetime.now(),
        confidence=0.72,
        data={
            'skin_color': 'pale',
            'breathing_effort': 'minimal',
            'level_of_consciousness': 'decreased',
            'critical_flags': ['Decreased level of consciousness', 'Minimal respiratory effort']
        },
        warnings=['Altered mental status'],
        errors=[],
        metadata={'model': 'medgemma-vqa'}
    )
    
    return rppg_result, hear_result, vqa_result


def main():
    print("\n" + "=" * 80)
    print("MULTI-MODAL FUSION ENGINE TEST with MedGemma AI")
    print("=" * 80)
    print("\nInitializing fusion engine with MedGemma...")
    
    fusion = FusionEngine(config={'use_medgemma': True})
    
    if not fusion.use_medgemma:
        print("\n⚠️ MedGemma not available - running without AI interpretation")
    
    # Test Case 1: Healthy
    rppg, hear, vqa = test_case_1_healthy()
    report = fusion.fuse(rppg_result=rppg, hear_result=hear, vqa_result=vqa)
    print_report(report)
    
    # Test Case 2: Respiratory Distress
    rppg, hear, vqa = test_case_2_respiratory_distress()
    report = fusion.fuse(rppg_result=rppg, hear_result=hear, vqa_result=vqa)
    print_report(report)
    
    # Test Case 3: Severe Hypoxemia
    rppg, hear, vqa = test_case_3_hypoxemia()
    report = fusion.fuse(rppg_result=rppg, hear_result=hear, vqa_result=vqa)
    print_report(report)
    
    # Test Case 4: Bradypnea
    rppg, hear, vqa = test_case_4_bradypnea()
    report = fusion.fuse(rppg_result=rppg, hear_result=hear, vqa_result=vqa)
    print_report(report)
    
    print("\n" + "=" * 80)
    print("MULTI-MODAL TESTING COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
