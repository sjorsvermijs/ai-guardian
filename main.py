"""
Main application entry point
Demonstrates how to use the AI Guardian system
"""

import numpy as np
from pathlib import Path

from src.pipelines.hear import HeARPipeline
from src.pipelines.rppg import RPPGPipeline
from src.pipelines.medgemma_vqa import MedGemmaVQAPipeline
from src.core.fusion_engine import FusionEngine
from src.core.config import config
from src.utils.preprocessing import (
    preprocess_audio,
    preprocess_video,
    preprocess_image
)


def main():
    """Main application workflow"""
    
    print("="*60)
    print("AI Guardian - Medical Triage System")
    print("Initializing pipelines...")
    print("="*60)
    
    # Initialize pipelines
    hear_pipeline = HeARPipeline(config.get_pipeline_config('hear'))
    rppg_pipeline = RPPGPipeline(config.get_pipeline_config('rppg'))
    vqa_pipeline = MedGemmaVQAPipeline(config.get_pipeline_config('vqa'))
    
    # Initialize fusion engine
    fusion_engine = FusionEngine(config.get_fusion_config())
    
    # Initialize all pipelines
    print("\n1. Initializing HeAR Pipeline...")
    hear_pipeline.initialize()
    
    print("2. Initializing rPPG Pipeline...")
    rppg_pipeline.initialize()
    
    print("3. Initializing MedGemma VQA Pipeline...")
    vqa_pipeline.initialize()
    
    print("\n" + "="*60)
    print("All pipelines initialized successfully!")
    print("="*60)
    
    # Example: Process dummy data
    print("\nProcessing sample data...")
    
    # TODO: Replace with actual data loading
    # For now, using dummy data as placeholders
    
    # 1. Process audio data (HeAR)
    print("\n[1/3] Processing audio data...")
    dummy_audio = np.random.randn(16000 * 5)  # 5 seconds at 16kHz
    audio_preprocessed = preprocess_audio(dummy_audio)
    hear_result = hear_pipeline.process(audio_preprocessed)
    print(f"   ✓ HeAR confidence: {hear_result.confidence:.2f}")
    
    # 2. Process video data (rPPG)
    print("\n[2/3] Processing video data...")
    dummy_video = np.random.randint(0, 255, (300, 480, 640, 3), dtype=np.uint8)  # 10s at 30fps
    video_preprocessed = preprocess_video(dummy_video)
    rppg_result = rppg_pipeline.process(video_preprocessed)
    print(f"   ✓ rPPG confidence: {rppg_result.confidence:.2f}")
    
    # 3. Process image data (VQA)
    print("\n[3/3] Processing image data...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image_preprocessed = preprocess_image(dummy_image)
    vqa_result = vqa_pipeline.process(image_preprocessed)
    print(f"   ✓ VQA confidence: {vqa_result.confidence:.2f}")
    
    # Fuse results
    print("\n" + "="*60)
    print("Fusing pipeline results...")
    print("="*60)
    
    triage_report = fusion_engine.fuse(
        hear_result=hear_result,
        rppg_result=rppg_result,
        vqa_result=vqa_result
    )
    
    # Display results
    print("\n" + "="*60)
    print("TRIAGE REPORT")
    print("="*60)
    print(f"\nTimestamp: {triage_report.timestamp}")
    print(f"Priority: {triage_report.priority.value}")
    print(f"Overall Confidence: {triage_report.confidence:.2%}")
    print(f"\nSummary: {triage_report.summary}")
    
    if triage_report.critical_alerts:
        print(f"\n⚠️  Critical Alerts ({len(triage_report.critical_alerts)}):")
        for alert in triage_report.critical_alerts:
            print(f"   • {alert}")
    
    print(f"\nRecommendations:")
    for rec in triage_report.recommendations:
        print(f"   • {rec}")
    
    print("\n" + "="*60)
    
    # Cleanup
    print("\nCleaning up resources...")
    hear_pipeline.cleanup()
    rppg_pipeline.cleanup()
    vqa_pipeline.cleanup()
    
    print("✓ Done!")


if __name__ == "__main__":
    main()
