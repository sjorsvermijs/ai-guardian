#!/usr/bin/env python3
"""
Simple Webcam rPPG Monitor - Direct OpenCV control for better responsiveness
"""
import os
import sys
import warnings

# Set environment for macOS
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

# Suppress warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.pipelines.rppg.pipeline import RPPGPipeline
from src.core.config import config

def main():
    """Simple webcam monitoring with direct OpenCV control."""
    print("=" * 80)
    print("🎥 AI GUARDIAN - SIMPLE WEBCAM MONITOR")
    print("=" * 80)
    print()
    print("This tool captures video then processes it for vital signs.")
    print("More responsive than live preview mode.")
    print()
    
    duration = float(input("Capture duration in seconds (default 30): ") or "30")
    camera_index = int(input("Camera index (default 0): ") or "0")
    
    print(f"\n{'=' * 80}")
    print("Opening camera...")
    
    # Open camera directly
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Failed to open camera!")
        print("Try running: python test_camera.py")
        return
    
    # Get camera properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✓ Camera opened: {width}x{height} @ {fps:.1f}fps")
    print(f"\nCapturing {duration}s of video...")
    print("Press ESC to stop early, or wait for countdown...")
    print()
    
    frames = []
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    frame_count = 0
    
    # Capture loop with preview
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame!")
            break
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        elapsed = current_time - start_time
        
        if elapsed >= duration:
            break
        
        # Store frame for processing
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
        
        # Show preview with countdown
        display = frame.copy()
        remaining = duration - elapsed
        
        # Draw info
        cv2.putText(display, f"Capturing: {remaining:.1f}s remaining", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Frames: {frame_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "Press ESC to stop", 
                   (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1)
        
        cv2.imshow("Capturing Video", display)
        
        # Check for ESC key - this is responsive!
        if cv2.waitKey(1) & 0xFF == 27:
            print("\nStopped by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(frames) == 0:
        print("❌ No frames captured!")
        return
    
    print(f"\n✓ Captured {len(frames)} frames ({len(frames)/fps:.1f}s of video)")
    print("\nInitializing rPPG pipeline...")
    
    # Process with rPPG pipeline
    pipeline = RPPGPipeline(config.rppg_config)
    pipeline.initialize()
    
    print("Processing video...")
    frames_array = np.array(frames, dtype=np.uint8)
    result = pipeline.process(frames_array)
    
    # Display results
    print("\n" + "=" * 80)
    print("💓 RESULTS")
    print("=" * 80)
    
    hr = result.data.get('heart_rate')
    rr = result.data.get('respiratory_rate')
    sqi = result.data.get('signal_quality', 0)
    
    print(f"\n📊 Vital Signs:")
    if hr:
        hr_status = "✅" if 60 <= hr <= 100 else "⚠️"
        print(f"  {hr_status} Heart Rate:       {hr:.1f} BPM")
    
    if rr:
        rr_status = "✅" if 12 <= rr <= 20 else "⚠️"
        print(f"  {rr_status} Respiratory Rate: {rr:.1f} breaths/min")
    else:
        print(f"  ⚠️  Respiratory Rate: Not available (SQI too low)")
    
    sqi_status = "✅" if sqi > 0.5 else ("⚠️" if sqi > 0.3 else "❌")
    print(f"  {sqi_status} Signal Quality:   {sqi:.1%}")
    print(f"  📊 Confidence:       {result.confidence:.1%}")
    
    if result.warnings:
        print(f"\n⚠️  Warnings:")
        for warning in result.warnings:
            print(f"  • {warning}")
    
    # HRV if available
    hrv = result.data.get('heart_rate_variability', {})
    if hrv and 'LF/HF' in hrv:
        print(f"\n💓 HRV Metrics:")
        print(f"  SDNN:      {hrv.get('sdnn', 0):.1f} ms")
        print(f"  RMSSD:     {hrv.get('rmssd', 0):.1f} ms")
        print(f"  LF/HF:     {hrv.get('LF/HF', 0):.2f}")
        
        lf_hf = hrv.get('LF/HF', 0)
        if lf_hf > 2.5:
            print(f"  Balance:   Sympathetic dominant (stressed)")
        elif lf_hf < 1.5:
            print(f"  Balance:   Parasympathetic dominant (relaxed)")
        else:
            print(f"  Balance:   Balanced")
    
    print("\n" + "=" * 80)
    
    pipeline.cleanup()
    
    # Ask to repeat
    repeat = input("\nRun another measurement? (y/n): ").lower()
    if repeat == 'y':
        main()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        cv2.destroyAllWindows()
        sys.exit(0)
