#!/usr/bin/env python
"""
Inspect what the open-rppg model actually returns
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rppg
import json

# Initialize model
model = rppg.Model('PhysNet.pure')

# Process video
video_path = "data/sample/video/sample_video.mp4"
result = model.process_video(video_path)

print("="*70)
print("Raw Model Output Inspection")
print("="*70)

print(f"\nResult type: {type(result)}")
print(f"\nResult keys: {result.keys() if isinstance(result, dict) else 'N/A'}")

print(f"\nFull result:")
for key, value in result.items():
    print(f"  {key}: {value}")
    print(f"    Type: {type(value)}")
    if isinstance(value, dict):
        print(f"    Contents: {value}")

print("\n" + "="*70)
print("Field Explanations:")
print("="*70)

print("""
From the open-rppg model output:

• 'hr' (Heart Rate): The estimated heart rate in BPM
• 'SQI' (Signal Quality Index): Quality score from the rPPG signal extraction
• 'hrv' (Heart Rate Variability): Dictionary of HRV metrics (if computed)
• 'latency': Processing latency in seconds

The 'SQI' is what we use for both:
  - result.data['signal_quality'] 
  - result.confidence

They are currently THE SAME VALUE in our pipeline.
""")

print("="*70)
