# rPPG Pipeline - Usage Guide

## Overview

The rPPG (remote Photoplethysmography) pipeline extracts vital signs from video using the open-rppg library. It analyzes subtle color changes in skin to detect blood volume pulse (BVP) signals and calculate heart rate.

## Features

- ✅ Heart rate extraction from video
- ✅ Signal quality assessment
- ✅ BVP signal extraction
- ✅ Support for video file paths or numpy arrays
- ✅ Multiple model support (PhysNet, EfficientPhys, TSCAN, etc.)

## Installation

The pipeline uses the `open-rppg` library which is already in requirements.txt:

```bash
pip install open-rppg
```

## Quick Start

### Basic Usage with Video File

```python
from src.pipelines.rppg import RPPGPipeline
from src.core.config import config

# Initialize pipeline
pipeline = RPPGPipeline(config.get_pipeline_config('rppg'))
pipeline.initialize()

# Process video
result = pipeline.process("path/to/video.mp4")

# Get results
print(f"Heart Rate: {result.data['heart_rate']:.1f} BPM")
print(f"Signal Quality: {result.data['signal_quality']:.1%}")
print(f"Confidence: {result.confidence:.1%}")

# Cleanup
pipeline.cleanup()
```

### Usage with NumPy Array

```python
import cv2
import numpy as np

# Load video frames
cap = cv2.VideoCapture("video.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

frames_array = np.array(frames)  # Shape: (num_frames, height, width, 3)

# Process with pipeline
pipeline = RPPGPipeline(config.get_pipeline_config('rppg'))
pipeline.initialize()
result = pipeline.process(frames_array)
pipeline.cleanup()
```

## Configuration

Configure the pipeline in `src/core/config.py`:

```python
rppg_config = {
    'model_name': 'PhysNet.pure',  # Model to use
    'fps': 30,                      # Frame rate for numpy array input
}
```

### Available Models

The open-rppg library supports multiple models:

**Deep Learning Models:**

- `PhysNet.pure` - Default, good balance of speed and accuracy
- `PhysNet.rlap` - Alternative PhysNet variant
- `EfficientPhys.pure` - More efficient model
- `EfficientPhys.rlap` - Alternative EfficientPhys
- `TSCAN.pure` - Temporal shift attention network
- `TSCAN.rlap` - Alternative TSCAN
- `PhysFormer.pure` - Transformer-based model
- `PhysFormer.rlap` - Alternative PhysFormer
- `PhysMamba.pure` - State space model
- `PhysMamba.rlap` - Alternative PhysMamba
- `RhythmMamba.pure` - Rhythm-focused model
- `RhythmMamba.rlap` - Alternative RhythmMamba
- `FacePhys.rlap` - Face-optimized model
- `ME-chunk.rlap` - Motion-enhanced chunked
- `ME-flow.rlap` - Motion-enhanced flow

To use a different model:

```python
config.rppg_config['model_name'] = 'EfficientPhys.pure'
```

## Pipeline Result

The pipeline returns a `PipelineResult` object with:

### `result.data` Dictionary:

- `heart_rate` (float): Estimated heart rate in BPM
- `signal_quality` (float): Quality score 0-1
- `heart_rate_variability` (dict): HRV metrics if available
- `latency_ms` (float): Processing latency in milliseconds
- `bvp_signal_length` (int): Number of BVP samples extracted

### `result.confidence` (float):

Overall confidence score (0-1), based on signal quality

### `result.warnings` (list):

- Signal quality warnings
- Abnormal heart rate warnings

### `result.metadata` (dict):

- `model_name`: Model used for processing
- `fps`: Video frame rate
- `num_frames`: Total frames processed
- `duration_seconds`: Video duration

## Understanding Signal Quality

The signal quality indicator (SQI) tells you how reliable the results are:

- **> 0.7**: High quality - Results are reliable
- **0.5 - 0.7**: Medium quality - Results should be interpreted with caution
- **0.3 - 0.5**: Low quality - Results may be unreliable
- **< 0.3**: Very low quality - Results are unreliable

### Factors Affecting Signal Quality:

1. **Face Visibility**: Subject's face must be clearly visible
2. **Lighting**: Consistent, good lighting is essential
3. **Motion**: Minimal head movement improves accuracy
4. **Video Quality**: Higher resolution and frame rate help
5. **Skin Tone**: Algorithm works across skin tones but lighting matters

### Improving Signal Quality:

- Record with stable lighting (avoid flickering lights)
- Minimize subject movement
- Ensure face is well-lit and clearly visible
- Use higher frame rates (30fps minimum)
- Record in uncompressed or high-quality formats

## Testing

Run the test scripts:

```bash
# Simple test
python test_rppg_pipeline.py

# Complete test suite (both file and array inputs)
python test_rppg_complete.py
```

## Example Output

```
Pipeline: rPPG
Timestamp: 2026-02-04 20:03:20
Confidence: 85.3%

✅ Processing successful!

💓 Vital Signs:
   Heart Rate:     72.5 BPM
   Signal Quality: 85.3%
   Latency:        0.0 ms
   BVP Samples:    2403

📈 Video Info:
   Model:    PhysNet.pure
   FPS:      29.97
   Frames:   2401
   Duration: 80.1 sec
```

## Troubleshooting

### "Very low signal quality"

- Check that face is clearly visible in the video
- Ensure consistent lighting
- Verify minimal movement
- Try recording with recommended settings

### ImportError: No module named 'rppg'

```bash
pip install open-rppg
```

### Model not found error

- Check that model name is in the supported models list
- Use `rppg.supported_models` to see available models

### High heart rate values (> 150 BPM)

- Often caused by poor signal quality
- Check signal_quality score
- Verify video quality and recording conditions

## References

- [open-rppg GitHub](https://github.com/remotebiosensing/rppg)
- [rPPG Research Paper](https://arxiv.org/abs/2307.12644)
- [PhysRecorder](https://github.com/KegangWangCCNU/PhysRecorder) - Recommended for recording videos

## Notes

- The pipeline automatically handles video file to numpy array conversion
- Temporary files are created and cleaned up automatically
- For best results, use videos specifically recorded for rPPG analysis
- The open-rppg library will warn about non-key frames in standard videos
