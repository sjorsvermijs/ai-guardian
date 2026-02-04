# rPPG Pipeline Testing

## ✅ Implementation Complete!

The rPPG pipeline has been successfully implemented using the `open-rppg` library and is ready to use.

## Quick Test

Run the simplest test:

```bash
python tests/test_rppg_quick.py
```

Expected output:
```
✅ Heart Rate: 84.5 BPM
📊 Quality: 20.7%
⏱️  Processed: 80.1 sec video
```

## Available Test Scripts

### 1. `tests/test_rppg_quick.py` - Minimal Test
Fastest way to verify the pipeline works.

```bash
python tests/test_rppg_quick.py
```

### 2. `tests/pipelines/test_rppg_detailed.py` - Detailed Test
Shows full results with metadata and warnings.

```bash
python tests/pipelines/test_rppg_detailed.py
```

### 3. `tests/pipelines/test_rppg_complete.py` - Complete Test Suite
Tests both video file and numpy array inputs.

```bash
python tests/pipelines/test_rppg_complete.py
```

## What Was Implemented

### Core Pipeline ([src/pipelines/rppg/pipeline.py](src/pipelines/rppg/pipeline.py))

- ✅ Integration with open-rppg library
- ✅ PhysNet.pure model (default)
- ✅ Support for 15+ different rPPG models
- ✅ Video file input support
- ✅ NumPy array input support
- ✅ Automatic video format conversion
- ✅ Signal quality assessment
- ✅ Heart rate extraction
- ✅ BVP signal extraction
- ✅ Comprehensive error handling
- ✅ Resource cleanup

### Features

**Input Types:**
- Video file paths (mp4, avi, etc.)
- NumPy arrays of video frames

**Outputs:**
- Heart rate (BPM)
- Signal quality (0-1)
- Heart rate variability metrics
- BVP signal
- Processing latency
- Video metadata

**Models Supported:**
- PhysNet (pure, rlap)
- EfficientPhys (pure, rlap)
- TSCAN (pure, rlap)
- PhysFormer (pure, rlap)
- PhysMamba (pure, rlap)
- RhythmMamba (pure, rlap)
- FacePhys, ME-chunk, ME-flow

## Usage Example

```python
from src.pipelines.rppg import RPPGPipeline
from src.core.config import config

# Initialize
pipeline = RPPGPipeline(config.get_pipeline_config('rppg'))
pipeline.initialize()

# Process (accepts video path or numpy array)
result = pipeline.process("path/to/video.mp4")

# Get results
print(f"Heart Rate: {result.data['heart_rate']:.1f} BPM")
print(f"Quality: {result.data['signal_quality']:.1%}")

# Cleanup
pipeline.cleanup()
```

## Configuration

Edit [src/core/config.py](src/core/config.py):

```python
rppg_config = {
    'model_name': 'PhysNet.pure',  # Change to any supported model
    'fps': 30,
}
```

## Test Results

Sample video (`data/sample/video/sample_video.mp4`):
- ✅ Heart rate detected: ~84-98 BPM
- ⚠️ Signal quality: 20.7% (low - general video, not optimized for rPPG)
- ⏱️ Duration: 80 seconds
- 📊 Frames: 2401 frames at 29.97 fps

## Notes on Signal Quality

The current sample video has low signal quality (~20%) because:
- It's a general-purpose video, not recorded for rPPG
- Contains non-key frames (2385 of 2403 frames)
- May have varying lighting or motion

For production use:
- Use videos specifically recorded for health monitoring
- Ensure good, consistent lighting
- Minimize subject movement
- Use recommended recording tools like [PhysRecorder](https://github.com/KegangWangCCNU/PhysRecorder)

## Documentation

Full documentation available in [docs/rppg_pipeline_guide.md](docs/rppg_pipeline_guide.md)

## Next Steps

The rPPG pipeline is ready for integration with:
- HeAR pipeline (audio analysis)
- MedGemma VQA pipeline (visual inspection)
- Fusion Engine (combines all three pipelines)

You can now build on this foundation to create the complete AI Guardian system!

---

## Test File Locations

All test files are organized in the `tests/` directory:
- `tests/test_rppg_quick.py` - Quick test
- `tests/pipelines/test_rppg_detailed.py` - Detailed test
- `tests/pipelines/test_rppg_complete.py` - Complete test suite
