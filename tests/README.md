# rPPG Pipeline Tests

This directory contains tests for the rPPG (remote photoplethysmography) pipeline.

## Test Files

### Quick Test

- **`test_rppg_quick.py`** - Minimal test for quick verification
  ```bash
  python tests/test_rppg_quick.py
  ```

### Detailed Tests (in `pipelines/` subdirectory)

- **`pipelines/test_rppg_detailed.py`** - Detailed test with full output

  ```bash
  python tests/pipelines/test_rppg_detailed.py
  ```

- **`pipelines/test_rppg_complete.py`** - Complete test suite (file + array inputs)
  ```bash
  python tests/pipelines/test_rppg_complete.py
  ```

### Unit Tests

- **`test_hear_pipeline.py`** - Unit tests for HeAR pipeline
- **`test_fusion_engine.py`** - Unit tests for Fusion Engine

## Running Tests

All tests can be run from the project root:

```bash
# Quick test
python tests/test_rppg_quick.py

# Run with pytest
pytest tests/

# Run specific test file
pytest tests/test_fusion_engine.py
```

## Test Requirements

- Sample video file at: `data/sample/video/sample_video.mp4`
- All dependencies installed: `pip install -r requirements.txt`
- Virtual environment activated

## Expected Results

The rPPG tests should produce:

- Heart rate measurements (70-100 BPM typical)
- Signal quality scores (0-1 range)
- Processing metadata (FPS, frame count, duration)

Note: Signal quality may be low (~20%) for general videos not optimized for rPPG analysis.
