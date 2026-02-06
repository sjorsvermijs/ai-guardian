# Webcam rPPG Monitoring

AI Guardian can measure vital signs in real-time using your laptop's webcam through remote photoplethysmography (rPPG).

## Overview

The rPPG pipeline detects subtle color changes in facial skin caused by blood flow, enabling contactless measurement of:

- **Heart Rate** (BPM)
- **Respiratory Rate** (breaths/min) - when signal quality > 50%
- **Heart Rate Variability (HRV)** - SDNN, RMSSD, LF/HF ratio
- **Signal Quality Index (SQI)** - reliability indicator (0-100%)

## Quick Start

### Basic Webcam Monitoring

```bash
python webcam_monitor.py
```

This script:

1. Captures video for a specified duration (default: 30s)
2. Processes the video using the ME-chunk.rlap model
3. Displays vital signs results

**Usage:**

```
Enter capture duration in seconds (recommended: 30-60): 30
Enter camera index (0 for default, 1 for external): 0
```

### Live Preview Monitor

```bash
python webcam_monitor_live.py
```

Features:

- Real-time video preview with face detection box
- Live heart rate and respiratory rate overlay
- Signal quality indicator
- Press 'q' to quit, 's' for snapshot measurement

## Programmatic Usage

### Simple Webcam Capture

```python
from src.pipelines.rppg.pipeline import RPPGPipeline
from src.core.config import config

# Initialize pipeline
pipeline = RPPGPipeline(config.rppg_config)
pipeline.initialize()

# Capture 30 seconds from default webcam
result = pipeline.process_webcam(camera_index=0, duration=30)

# Access results
print(f"Heart Rate: {result.data['heart_rate']:.1f} BPM")
print(f"Respiratory Rate: {result.data['respiratory_rate']:.1f} breaths/min")
print(f"Signal Quality: {result.data['signal_quality']:.1%}")

# Cleanup
pipeline.cleanup()
```

### Custom Live Monitoring

```python
import cv2
import time

# Initialize pipeline
pipeline = RPPGPipeline(config.rppg_config)
pipeline.initialize()

# Start live monitoring
with pipeline.model.video_capture(0):  # 0 = default camera
    last_update = 0

    for frame, box in pipeline.model.preview:
        # Update metrics every 2 seconds
        if time.time() - last_update > 2.0:
            result = pipeline.model.hr(start=-30)  # Last 30 seconds

            if result and result['hr']:
                hr = result['hr']
                sqi = result['SQI']
                print(f"HR: {hr:.1f} BPM, SQI: {sqi:.1%}")

            last_update = time.time()

        # Display frame (convert RGB to BGR)
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw face box if detected
        if box is not None:
            y1, y2 = box[0]
            x1, x2 = box[1]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Monitor", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
pipeline.cleanup()
```

## Tips for Best Results

### Environment Setup

- **Lighting**: Natural daylight is best; avoid harsh overhead lights or backlighting
- **Position**: Face camera directly, 30-60cm away
- **Stability**: Rest elbows on desk, minimize head movement
- **Background**: Avoid busy or bright backgrounds

### Camera Settings

- Use highest resolution available (720p or 1080p)
- Disable auto-exposure if possible for consistent lighting
- Ensure webcam is clean and focused

### Signal Quality Optimization

| SQI Range | Quality   | Recommendations                                |
| --------- | --------- | ---------------------------------------------- |
| 0-30%     | Poor      | Check lighting, face visibility, reduce motion |
| 30-50%    | Fair      | Improve lighting or camera position            |
| 50-70%    | Good      | Results are reliable                           |
| 70%+      | Excellent | Optimal conditions, HRV available              |

### Duration Recommendations

- **Minimum**: 10 seconds (basic HR only)
- **Recommended**: 30-60 seconds (HR + RR with good accuracy)
- **Optimal**: 60-120 seconds (stable HRV metrics)

## Model Performance

The default **ME-chunk.rlap** model achieves:

- **84.5% SQI** on high-quality webcam video
- **4.5 seconds** processing time for ~80 second video
- **Heart rate accuracy**: ±2 BPM in good conditions
- **Respiratory rate**: Available when SQI > 50%

## Troubleshooting

### "No heart rate detected"

- Ensure face is fully visible and well-lit
- Remove glasses if possible
- Increase capture duration to 60+ seconds
- Try different lighting conditions

### Low Signal Quality (< 30%)

- Move to area with better lighting
- Clean webcam lens
- Reduce head/body movement
- Ensure no heavy makeup or face covering

### Face Not Detected

- Center face in frame
- Move closer to camera (but not too close)
- Remove obstructions (hands, hair, mask)
- Ensure adequate contrast with background

### Webcam Access Error

```bash
# macOS: Grant camera permission
System Preferences → Security & Privacy → Camera → Enable for Terminal/IDE

# Or run with environment variable (scripts do this automatically):
export OPENCV_AVFOUNDATION_SKIP_AUTH=1

# If still having issues, try granting camera access to Python:
System Preferences → Security & Privacy → Camera → Add Python

# Linux: Check camera permissions
ls -l /dev/video*
sudo usermod -a -G video $USER
```

**Note**: The webcam monitor scripts automatically set `OPENCV_AVFOUNDATION_SKIP_AUTH=1` for macOS compatibility.

## Clinical Interpretation

### Heart Rate

- **40-60 BPM**: Bradycardia (athletic or concerning)
- **60-100 BPM**: Normal resting heart rate
- **100-120 BPM**: Mild tachycardia
- **> 120 BPM**: Tachycardia (exercise, stress, or medical)

### Respiratory Rate

- **< 12 br/min**: Bradypnea (slow breathing)
- **12-20 br/min**: Normal adult respiratory rate
- **20-30 br/min**: Tachypnea (fast breathing)
- **> 30 br/min**: Severe tachypnea (medical attention needed)

### HRV - LF/HF Ratio

- **< 1.5**: Parasympathetic dominant (relaxed, recovered)
- **1.5-2.5**: Balanced autonomic function
- **> 2.5**: Sympathetic dominant (stressed, fatigued)

## Privacy & Security

All processing happens **locally on your device**:

- ✅ No video uploaded to cloud
- ✅ No data leaves your computer
- ✅ Temporary files automatically deleted
- ✅ Open-source, auditable code

## Limitations

- Not a medical device - for wellness monitoring only
- Requires good lighting and stable positioning
- Less accurate than clinical-grade monitors
- Cannot measure blood pressure or SpO2 directly
- Respiratory rate requires high signal quality (SQI > 50%)

## References

The webcam monitoring uses the open-rppg library with ME-chunk.rlap model:

- Wang et al., "Memory-efficient Low-latency rPPG through Temporal-Spatial State Space Duality" (2025)
- Achieves state-of-the-art performance with 4.5s processing time

For more information: https://github.com/KegangWangCCNU/open-rppg
