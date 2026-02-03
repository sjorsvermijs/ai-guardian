# AI Guardian

A comprehensive medical triage system that combines three AI pipelines for high-confidence health assessment:

- **HeAR (Health Acoustic Representations)**: Analyzes audio for respiratory patterns and acoustic health indicators
- **rPPG (remote photoplethysmography)**: Extracts vital signs from video using subtle skin color changes
- **MedGemma VQA (Visual Question Answering)**: Visually inspects images for critical medical red flags

By fusing these three modalities, AI Guardian provides medical reasoning that goes far beyond what any single sensor could achieve alone.

## 🏗️ Project Structure

```
ai-guardian/
├── src/
│   ├── pipelines/
│   │   ├── hear/              # HeAR audio analysis pipeline
│   │   │   ├── __init__.py
│   │   │   └── pipeline.py
│   │   ├── rppg/              # rPPG vital signs pipeline
│   │   │   ├── __init__.py
│   │   │   └── pipeline.py
│   │   └── medgemma_vqa/      # MedGemma VQA visual inspection
│   │       ├── __init__.py
│   │       └── pipeline.py
│   ├── core/                  # Core system components
│   │   ├── __init__.py
│   │   ├── base_pipeline.py   # Base pipeline interface
│   │   ├── fusion_engine.py   # Multi-modal fusion logic
│   │   └── config.py          # Configuration management
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       └── preprocessing.py   # Data preprocessing
├── tests/                     # Unit and integration tests
├── configs/                   # Configuration files
├── data/                      # Sample data (not in repo)
├── models/                    # Model weights (not in repo)
├── main.py                    # Main application entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd ai-guardian

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

```bash
# TODO: Add instructions for downloading pretrained models
# - HeAR model weights
# - rPPG model weights
# - MedGemma VQA model
```

### 3. Run the Application

```bash
python main.py
```

## 📋 Pipeline Details

### HeAR Pipeline
Analyzes audio recordings to detect:
- Respiratory rate and patterns
- Coughing episodes
- Breathing abnormalities
- Acoustic distress signals

**Input**: Audio waveform (NumPy array)  
**Output**: Acoustic health indicators with confidence scores

### rPPG Pipeline
Extracts vital signs from video:
- Heart rate (BPM)
- Oxygen saturation (SpO2)
- Respiratory rate
- Heart rate variability

**Input**: Video frames (NumPy array)  
**Output**: Vital signs measurements with quality metrics

### MedGemma VQA Pipeline
Visual inspection for critical indicators:
- Cyanosis (bluish discoloration)
- Nasal flaring
- Chest retractions
- Respiratory distress signs
- Skin color abnormalities

**Input**: Image (NumPy array)  
**Output**: VQA answers with critical flags

## 🔄 Fusion Engine

The Fusion Engine combines results from all three pipelines:

1. **Weighted Aggregation**: Combines confidence scores using configurable weights
2. **Cross-Validation**: Validates findings across modalities (e.g., cyanosis + low SpO2)
3. **Priority Assignment**: Determines triage priority (CRITICAL, URGENT, MODERATE, LOW)
4. **Report Generation**: Creates comprehensive triage report with recommendations

## ⚙️ Configuration

Edit pipeline configurations in [src/core/config.py](src/core/config.py):

```python
# Example: Adjust fusion weights
config.fusion_config = {
    'hear_weight': 0.25,
    'rppg_weight': 0.35,
    'vqa_weight': 0.40
}
```

## 🧪 Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
flake8 src/ tests/
```

## 🎯 Usage Example

```python
from src.pipelines.hear import HeARPipeline
from src.pipelines.rppg import RPPGPipeline
from src.pipelines.medgemma_vqa import MedGemmaVQAPipeline
from src.core.fusion_engine import FusionEngine
from src.core.config import config

# Initialize pipelines
hear = HeARPipeline(config.get_pipeline_config('hear'))
rppg = RPPGPipeline(config.get_pipeline_config('rppg'))
vqa = MedGemmaVQAPipeline(config.get_pipeline_config('vqa'))

hear.initialize()
rppg.initialize()
vqa.initialize()

# Process data
hear_result = hear.process(audio_data)
rppg_result = rppg.process(video_frames)
vqa_result = vqa.process(image)

# Fuse results
fusion = FusionEngine(config.get_fusion_config())
triage_report = fusion.fuse(hear_result, rppg_result, vqa_result)

print(f"Priority: {triage_report.priority.value}")
print(f"Confidence: {triage_report.confidence:.2%}")
```

## 📊 Triage Priority Levels

- **CRITICAL**: Immediate medical intervention required (call 911)
- **URGENT**: Needs prompt attention within 1-2 hours
- **MODERATE**: Schedule appointment with healthcare provider
- **LOW**: Continue routine monitoring

## 🔐 Privacy & Security

This system processes sensitive medical data. Ensure:
- All data is handled in compliance with HIPAA/local regulations
- Models are run locally or in secure environments
- Patient data is encrypted at rest and in transit
- Proper consent is obtained before processing

## ⚠️ Disclaimer

**This is a research prototype and NOT approved for clinical use.**  
Always consult qualified healthcare professionals for medical decisions.

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📧 Contact

[Add contact information here]
