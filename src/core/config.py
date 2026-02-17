"""
Configuration management for AI Guardian
"""

import os
from pathlib import Path
from typing import Dict, Any


# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
CONFIGS_DIR = BASE_DIR / "configs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
CONFIGS_DIR.mkdir(exist_ok=True)


class Config:
    """Main configuration class for AI Guardian"""
    
    def __init__(self):
        # HeAR Pipeline Configuration
        self.hear_config = {
            'model_path': MODELS_DIR / 'hear_model.pt',
            'sample_rate': 16000,
            'window_size': 5,  # seconds
            'overlap': 0.5
        }
        
        # rPPG Pipeline Configuration
        self.rppg_config = {
            'model_name': 'ME-chunk.rlap',  # Best performing model (84.5% SQI)
            'model_path': MODELS_DIR / 'rppg_model.pt',
            'fps': 15,  # 15fps sufficient for rPPG (signal <4Hz, Nyquist ≥8fps)
            'window_size': 10,  # seconds
            'roi_method': 'face_detection',
            'signal_processing': {
                'filter_type': 'butterworth',
                'filter_order': 3,
                'lowcut': 0.7,  # Hz
                'highcut': 4.0  # Hz
            }
        }
        
        # MedGemma VQA Configuration
        self.vqa_config = {
            'model_name': 'google/medgemma-vqa',
            'model_path': MODELS_DIR / 'medgemma_vqa',
            'max_questions': 10,
            'confidence_threshold': 0.7
        }

        # VGA (Visual Grading Assessment) Configuration
        self.vga_config = {
            'screenshot_count': 10,
            'model_path': MODELS_DIR / 'vga_model',
            'confidence_threshold': 0.7
        }

        # Cry Pipeline Configuration
        self.cry_config = {
            'backend': 'ast',  # 'ast' or 'hubert'
            'sample_rate': 16000,
            'embedding_dim': 768,
            'model_path': MODELS_DIR / 'cry_classifier'
        }
        
        # Fusion Engine Configuration
        self.fusion_config = {
            'hear_weight': 0.25,
            'rppg_weight': 0.35,
            'vqa_weight': 0.40,
            'min_confidence': 0.6,
            'cross_validation_enabled': True
        }

        # Clinical Guidelines Configuration
        self.guidelines_config = {
            'enabled': True,
            'max_results': 4,
            'max_prompt_tokens': 300,
            'sources': ['nice_ng143', 'who_imci', 'pews'],
        }
        
        # Data processing
        self.data_config = {
            'max_video_duration': 30,  # seconds
            'max_audio_duration': 30,  # seconds
            'image_size': (640, 480),
            'video_format': 'mp4',
            'audio_format': 'wav'
        }
        
        # Logging
        self.logging_config = {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_dir': BASE_DIR / 'logs'
        }
        
        # Ensure log directory exists
        self.logging_config['log_dir'].mkdir(exist_ok=True)
    
    def get_pipeline_config(self, pipeline_name: str) -> Dict[str, Any]:
        """Get configuration for a specific pipeline"""
        config_map = {
            'hear': self.hear_config,
            'rppg': self.rppg_config,
            'medgemma_vqa': self.vqa_config,
            'vqa': self.vqa_config,
            'vga': self.vga_config,
            'cry': self.cry_config,
            'guidelines': self.guidelines_config
        }
        return config_map.get(pipeline_name.lower(), {})
    
    def get_fusion_config(self) -> Dict[str, Any]:
        """Get fusion engine configuration"""
        return self.fusion_config
    
    def update_config(self, section: str, updates: Dict[str, Any]) -> None:
        """Update a configuration section"""
        if hasattr(self, f'{section}_config'):
            config = getattr(self, f'{section}_config')
            config.update(updates)
        else:
            raise ValueError(f"Unknown configuration section: {section}")


# Global configuration instance
config = Config()
