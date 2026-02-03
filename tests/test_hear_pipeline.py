"""
Example test file for HeAR Pipeline
"""

import pytest
import numpy as np
from src.pipelines.hear import HeARPipeline
from src.core.config import config


class TestHeARPipeline:
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing"""
        return HeARPipeline(config.get_pipeline_config('hear'))
    
    def test_initialization(self, pipeline):
        """Test pipeline initialization"""
        assert pipeline.initialize() == True
        assert pipeline.is_initialized == True
    
    def test_valid_audio_input(self, pipeline):
        """Test processing with valid audio input"""
        pipeline.initialize()
        
        # Create dummy audio data (5 seconds at 16kHz)
        audio_data = np.random.randn(16000 * 5).astype(np.float32)
        
        result = pipeline.process(audio_data)
        
        assert result.pipeline_name == "HeAR"
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.errors) == 0
    
    def test_invalid_audio_input(self, pipeline):
        """Test processing with invalid input"""
        pipeline.initialize()
        
        # Invalid input (not numpy array)
        invalid_input = [1, 2, 3, 4, 5]
        
        result = pipeline.process(invalid_input)
        
        assert result.confidence == 0.0
        assert len(result.errors) > 0
    
    def test_cleanup(self, pipeline):
        """Test resource cleanup"""
        pipeline.initialize()
        pipeline.cleanup()
        
        assert pipeline.is_initialized == False
