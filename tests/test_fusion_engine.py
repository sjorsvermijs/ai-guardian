"""
Example test file for Fusion Engine
"""

import pytest
from datetime import datetime
from src.core.fusion_engine import FusionEngine, TriagePriority
from src.core.base_pipeline import PipelineResult


class TestFusionEngine:
    
    @pytest.fixture
    def fusion_engine(self):
        """Create fusion engine instance"""
        config = {
            'hear_weight': 0.25,
            'rppg_weight': 0.35,
            'vqa_weight': 0.40
        }
        return FusionEngine(config)
    
    @pytest.fixture
    def sample_results(self):
        """Create sample pipeline results"""
        hear_result = PipelineResult(
            pipeline_name="HeAR",
            timestamp=datetime.now(),
            confidence=0.85,
            data={'respiratory_rate': 18, 'cough_detected': False},
            warnings=[],
            errors=[],
            metadata={}
        )
        
        rppg_result = PipelineResult(
            pipeline_name="rPPG",
            timestamp=datetime.now(),
            confidence=0.90,
            data={'heart_rate': 72, 'spo2': 98, 'respiratory_rate': 16},
            warnings=[],
            errors=[],
            metadata={}
        )
        
        vqa_result = PipelineResult(
            pipeline_name="MedGemma_VQA",
            timestamp=datetime.now(),
            confidence=0.88,
            data={'critical_flags': [], 'overall_assessment': 'normal'},
            warnings=[],
            errors=[],
            metadata={}
        )
        
        return hear_result, rppg_result, vqa_result
    
    def test_fusion_normal_case(self, fusion_engine, sample_results):
        """Test fusion with normal vital signs"""
        hear, rppg, vqa = sample_results
        
        report = fusion_engine.fuse(hear, rppg, vqa)
        
        assert report.priority in [TriagePriority.LOW, TriagePriority.MODERATE]
        assert 0.0 <= report.confidence <= 1.0
        assert len(report.recommendations) > 0
    
    def test_fusion_critical_case(self, fusion_engine, sample_results):
        """Test fusion with critical indicators"""
        hear, rppg, vqa = sample_results
        
        # Modify to create critical condition
        rppg.data['spo2'] = 85  # Low oxygen
        vqa.data['critical_flags'] = ['Cyanosis detected']
        
        report = fusion_engine.fuse(hear, rppg, vqa)
        
        assert report.priority in [TriagePriority.CRITICAL, TriagePriority.URGENT]
        assert len(report.critical_alerts) > 0
    
    def test_weighted_confidence(self, fusion_engine):
        """Test weighted confidence calculation"""
        confidences = [
            ('hear', 0.8),
            ('rppg', 0.9),
            ('medgemma_vqa', 0.85)
        ]
        
        weighted_conf = fusion_engine._calculate_weighted_confidence(confidences)
        
        assert 0.0 <= weighted_conf <= 1.0
        assert weighted_conf > 0.8  # Should be high since all are high
