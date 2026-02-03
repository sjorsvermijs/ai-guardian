"""
Fusion Engine - Combines outputs from HeAR, rPPG, and MedGemma VQA
Generates final triage report with high-confidence medical reasoning
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from .base_pipeline import PipelineResult


class TriagePriority(Enum):
    """Triage priority levels"""
    CRITICAL = "CRITICAL"  # Immediate intervention required
    URGENT = "URGENT"      # Needs prompt attention
    MODERATE = "MODERATE"  # Can wait for standard care
    LOW = "LOW"           # Non-urgent monitoring


@dataclass
class TriageReport:
    """Final triage assessment combining all pipeline outputs"""
    timestamp: datetime
    priority: TriagePriority
    confidence: float
    summary: str
    vital_signs: Dict[str, Any]
    acoustic_indicators: Dict[str, Any]
    visual_indicators: Dict[str, Any]
    critical_alerts: List[str]
    recommendations: List[str]
    individual_results: Dict[str, PipelineResult]
    metadata: Dict[str, Any]


class FusionEngine:
    """
    Combines and analyzes results from all three pipelines
    Uses weighted voting and cross-validation to produce high-confidence triage
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize fusion engine
        
        Args:
            config: Configuration including weights and thresholds
        """
        self.config = config or {}
        
        # Pipeline weights for final decision
        self.weights = {
            'hear': self.config.get('hear_weight', 0.25),
            'rppg': self.config.get('rppg_weight', 0.35),
            'medgemma_vqa': self.config.get('vqa_weight', 0.40)
        }
        
        # Thresholds for triage priority
        self.thresholds = {
            'critical': 0.85,
            'urgent': 0.65,
            'moderate': 0.40
        }
    
    def fuse(self, 
             hear_result: PipelineResult = None,
             rppg_result: PipelineResult = None,
             vqa_result: PipelineResult = None) -> TriageReport:
        """
        Fuse results from all pipelines into a comprehensive triage report
        
        Args:
            hear_result: Result from HeAR pipeline
            rppg_result: Result from rPPG pipeline
            vqa_result: Result from MedGemma VQA pipeline
            
        Returns:
            TriageReport: Comprehensive medical assessment
        """
        timestamp = datetime.now()
        critical_alerts = []
        recommendations = []
        
        # Aggregate confidence scores
        confidences = []
        if hear_result and hear_result.confidence > 0:
            confidences.append(('hear', hear_result.confidence))
        if rppg_result and rppg_result.confidence > 0:
            confidences.append(('rppg', rppg_result.confidence))
        if vqa_result and vqa_result.confidence > 0:
            confidences.append(('medgemma_vqa', vqa_result.confidence))
        
        # Calculate weighted confidence
        overall_confidence = self._calculate_weighted_confidence(confidences)
        
        # Extract vital signs from rPPG
        vital_signs = {}
        if rppg_result:
            vital_signs = rppg_result.data.copy()
            
            # Check for abnormal vital signs
            hr = vital_signs.get('heart_rate', 0)
            rr = vital_signs.get('respiratory_rate', 0)
            spo2 = vital_signs.get('spo2', 100)
            
            if hr > 120 or hr < 50:
                critical_alerts.append(f"Abnormal heart rate: {hr} BPM")
            if rr > 30 or rr < 8:
                critical_alerts.append(f"Abnormal respiratory rate: {rr} breaths/min")
            if spo2 < 90:
                critical_alerts.append(f"Low oxygen saturation: {spo2}%")
        
        # Extract acoustic indicators from HeAR
        acoustic_indicators = {}
        if hear_result:
            acoustic_indicators = hear_result.data.copy()
            if hear_result.data.get('cough_detected'):
                critical_alerts.append("Persistent coughing detected")
        
        # Extract visual indicators from VQA
        visual_indicators = {}
        if vqa_result:
            visual_indicators = vqa_result.data.copy()
            vqa_flags = vqa_result.data.get('critical_flags', [])
            critical_alerts.extend(vqa_flags)
        
        # Cross-validate findings
        cross_validated_alerts = self._cross_validate(
            hear_result, rppg_result, vqa_result
        )
        critical_alerts.extend(cross_validated_alerts)
        
        # Determine triage priority
        priority = self._determine_priority(
            critical_alerts, 
            overall_confidence,
            vital_signs
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            priority, 
            critical_alerts,
            vital_signs
        )
        
        # Create summary
        summary = self._generate_summary(
            priority,
            len(critical_alerts),
            vital_signs
        )
        
        return TriageReport(
            timestamp=timestamp,
            priority=priority,
            confidence=overall_confidence,
            summary=summary,
            vital_signs=vital_signs,
            acoustic_indicators=acoustic_indicators,
            visual_indicators=visual_indicators,
            critical_alerts=list(set(critical_alerts)),  # Remove duplicates
            recommendations=recommendations,
            individual_results={
                'hear': hear_result,
                'rppg': rppg_result,
                'medgemma_vqa': vqa_result
            },
            metadata={
                'fusion_engine_version': '0.1.0',
                'num_pipelines_active': len(confidences),
                'weights_used': self.weights
            }
        )
    
    def _calculate_weighted_confidence(self, confidences: List[tuple]) -> float:
        """Calculate weighted average of pipeline confidences"""
        if not confidences:
            return 0.0
        
        total_weight = sum(self.weights[name] for name, _ in confidences)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(
            self.weights[name] * conf 
            for name, conf in confidences
        )
        
        return weighted_sum / total_weight
    
    def _cross_validate(self, 
                       hear_result: PipelineResult,
                       rppg_result: PipelineResult,
                       vqa_result: PipelineResult) -> List[str]:
        """Cross-validate findings across pipelines"""
        alerts = []
        
        # Example: If both VQA sees cyanosis and rPPG shows low SpO2
        if vqa_result and rppg_result:
            vqa_flags = vqa_result.data.get('critical_flags', [])
            spo2 = rppg_result.data.get('spo2', 100)
            
            if any('cyanosis' in flag.lower() for flag in vqa_flags) and spo2 < 92:
                alerts.append("VALIDATED: Hypoxemia confirmed by visual and vital signs")
        
        # Example: Respiratory distress seen visually and acoustically
        if hear_result and vqa_result:
            breathing_pattern = hear_result.data.get('breathing_pattern', 'normal')
            vqa_flags = vqa_result.data.get('critical_flags', [])
            
            if breathing_pattern != 'normal' and any('respiratory' in flag.lower() for flag in vqa_flags):
                alerts.append("VALIDATED: Respiratory distress confirmed by audio and visual")
        
        return alerts
    
    def _determine_priority(self,
                          critical_alerts: List[str],
                          confidence: float,
                          vital_signs: Dict[str, Any]) -> TriagePriority:
        """Determine triage priority based on all factors"""
        
        # Critical conditions
        critical_keywords = ['cyanosis', 'hypoxemia', 'validated']
        if any(any(keyword in alert.lower() for keyword in critical_keywords) 
               for alert in critical_alerts):
            return TriagePriority.CRITICAL
        
        # Check vital signs
        spo2 = vital_signs.get('spo2', 100)
        hr = vital_signs.get('heart_rate', 70)
        rr = vital_signs.get('respiratory_rate', 15)
        
        if spo2 < 88 or hr > 140 or hr < 40 or rr > 35:
            return TriagePriority.CRITICAL
        
        # Urgent conditions
        if len(critical_alerts) >= 2 or spo2 < 92 or hr > 120:
            return TriagePriority.URGENT
        
        # Moderate conditions
        if len(critical_alerts) >= 1 or confidence > 0.6:
            return TriagePriority.MODERATE
        
        return TriagePriority.LOW
    
    def _generate_recommendations(self,
                                 priority: TriagePriority,
                                 alerts: List[str],
                                 vital_signs: Dict[str, Any]) -> List[str]:
        """Generate medical recommendations based on findings"""
        recommendations = []
        
        if priority == TriagePriority.CRITICAL:
            recommendations.append("IMMEDIATE medical attention required")
            recommendations.append("Call emergency services (911)")
            recommendations.append("Monitor vital signs continuously")
        elif priority == TriagePriority.URGENT:
            recommendations.append("Seek medical attention within 1-2 hours")
            recommendations.append("Continue monitoring symptoms")
        elif priority == TriagePriority.MODERATE:
            recommendations.append("Schedule appointment with healthcare provider")
            recommendations.append("Monitor symptoms for changes")
        else:
            recommendations.append("Continue routine monitoring")
            recommendations.append("Consult healthcare provider if symptoms worsen")
        
        return recommendations
    
    def _generate_summary(self,
                         priority: TriagePriority,
                         num_alerts: int,
                         vital_signs: Dict[str, Any]) -> str:
        """Generate human-readable summary"""
        
        priority_text = priority.value
        hr = vital_signs.get('heart_rate', 'N/A')
        rr = vital_signs.get('respiratory_rate', 'N/A')
        spo2 = vital_signs.get('spo2', 'N/A')
        
        summary = f"Triage Priority: {priority_text}. "
        summary += f"Detected {num_alerts} concerning indicator(s). "
        summary += f"Vital signs - HR: {hr}, RR: {rr}, SpO2: {spo2}."
        
        return summary
